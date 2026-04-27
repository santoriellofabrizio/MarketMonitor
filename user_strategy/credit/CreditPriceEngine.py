from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Set, NamedTuple

import pandas as pd
from dateutil.utils import today
import datetime as dt

from sfm_data_provider.analytics.adjustments import (
    Adjuster, TerComponent, FxSpotComponent, FxForwardCarryComponent,
    DividendComponent, YtmComponent, RepoComponent,
)
from sfm_data_provider.core.enums.instrument_types import InstrumentType
from sfm_data_provider.core.holidays.holiday_manager import HolidayManager
from sfm_data_provider.core.instruments.instruments import Instrument
from sfm_data_provider.interface.bshdata import BshData

from market_monitor.utils.book import CompositeBook, best_bid_ask
from market_monitor.utils.book_utils import SpreadEWMA

from user_strategy.strategy_templates.BasePriceEngine import BasePriceEngine
from user_strategy.utils.EtfUniverse import EtfUniverse
from user_strategy.utils.InputParamsFIQuoting import InputParamsFIQuoting
from user_strategy.utils.pricing_models.PricingModel import (
    ClusterPricingModel,
    DriverPricingModel,
    NavPricingModel,
    CreditFuturesCalendarSpreadPricingModel,
    CreditFuturesInterestRatePricingModel,
    calculate_cluster_correction,
    round_series_to_tick,
)
from user_strategy.utils.pricing_models.PricingModelRegistry import PricingModelRegistry
from user_strategy.utils.pricing_models.IRPManager import IRPManager
from user_strategy.utils.subscription_helper import SubscriptionHelper

# ── Constants ─────────────────────────────────────────────────────────────────

_RETRY_MARKETS = ("NA", "FP")
_MAX_FAILED_RATIO = 0.50
_KAFKA_MAPPING = {"IM": "ETFP", "NA": "XAMS", "FP": "XPAR"}
_ETF_MARKETS = ("IM", "FP", "NA")
_SNAPSHOT_TIME = dt.time(16, 45)
_BOOK_READY_TIME = dt.time(9, 5)

_TARGET_INSTRUMENT_TYPES = (
    "ETF", "IRS", "CDS", "STIR FUTURE", "ZCIS", "XCCY SWAP",
    "FX SPOT", "FX SWAP", "FX FORWARD", "PERPETUAL BOND FUTURE",
    "GENERIC TREASURY YIELD", "ON FINANCING RATE", "PERPETUAL CDS",
    "PERPETUAL EQUITY FUTURE", "INDEX", "IRP",
)


# ── Publish spec ──────────────────────────────────────────────────────────────

class _PublishSpec(NamedTuple):
    """Describes how a single model output is exported."""
    export_key: str
    model_name: str
    round_to_tick: bool = False
    drop_na: bool = False


_PUBLISH_SPECS: tuple[_PublishSpec, ...] = (
    _PublishSpec("th_live_cluster_price", "th live cluster price"),
    _PublishSpec("th_live_driver_price", "th live driver price", round_to_tick=True),
    _PublishSpec("th_live_brother_price", "th live brother price", round_to_tick=True),
    _PublishSpec("th_live_credit_futures_cluster_price", "th live cluster credit futures price"),
    _PublishSpec("th_live_credit_futures_brother_price", "th live brother credit futures price"),
    _PublishSpec("th_live_credit_futures_spread_price", "th live spread credit futures price", drop_na=True),
    _PublishSpec("th_live_credit_futures_ir_price", "th live ir credit futures price", drop_na=True),
)


# ── Engine ────────────────────────────────────────────────────────────────────

class CreditPriceEngine(BasePriceEngine):
    """
    Fixed income market monitor.

    Responsibilities
    ----------------
    - Instrument subscription  : _setup_instrument_universe, on_market_data_setting
    - Book management          : get_mid, _update_mids
    - Price propagation        : _setup_pricing_models, calculate_theoretical_prices, _publish_prices
    """

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(self, *args, **kwargs) -> None:
        self.activate_multicurrency: bool = False
        self.price_source: str = "kafka"
        self.book_mid: pd.Series | None = None

        self.calendar = HolidayManager()
        self.input_params = InputParamsFIQuoting(kwargs)

        self.end = self.calendar.previous_business_day(today(), "ETFP")
        self.start_date = self.calendar.subtract_business_days(
            today(), kwargs.get("number_of_days", 10), "ETFP"
        )

        self.API = BshData(
            r"C:\AFMachineLearning\Libraries\MarketMonitor\etc\config\bshdata_config.yaml"
        )
        self.db_path: str = kwargs["sql_db_fi_file"]

        self.book_filter = SpreadEWMA(**kwargs.pop("book_filter_params", {}))

        super().__init__(*args, **kwargs)

    # ── Instrument universe ───────────────────────────────────────────────────

    def _setup_instrument_universe(self) -> None:
        self._etf_universe = EtfUniverse(
            api=self.API,
            markets=list(_ETF_MARKETS),
            underlying=["FIXED INCOME", "MONEY MARKET"],
        )

        self.all_etf_isin: list[str] = self._etf_universe.isins
        self.etfs_by_market: dict = self._etf_universe.by_market
        self.currency_per_isin_market: dict = self._etf_universe.currency_per_isin_market
        self.isin_ticker: dict = self._etf_universe.get_tickers()

        self.composite_book = CompositeBook().add_filter(self.book_filter, self.all_etf_isin)

        raw_ids = self._load_instruments_from_db()
        self.all_instruments = raw_ids
        instruments = self.API.market.build_instruments(list(raw_ids), autocomplete=True)
        total = len(instruments)

        # FIX: previously each instrument was appended twice (duplicate loop body)
        self.instruments_by_type: dict[InstrumentType, list[Instrument]] = defaultdict(list)
        self.factored_instruments: list[Instrument] = []

        for i, inst in enumerate(instruments):
            self.instruments_by_type[inst.type].append(inst)
            self.factored_instruments.append(inst)
            print(
                f"\rBuilding {i + 1}/{total} | {'█' * ((i + 1) * 20 // total):20} | {(i + 1) / total:>4.0%}",
                end="", flush=True,
            )
        print()

    # ── Subscription ──────────────────────────────────────────────────────────

    def on_market_data_setting(self) -> None:
        live_sub = SubscriptionHelper(self.API, self.global_subscription_service)

        for inst in self.factored_instruments:
            match inst.type:
                case InstrumentType.ETP:
                    self._subscribe_etp(inst, live_sub)
                case InstrumentType.FUTURE:
                    self._subscribe_with_book(inst, live_sub, ["BID", "ASK"])
                case InstrumentType.CURRENCYPAIR:
                    self._subscribe_with_book(inst, live_sub, ["BID", "ASK"])
                case _:
                    self._subscribe_last_price(inst, live_sub)

    def _subscribe_etp(self, inst: Instrument, live_sub: SubscriptionHelper) -> None:
        for market in _ETF_MARKETS:
            if inst.isin not in self.etfs_by_market[market]:
                continue
            currency = self.currency_per_isin_market.get((market, inst.isin), "EUR")
            if not self.activate_multicurrency and currency != "EUR":
                continue
            book_id = f"{market}:{inst.id}:{currency}"
            live_sub.subscribe_instrument(
                inst, "bloomberg", ["BID", "ASK"],
                book_id=book_id,
                params={"option": 1},
                subscription_string=f"{inst.isin} {market} EQUITY",
                currency=currency,
                market=market,
            )
            self.composite_book.register(
                book_id=book_id, instr_id=inst.id, market=market, currency=currency
            )

    def _subscribe_with_book(
            self, inst: Instrument, live_sub: SubscriptionHelper, fields: list[str]
    ) -> None:
        live_sub.subscribe_instrument(
            inst, "bloomberg", fields,
            book_id=inst.id,
            params={"option": 1},
        )
        self.composite_book.register(book_id=inst.id, instr_id=inst.id, currency=inst.currency)

    def _subscribe_last_price(self, inst: Instrument, live_sub: SubscriptionHelper) -> None:
        live_sub.subscribe_instrument(
            inst, "bloomberg", ["LAST_PRICE"],
            book_id=inst.id,
            params={"option": 1},
        )
        self.composite_book.register(book_id=inst.id, instr_id=inst.id)

    # ── Historical data ───────────────────────────────────────────────────────

    def _setup_historical_data(self) -> None:
        self._fetch_info_data()  # must come first: populates fx_list used below
        self._fetch_market_prices()
        self._apply_overrides()
        self._build_adjuster()

        self.corrected_returns = pd.DataFrame(
            index=self.historical_prices.index,
            columns=self.historical_prices.columns,
            dtype=float,
        )

    def _fetch_info_data(self) -> None:
        self.nav = self.API.info.get_etp_fields(
            start=self.start_date, isin=self.all_etf_isin,
            source="oracle", fields="NAV",
            fallbacks=[{"source": "bloomberg"}],
        )
        self.ter = self.API.info.get_ter(
            id=self.all_etf_isin, fallbacks=[{"source": "bloomberg"}]
        ) / 100
        self.ytm = self.API.info.get_etp_fields(
            "ytm", isin=self.all_etf_isin, source="timescale",
            start=self.start_date, end=self.end,
        )
        self.dividends = self.API.info.get_dividends(
            id=self.all_etf_isin, start=self.start_date
        )
        self.repo_per_currency = self.API.market.get_daily_repo_rates(
            start=self.start_date, end=self.end,
            currencies=["EUR", "USD", "JPY", "CHF"],
        )
        self.fx_composition = self.API.info.get_fx_composition(self.all_etf_isin)
        self.fx_forward = self.API.info.get_fx_composition(
            self.all_etf_isin, fx_fxfwrd="fxfwrd"
        )
        # Build the full FX pair list once, used by both spot and forward fetches
        all_ccys = set(self.fx_composition.columns) | set(self.fx_forward.columns)
        self.fx_list = [f"EUR{ccy}" for ccy in all_ccys if ccy != "EUR"]

        self.fx_forward_prices = self.API.market.get_daily_fx_forward(
            quoted_currency=self.fx_forward.columns.tolist(),
            start=self.start_date, end=self.end,
        )
        self.reference_tick_size: dict = self.API.info.get_etp_fields(
            isin=self.all_etf_isin, fields=["REFERENCE_TICK_SIZE"], source="bloomberg",
        )["REFERENCE_TICK_SIZE"].to_dict()

    def _fetch_market_prices(self) -> None:
        """Fetch all historical price series and join them into a single DataFrame."""
        business_days = self.calendar.get_business_days(self.start_date, self.end, "ETFP")
        hist_prices: list[pd.DataFrame] = []

        for inst_type, instruments in self.instruments_by_type.items():
            ids = [i.id for i in instruments]
            match inst_type:
                case InstrumentType.INDEX:
                    hist_prices.append(
                        self.API.market.get_daily_index(self.start_date, self.end, ids)
                    )
                case InstrumentType.FUTURE:
                    hist_prices.append(
                        self.API.market.get_daily_future(self.start_date, self.end, id=ids)
                    )
                case InstrumentType.ETP:
                    self.etf_prices = self.API.market.get_daily_etf(
                        id=ids, start=self.start_date, end=self.end,
                        currency="EUR", market="EURONEXT",
                        snapshot_time=_SNAPSHOT_TIME, timeout=10,
                    )
                    hist_prices.append(self.etf_prices)
                case InstrumentType.CURRENCYPAIR:
                    self.fx_prices = self.API.market.get_daily_currency(
                        id=self.fx_list, start=self.start_date, end=self.end,
                        snapshot_time=_SNAPSHOT_TIME, fallbacks=[{"source": "bloomberg"}]
                    )
                case InstrumentType.CDXINDEX:
                    hist_prices.append(
                        self.API.market.get_daily_cdx(self.start_date, self.end, id=ids)
                    )
                case InstrumentType.SWAP:
                    hist_prices.append(
                        self.API.market.get_daily_swap(self.start_date, self.end, id=ids)
                    )

        self.historical_prices = (
            pd.concat(hist_prices, axis=1)
            .reindex(business_days)
            .interpolate("time")
        )

    def _apply_overrides(self) -> None:
        """Apply FX and YTM proxy mappings loaded from the database."""
        fx_override = self._load_fx_override()
        for fx_df in (self.fx_composition, self.fx_forward):
            for isin, proxy_isin in fx_override.items():
                fx_df.loc[isin] = fx_df.loc[proxy_isin]

        for isin, proxy_isin in self._load_ytm_override().items():
            self.ytm[isin] = self.ytm[proxy_isin]

    def _build_adjuster(self) -> None:
        future_currencies = {
            i.id: i.currency
            for i in self.instruments_by_type[InstrumentType.FUTURE]
        }
        self.adjuster = (
            Adjuster(
                self.historical_prices,
                instruments={i.id: i for i in self.factored_instruments if i.type != "CURRENCYPAIR"}
            )
            .add(TerComponent(self.ter))
            .add(FxSpotComponent(self.fx_composition, self.fx_prices))
            .add(FxForwardCarryComponent(self.fx_forward, self.fx_forward_prices, "1M", self.fx_prices))
            .add(DividendComponent(self.dividends, self.etf_prices, fx_prices=self.fx_prices))
            .add(
                YtmComponent(self.ytm).add(
                    RepoComponent(self.repo_per_currency, "currency", future_currencies)
                )
            )
        )

    # ── Database helpers ──────────────────────────────────────────────────────

    def _load_instruments_from_db(self) -> Set[str]:
        placeholders = ", ".join("?" * len(_TARGET_INSTRUMENT_TYPES))
        query = f"SELECT INSTRUMENT_ID FROM InstrumentsAnagraphic WHERE INSTRUMENT_TYPE IN ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            rows = pd.read_sql(query, conn, params=_TARGET_INSTRUMENT_TYPES)
        return {v.upper() for v in rows["INSTRUMENT_ID"]}

    def _load_credit_futures(self) -> Set[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = pd.read_sql("SELECT INSTRUMENT_ID FROM CreditFutures", conn)
        return {v.upper() for v in rows["INSTRUMENT_ID"]}

    def _load_ytm_override(self) -> dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            return (
                pd.read_sql("SELECT * FROM YasMapping", conn)
                .set_index("INSTRUMENT_ID")["MAPPING_INSTRUMENT_ID"]
                .to_dict()
            )

    def _load_fx_override(self) -> dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            return (
                pd.read_sql("SELECT * FROM FxMapping", conn)
                .set_index("INSTRUMENT_ID")["MAPPING_INSTRUMENT_ID"]
                .to_dict()
            )

    # ── Book management ───────────────────────────────────────────────────────

    def wait_for_book_initialization(self) -> bool:

        while self.global_subscription_service.get_pending_subscriptions('bloomberg'):
            return False

        for failed_sub in self.global_subscription_service.get_failed_subscriptions():
            if failed_sub['source'] == 'bloomberg':
                id = failed_sub["id"]
                try:
                    market, isin, currency = id.split(":")
                except Exception:
                    continue   #if not ETFs do not retry
                ticker = self.isin_ticker.get(isin)
                self.global_subscription_service.unsubscribe(id, 'bloomberg')
                self.global_subscription_service.subscribe_bloomberg(id=id,
                                                                     subscription_string=f"{ticker} {market} EQUITY",
                                                                     fields=["BID", "ASK"],
                                                                     params={"interval": 1})

        while not dt.datetime.today().time() > _BOOK_READY_TIME:
            return False

        return True

    def update_HF(self) -> pd.Series:
        self._update_mids()
        self._refresh_corrected_returns()
        self._update_models_returns()
        self.calculate_theoretical_prices()
        self._publish_prices()

        return self.book_mid

    def _update_mids(self) -> None:
        """Update book_mid by aggregating bid/ask per ISIN across markets via CompositeBook."""
        raw_book = self.market_data.get_data_field(field=["BID", "ASK", "LAST_PRICE"])

        fx_rates = (
            raw_book
            .loc[raw_book.index.intersection(self.fx_list), ["BID", "ASK"]]
            .mean(axis=1)
        )
        self.composite_book.update(raw_book)

        aggregated_mid = self.composite_book.agg(by=[], fx_rate=fx_rates)
        bid = aggregated_mid.get_field("BID", max)
        ask = aggregated_mid.get_field("ASK", min)
        mids = ((bid + ask) / 2).fillna(aggregated_mid.get_field("LAST_PRICE", max))

        if self.book_mid is None:
            self.book_mid = mids
        self.book_mid = mids.combine_first(self.book_mid)

    def _refresh_corrected_returns(self) -> None:
        """Recompute adjusted returns from the latest mid prices."""
        live_instruments = self.book_mid.index.intersection(self.all_instruments)
        live_fx = self.book_mid.index.intersection(self.fx_list)
        with self.adjuster.live_update(
                self.book_mid[live_instruments],
                fx_prices=self.book_mid[live_fx],
        ):
            self.corrected_returns = self.adjuster.get_clean_returns(cumulative=True).T

    def _update_models_returns(self) -> None:
        """Push the latest corrected returns into all active models."""
        for name in self.models.model_names:
            self.models.set_returns_source(name, self.corrected_returns)

    # ── Pricing models ────────────────────────────────────────────────────────

    def _setup_pricing_models(self) -> None:
        pc = self.input_params.pricing_config
        self.cluster_correction = calculate_cluster_correction(pc.hedge_ratios_cluster)
        self.brothers_correction = calculate_cluster_correction(pc.hedge_ratios_brothers)

        rs = self.corrected_returns
        self.models = PricingModelRegistry()

        self.models.register(
            name="th live cluster price",
            instruments=self.all_etf_isin,
            model=ClusterPricingModel(
                name="th live cluster price",
                beta=pc.hedge_ratios_cluster,
                returns=rs,
                forecast_aggregator=pc.forecast_aggregator_cluster,
                cluster_correction=self.cluster_correction,
            ),
            returns_source=rs,
        )
        self.models.register(
            name="th live brother price",
            instruments=self.all_etf_isin,
            model=ClusterPricingModel(
                name="th live brother price",
                beta=pc.hedge_ratios_brothers,
                returns=rs,
                forecast_aggregator=pc.forecast_aggregator_brother,
                cluster_correction=self.brothers_correction,
            ),
            returns_source=rs,
        )
        self.models.register(
            name="th live driver price",
            instruments=self.all_etf_isin,
            model=DriverPricingModel(
                name="th live driver price",
                beta=pc.hedge_ratios_drivers,
                returns=rs,
                forecast_aggregator=pc.forecast_aggregator_driver,
            ),
            returns_source=rs,
        )

    def calculate_theoretical_prices(self) -> None:
        self.models.predict_all(self.book_mid)

    def _publish_prices(self) -> None:
        export = self.publisher.gui.export_message
        for spec in _PUBLISH_SPECS:
            prices = self.models.get_prices(spec.model_name)
            if spec.round_to_tick:
                prices = round_series_to_tick(prices, self.reference_tick_size)
            if spec.drop_na:
                prices = prices.dropna()
            export(spec.export_key, prices, skip_if_unchanged=True)
