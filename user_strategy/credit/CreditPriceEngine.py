import logging
import sqlite3
from collections import deque, defaultdict
from typing import Set, Literal
from unittest import case

import pandas as pd
from PyQt5.uic.Compiler.qtproxies import LiteralProxyClass
from dateutil.utils import today
import datetime as dt
from time import sleep as sleep_time

from questionary import autocomplete
from sfm_data_provider.analytics.adjustments import Adjuster, SpecialtyEtfCarryComponent
from sfm_data_provider.analytics.adjustments.dividend import DividendComponent
from sfm_data_provider.analytics.adjustments.fx_forward_carry import FxForwardCarryComponent
from sfm_data_provider.analytics.adjustments.fx_spot import FxSpotComponent
from sfm_data_provider.analytics.adjustments.repo import RepoComponent
from sfm_data_provider.analytics.adjustments.ter import TerComponent
from sfm_data_provider.analytics.adjustments.ytm import YtmComponent
from sfm_data_provider.core.enums.instrument_types import InstrumentType
from sfm_data_provider.core.holidays.holiday_manager import HolidayManager
from sfm_data_provider.core.instruments.instrument_factory import InstrumentFactory
from sfm_data_provider.core.instruments.instruments import Instrument
from sfm_data_provider.core.requests.subscriptions import BloombergSubscriptionBuilder
from sfm_data_provider.interface.bshdata import BshData

from market_monitor.utils.book_utils import SpreadEWMA
from user_strategy.utils.BasePriceEngine import BasePriceEngine
from user_strategy.utils.InputParamsFIQuoting import InputParamsFIQuoting
from user_strategy.utils.pricing_models.PricingModel import (
    ClusterPricingModel, DriverPricingModel,
    CreditFuturesCalendarSpreadPricingModel, CreditFuturesInterestRatePricingModel, NavPricingModel,
    calculate_cluster_correction, round_series_to_tick,
)
from user_strategy.utils.pricing_models.PricingModelRegistry import PricingModelRegistry
from user_strategy.utils.pricing_models.IRPManager import IRPManager
from user_strategy.utils.enums import TICK_SIZE

_RETRY_MARKETS = ("NA", "FP")
_MAX_FAILED_RATIO = 1 / 100
_KAFKA_MAPPING = {"IM": "ETFP", "NA": "XAMS", "FP": "XPAR"}


class CreditPriceEngine(BasePriceEngine):
    """
    Fixed income market monitor. Responsibilities:
    - Instrument subscription: _setup_instrument_universe, _build_subscription_dict, on_market_data_setting
    - Book management: get_mid and helpers
    - Price propagation: _setup_pricing_models, calculate_theoretical_prices, _publish_prices, update_HF
    """

    # ── Subscription ──────────────────────────────────────────────────────────

    def __init__(self, *args, **kwargs):

        self.book_mid: pd.Series | None = None
        self.calendar: HolidayManager = HolidayManager()
        self.end = self.calendar.previous_business_day(today(), 'ETFP')
        self.start_date = self.calendar.subtract_business_days(today(),
                                                               kwargs.get('number_of_days',
                                                                          10), 'ETFP')
        self.API = BshData(r"C:\AFMachineLearning\Libraries\MarketMonitor\etc\config\bshdata_config.yaml")
        self.db_path = kwargs["sql_db_fi_file"]
        self.book_filter = SpreadEWMA(**kwargs.pop("book_filter_params", {}))

        super().__init__(*args, **kwargs)

    def _setup_instrument_universe(self) -> None:

        self._load_etf_universe_from_oracle()

        self.all_etf_isin = list(sorted({isin for sublist in self.etfs_by_market.values() for isin in sublist}))
        self.isin_ticker = self.API.info.get_etp_fields("TICKER", isin=self.all_etf_isin)["TICKER"].to_dict()

        self.all_instruments: set[str] = self._load_instruments_from_db()

        self.instruments_by_type = defaultdict(list)
        self.factored_instruments: list[Instrument] = []

        total = len(self.all_instruments)
        instruments = self.API.market.build_instruments(
            list(self.all_instruments), autocomplete=True
        )
        for i, inst in enumerate(instruments):
            self.instruments_by_type[inst.type].append(inst)
            self.factored_instruments.append(inst)
            self.instruments_by_type[inst.type].append(inst)
            self.factored_instruments.append(inst)
            self.API.market.register(inst)
            print(f"\rBuilding {i}/{total} inst. | {'█' * (i * 20 // total):20} | {i / total:>4.0%}", end="", flush=True)

        a = 0

    def _load_etf_universe_from_oracle(self):

        etf_loader = lambda mkt: self.API.general.get(
            fields=["etp_isins"],
            segments=[mkt],
            currency="EUR",
            underlying=["FIXED INCOME", "MONEY MARKET"],
            extra_fields=["CURRENCY"],
            source="oracle")["etp_isins"]

        self.etfs_by_market = {mkt: [] for mkt in ["NA", "FP", "IM"]}
        self.currency_per_isin_market: dict[tuple, str] = {}

        for mkt in ["NA", "FP", "IM"]:
            for isin, field_dict in etf_loader(mkt).items():
                for field, value in field_dict.items():
                    if field == "CURRENCY":
                        self.currency_per_isin_market[(isin, mkt)] = value
                self.etfs_by_market[mkt].append(isin)

    def _set_fx_information(self):
        for (isin, mkt), currency in self.currency_per_isin_market.items():
            self.market_data.set_currency_for_id(f"{isin}:{mkt}", currency)

    def _setup_historical_data(self) -> None:

        snapshot_time = dt.time(16, 45)
        self._fetch_info_data()
        self._fetch_market_prices(snapshot_time)
        self._apply_overrides()
        self._build_adjuster()
        self._setup_live_book()

        self.corrected_return = pd.DataFrame(
            index=self.historical_prices.index,
            columns=self.historical_prices.columns,
            dtype=float,
        )

    def _load_instruments_from_db(self) -> Set[str]:
        with (sqlite3.connect(self.db_path) as conn):
            return set(i.upper() for i in pd.read_sql("SELECT INSTRUMENT_ID"
                                                      " FROM InstrumentsAnagraphic",
                                                      conn)["INSTRUMENT_ID"].values)

    def _load_credit_futures(self):
        with (sqlite3.connect(self.db_path) as conn):
            return set(i.upper() for i in pd.read_sql("SELECT INSTRUMENT_ID"
                                                      " FROM CreditFutures",
                                                      conn).values.flatten())

    def _load_ytm_override(self):
        with (sqlite3.connect(self.db_path) as conn):
            return pd.read_sql("SELECT * FROM YasMapping", conn).set_index("INSTRUMENT_ID")[
                "MAPPING_INSTRUMENT_ID"].to_dict()

    def _setup_live_book(self) -> None:
        from user_strategy.utils.live_book import LiveBook

        # ETF book: sub_id is "{mkt}:{isin}", instrument_id is isin
        self.live_book_etf = LiveBook(default_method="best_bid_ask")
        for mkt, etfs in self.etfs_by_market.items():
            for isin in etfs:
                self.live_book_etf.register(
                    sub_id=f"{mkt}:{isin}",
                    instr_id=isin,
                    market=mkt,
                    currency=self.currency_per_isin_market.get((isin, mkt), "EUR"),
                )
        self.live_book_etf.add_filter(self.book_filter)

        # Non-ETF book: 1-to-1 sub_id == instrument_id
        non_etf_instruments = {
            i.id: i for i in self.factored_instruments
            if i.id not in self.all_etf_isin
        }
        self.live_book = (
            LiveBook()
            .register_from_instruments(non_etf_instruments)
        )

    def _load_fx_override(self):
        with (sqlite3.connect(self.db_path) as conn):
            return pd.read_sql("SELECT * FROM FxMapping", conn).set_index("INSTRUMENT_ID")[
                "MAPPING_INSTRUMENT_ID"].to_dict()

    def on_market_data_setting(self) -> None:

        self.book_mid = pd.Series(index=self.all_instruments, dtype=float)

        self._set_fx_information()
        self._subscribe_etfs('bloomberg')
        self._subscribe_futures('bloomberg')
        self._subscribe_fx()
        self._subscribe_stale()

    def _subscribe_etfs(self, price_source: Literal['kafka', 'bloomberg']):

        for mkt, etfs in self.etfs_by_market.items():
            for isin in etfs:
                if price_source == 'bloomberg':
                    self.global_subscription_service.subscribe_bloomberg(f"{mkt}:{isin}",
                                                                         f"{isin} {mkt} EQUITY",
                                                                         ["BID", "ASK"],
                                                                         {"interval": 1})
                    self.market_data.set_currency_for_id(f"{mkt}:{isin}", "EUR") #TODO aggiugni currency bloomberg
                else:
                    self.global_subscription_service.subscribe_kafka(id=isin,
                                                                     symbol_filter=isin,
                                                                     topic=f"COALESCENT_DUMA.{_KAFKA_MAPPING[mkt]}.BookBest",
                                                                     fields_mapping={
                                                                         "BID": "bidBestLevel.price",
                                                                         "ASK": "askBestLevel.price"})

    def _subscribe_futures(self, price_source: Literal['kafka', 'bloomberg']):
        for inst in self.instruments_by_type[InstrumentType.FUTURE]:
            if price_source == 'bloomberg':
                self.global_subscription_service.subscribe_bloomberg(inst.id,
                                                                     f"{inst.root}A {inst.suffix}",
                                                                     ["BID", "ASK"],
                                                                     {"interval": 1})
            else:
                # self.global_subscription_service.subscribe_kafka(id=inst.id,
                #                                                  symbol_filter=isin,
                #                                                  topic="COALESCENT_DUMA.XEUR.BookBest",
                #                                                  fields_mapping={
                #                                                      "BID": "bidBestLevel.price",
                #                                                      "ASK": "askBestLevel.price"})
                raise NotImplementedError

    def _subscribe_fx(self):
        for fx in self.fx_list:
            self.global_subscription_service.subscribe_bloomberg(fx,
                                                                 f"{fx} Curncy",
                                                                 ["BID", "ASK"],
                                                                 {"interval": 1})

    def _subscribe_stale(self):
        bbg_subscription = BloombergSubscriptionBuilder.build_subscription
        last_price_sub_class = [InstrumentType.SWAP, InstrumentType.INDEX, InstrumentType.CDXINDEX]
        for sec in [instr for instr in self.factored_instruments if instr.type in last_price_sub_class]:
            self.global_subscription_service.subscribe_bloomberg(sec.id,
                                                                 bbg_subscription(sec),
                                                                 ["LAST_PRICE"])

    # ── Historical data ───────────────────────────────────────────────────────

    def _fetch_market_prices(self, snapshot_time) -> None:
        hist_prices = []
        for type, inst in self.instruments_by_type.items():
            ids = [i.id for i in inst]
            match type:
                case InstrumentType.INDEX:
                    hist_prices.append(self.API.market.get_daily_index(self.start_date, self.end, ids))

                case InstrumentType.FUTURE:
                    hist_prices.append(self.API.market.get_daily_future(self.start_date, self.end, id=ids,
                                                                        # fallbacks=[{"source": "bloomberg"}]
                                                                        ))
                case InstrumentType.ETP:

                    self.etf_prices = self.API.market.get_daily_etf(
                        id=ids, start=self.start_date, end=self.end, currency='EUR', market="EURONEXT",
                        snapshot_time=snapshot_time, timeout=10,
                        # fallbacks=[{"source": "bloomberg", "market": mkt} for mkt in ["IM", "FP", "NA"]],
                    )
                    hist_prices.append(self.etf_prices)

                case InstrumentType.CURRENCYPAIR:
                    self.fx_prices = self.API.market.get_daily_currency(
                        id=self.fx_list, start=self.start_date, end=self.end,
                        snapshot_time=snapshot_time,
                        # fallbacks=[{"source": "bloomberg"}]
                        )
                    hist_prices.append(self.fx_prices)

                case InstrumentType.CDXINDEX:
                    hist_prices.append(self.API.market.get_daily_cdx(self.start_date, self.end, id=ids))

                case InstrumentType.SWAP:
                    hist_prices.append(self.API.market.get_daily_swap(self.start_date, self.end, id=ids))

        self.historical_prices = (pd.concat(hist_prices, axis=1)
                                  .reindex(self.calendar.get_business_days(self.start_date, self.end, 'ETFP'))
                                  .interpolate("time"))

    def _fetch_info_data(self) -> None:

        self.nav = self.API.info.get_etp_fields(
            start=self.start_date, isin=self.all_etf_isin,
            source="oracle", fields='NAV',
            fallbacks=[{"source": "bloomberg"}],
        )
        self.ter = self.API.info.get_ter(id=self.all_etf_isin, fallbacks=[{"source": "bloomberg"}]) / 100
        self.ytm = self.API.info.get_etp_fields(
            'ytm', isin=self.all_etf_isin, source="timescale",
            start=self.start_date, end=self.end,
        )
        self.dividends = self.API.info.get_dividends(id=self.all_etf_isin, start=self.start_date)
        self.repo_per_currency = self.API.market.get_daily_repo_rates(start=self.start_date,
                                                                      end=self.end,
                                                                      currencies=['EUR', 'USD', 'JPY', 'CHF'])

        self.fx_composition = self.API.info.get_fx_composition(self.all_etf_isin, fx_fxfwrd='fx')
        self.fx_forward = self.API.info.get_fx_composition(self.all_etf_isin, fx_fxfwrd="fxfwrd")

        self.fx_list = set(f"EUR{ccy}" for ccy in self.fx_composition.columns.to_list()
                                                  + self.fx_forward.columns.to_list())

        self.fx_forward_prices = self.API.market.get_daily_fx_forward(
            quoted_currency=self.fx_forward.columns.tolist(),
            start=self.start_date, end=self.end,
        )

    def _apply_overrides(self) -> None:
        self.fx_composition = self.API.info.get_fx_composition(self.all_etf_isin, fx_fxfwrd='fx')
        self.fx_forward = self.API.info.get_fx_composition(self.all_etf_isin, fx_fxfwrd="fxfwrd")

        for fx in [self.fx_composition, self.fx_forward]:

            for isin, isin_proxy in self._load_fx_override().items():
                fx.loc[isin] = fx.loc[isin_proxy]

        for isin, mapping in self._load_ytm_override().items():
            self.ytm[isin] = self.ytm[mapping]

    def _build_adjuster(self) -> None:
        self.adjuster = (
            Adjuster(self.historical_prices, instruments={i.id: i for i in self.factored_instruments})
            .add(TerComponent(self.ter))
            .add(FxSpotComponent(self.fx_composition, self.fx_prices))
            .add(FxForwardCarryComponent(self.fx_forward, self.fx_forward_prices, "1M", self.fx_prices))
            .add(DividendComponent(self.dividends, self.etf_prices, fx_prices=self.fx_prices))
            .add(YtmComponent(self.ytm)
                 # .add(SpecialtyEtfCarryComponent(self.repo_per_currency, self.fx_prices))
                 .add(RepoComponent(self.repo_per_currency,
                                    'currency',
                                    {i.id: i.currency for i in self.instruments_by_type[InstrumentType.FUTURE]})))
        )

    # ── Book management ───────────────────────────────────────────────────────

    def get_mid(self) -> pd.Series:
        self._update_etf_mids()
        self._update_non_etf_mids()
        self._refresh_corrected_returns()
        self._update_models_returns()
        return self.book_mid

    # ── helpers ───────────────────────────────────────────────────────────────────

    def _update_etf_mids(self) -> None:
        """Aggiorna book_mid per gli ETF aggregando bid/ask per ISIN via LiveBook."""
        raw_book = self.market_data.get_data_field(field=["BID", "ASK"])
        etf_sub_ids = [f"{mkt}:{isin}"
                       for mkt, etfs in self.etfs_by_market.items()
                       for isin in etfs]
        self.live_book_etf.update(raw_book.reindex(etf_sub_ids))
        self.book_mid.update(self.live_book_etf.agg(by=[]).get_data().as_series())

    def _update_non_etf_mids(self) -> None:
        """Aggiorna book_mid per i non-ETF usando mid oppure LAST_PRICE come fallback."""
        last_book = self.market_data.get_data_field(field=["BID", "ASK", "LAST_PRICE"])
        non_etfs = last_book.loc[~last_book.index.isin(self.all_etf_isin)]
        self.live_book.update(non_etfs[["BID", "ASK"]])
        non_etfs_mid = self.live_book.agg(by=[]).get_data().as_series().combine_first(non_etfs["LAST_PRICE"])
        self.book_mid.update(non_etfs_mid)

    def _refresh_corrected_returns(self) -> None:
        """Ricalcola i rendimenti corretti (adjusted) a partire dai prezzi mid aggiornati."""
        with self.adjuster.live_update(self.book_mid[self.all_instruments], fx_prices=self.book_mid[self.fx_list]):
            self.corrected_returns = self.adjuster.get_clean_returns(cumulative=True).T

    def _update_models_returns(self) -> None:
        """Aggiorna la sorgente dei rendimenti per tutti i modelli attivi."""
        for name in self.models.model_names:
            self.models.set_returns_source(name, self.corrected_returns)

    def wait_for_book_initialization(self) -> bool:
        return dt.datetime.today().time() > dt.time(9, 5) and self._initialize_bloomberg_subscriptions()

    def _initialize_bloomberg_subscriptions(self) -> bool:
        self._wait_pending_subscriptions()
        self._retry_failed_subscriptions()
        return self._is_failure_rate_acceptable(self._collect_bad_sec_isins())

    def _wait_pending_subscriptions(self) -> None:
        while self.market_data.get_pending_subscriptions("bloomberg"):
            sleep_time(1)

    def _retry_failed_subscriptions(self) -> None:
        failed = {s.get("id") for s in self.global_subscription_service.get_failed_subscriptions() if
                  s.get("id") in self.all_etf_isin}
        for sub in failed:
            self.global_subscription_service.unsubscribe(sub, 'bloomberg')
        for market in _RETRY_MARKETS:
            for isin in failed:
                self.global_subscription_service.subscribe_bloomberg(isin, f"{isin} {market} EQUITY", ["BID", "ASK"])
            sleep_time(5)

    def _collect_bad_sec_isins(self) -> list[str]:
        return [s.get("id") for s in self.global_subscription_service.get_failed_subscriptions() if
                s.get("last_error") == "BAD_SEC"]

    def _is_failure_rate_acceptable(self, failed: list[str]) -> bool:
        return bool(self.instruments_list) and len(failed) / len(self.instruments_list) < _MAX_FAILED_RATIO

    # ── Pricing models ────────────────────────────────────────────────────────

    def _setup_pricing_models(self) -> None:
        pc = self.input_params.pricing_config
        self.cluster_correction = calculate_cluster_correction(pc.hedge_ratios_cluster)
        self.brothers_correction = calculate_cluster_correction(pc.hedge_ratios_brothers)
        self.models = PricingModelRegistry()

        rs = self.corrected_returns

        def reg(name, instruments, model):
            self.models.register(name=name, instruments=instruments, model=model, returns_source=rs)

        reg("th live cluster price", self.all_etf_isin, ClusterPricingModel(
            name="th live cluster price", beta=pc.hedge_ratios_cluster,
            returns=rs, forecast_aggregator=pc.forecast_aggregator_cluster,
            cluster_correction=self.cluster_correction,
        ))
        reg("th live brother price", self.all_etf_isin, ClusterPricingModel(
            name="th live brother price", beta=pc.hedge_ratios_brothers,
            returns=rs, forecast_aggregator=pc.forecast_aggregator_brother,
            cluster_correction=self.brothers_correction,
        ))
        reg("th live driver price", self.all_etf_isin, DriverPricingModel(
            name="th live driver price", beta=pc.hedge_ratios_drivers,
            returns=rs, forecast_aggregator=pc.forecast_aggregator_driver,
        ))

        cf = self._load_credit_futures()

        # reg("th live cluster credit futures price", cf_idx, ClusterPricingModel(
        #     name="th live cluster credit futures price",
        #     beta=pc.hedge_ratios_credit_futures_cluster.loc[cf_idx],
        #     returns=rs, forecast_aggregator=pc.forecast_aggregator_cluster,
        #     disable_warning=True,
        # ))
        # reg("th live brother credit futures price", cf, ClusterPricingModel(
        #     name="th live brother credit futures price",
        #     beta=pc.hedge_ratios_credit_futures_brothers.loc[cf],
        #     returns=rs, forecast_aggregator=pc.forecast_aggregator_brother,
        #     disable_warning=True,
        # ))
        #
        # reg(
        #     name="th live nav price",
        #     instruments=self.etf_list,
        #     model=NavPricingModel(
        #         name="th live nav price", target_variables=self.etf_list,
        #         irs_data=self.irs_data, nav_data=self.nav,
        #     )
        # )
        #
        # cfd = self.credit_futures_contracts_data
        # cfd['REGION'] = cfd['REGION'].replace("", "US")
        # proxy = pd.DataFrame(index=cf, columns=["Future Proxy", "IRP", "Expiry", "Proxy Expiry"])
        # proxy["Future Proxy"] = cfd['PREVIOUS_CONTRACT']
        # proxy["IRP"] = cfd.merge(self.irp_data.reset_index(), on="REGION")['INSTRUMENT_ID'].to_list()
        # proxy["Expiry"] = cfd['EXPIRY_DATE']
        # proxy["Proxy Expiry"] = cfd['PREVIOUS_CONTRACT_EXPIRY_DATE']
        # self.credit_futures_proxy = proxy
        #
        # reg("th live spread credit futures price", cf, CreditFuturesCalendarSpreadPricingModel(
        #     name="th live spread credit futures price", target_variables=cf,
        #     variables_proxy=proxy, irp_manager=self.irp_manager,
        # ))
        # reg("th live ir credit futures price", cf, CreditFuturesInterestRatePricingModel(
        #     name="th live ir credit futures price", target_variables=cf,
        #     variables_proxy=proxy[['IRP', 'Expiry']], irp_manager=self.irp_manager,
        # ))

    def calculate_theoretical_prices(self):
        self.models.predict_all(self.book_mid)

    def _publish_prices(self) -> None:
        export = self.publisher.gui.export_message

        for key, name, do_round, dropna in [
            ("th_live_cluster_price", "th live cluster price", False, False),
            ("th_live_driver_price", "th live driver price", True, False),
            ("th_live_brother_price", "th live brother price", True, False),
            ("th_live_credit_futures_cluster_price", "th live cluster credit futures price", False, False),
            ("th_live_credit_futures_brother_price", "th live brother credit futures price", False, False),
            ("th_live_credit_futures_spread_price", "th live spread credit futures price", False, True),
            ("th_live_credit_futures_ir_price", "th live ir credit futures price", False, True),
        ]:
            prices = self.models.get_prices(name)
            if do_round: prices = round_series_to_tick(prices, TICK_SIZE)
            if dropna:   prices = prices.dropna()
            export(key, prices, skip_if_unchanged=True)

    def update_HF(self):
        self.get_mid()
        self.calculate_theoretical_prices()
        self._publish_prices()
