import logging
from collections import deque

import numpy as np
import pandas as pd
from dateutil.utils import today
import datetime as dt

from sfm_data_provider.analytics.adjustments import Adjuster, SpecialtyEtfCarryComponent
from sfm_data_provider.analytics.adjustments.dividend import DividendComponent
from sfm_data_provider.analytics.adjustments.fx_forward_carry import FxForwardCarryComponent
from sfm_data_provider.analytics.adjustments.fx_spot import FxSpotComponent
from sfm_data_provider.analytics.adjustments.repo import RepoComponent
from sfm_data_provider.analytics.adjustments.ter import TerComponent
from sfm_data_provider.analytics.adjustments.ytm import YtmComponent
from sfm_data_provider.core.holidays.holiday_manager import HolidayManager
from sfm_data_provider.core.instruments.instrument_factory import InstrumentFactory
from sfm_data_provider.interface.bshdata import BshData

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI

from user_strategy.utils.InputParamsFIQuoting import InputParamsFIQuoting
from user_strategy.utils.pricing_models.PricingModel import (
    ClusterPricingModel, DriverPricingModel,
    CreditFuturesCalendarSpreadPricingModel, CreditFuturesInterestRatePricingModel, NavPricingModel
)
from user_strategy.utils.pricing_models.PricingModelRegistry import PricingModelRegistry
from user_strategy.utils.pricing_models.IRPManager import IRPManager
from user_strategy.utils.enums import TICK_SIZE

ON_RATES = {'EUR': 'ESTRON', 'USD': 'SOFRRATE', 'GBP': 'SONIO/N', 'JPY': 'MUTKCALM', 'CHF': 'SRFXON1'}


class CreditPriceEngine(StrategyUI):
    """
    Fixed income market monitor. Responsibilities:
    - Instrument subscription: _setup_instrument_universe, _build_subscription_dict, on_market_data_setting
    - Book management: get_mid and helpers
    - Price propagation: _setup_pricing_models, calculate_theoretical_prices, _publish_prices, update_HF
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrected_returns = pd.DataFrame()
        self.end = today().date()
        self.holidays = HolidayManager()
        self.start_date = self.holidays.subtract_business_days(today(), kwargs.get('number_of_days', 10))
        self.book_mid: pd.Series | None = None
        self.input_params = InputParamsFIQuoting(kwargs)
        self._cumulative_returns = True
        self.book_storage = deque(maxlen=20)
        self.gui_redis = RedisMessaging()
        self.API = BshData(r"C:\AFMachineLearning\Libraries\MarketMonitor\etc\config\bshdata_config.yaml")
        self.market_api = self.API.market
        self.info_api = self.API.info
        self._setup_instrument_universe()
        self._setup_historical_data()
        self._setup_pricing_models()

    # ── Subscription ──────────────────────────────────────────────────────────

    def _setup_instrument_universe(self) -> None:

        self.factory = InstrumentFactory()
        dc = self.input_params.data_config
        self.etf_isins = dc.etf_isins
        self.drivers_data = dc.drivers
        self.drivers_list = self.drivers_data.index.to_list()
        self.credit_futures_contracts_data = dc.credit_futures_data
        self.credit_futures_contracts = self.credit_futures_contracts_data.index.tolist()
        self.index_drivers = dc.index_data.index.to_list()

        self.irs_data = dc.irs_data
        self.irs_contracts_list = self.irs_data.index.to_list()
        self.irp_data = dc.irp_data
        cutoff_date = max(self.credit_futures_contracts_data['EXPIRY_DATE']) + dt.timedelta(days=97)
        self.irp_manager = IRPManager(
            cutoff_date, self.irp_data,
            self.irs_data.loc[~self.irs_data.index.isin(["ESTR3M", "SOFR3M"])]
        )
        self.irp_contracts_data = self.irp_manager.get_contracts_list_data()
        self.futures_list = [d for d in self.drivers_list if self.factory.classifier.future.matches(d)]
        self.cds_list = [d for d in self.drivers_list if self.factory.classifier.cds.matches(d)]
        self.irp_contracts_list = self.irp_contracts_data.index.to_list()

        self.trading_currency = dc.trading_currency
        self.fx_list = dc.currency_exposure.columns.tolist()
        self._all_securities = (self.etf_isins + self.fx_list + self.drivers_list +
                                self.credit_futures_contracts + self.irp_contracts_list)
        self._subscription_dict = self._build_subscription_dict()

    def _build_subscription_dict(self) -> dict[str, str]:
        sub = {isin: f"{isin} IM Equity" for isin in self.etf_isins}
        for df in [self.drivers_data, self.credit_futures_contracts_data,
                   self.irp_contracts_data, self.irs_data]:
            if 'BLOOMBERG_CODE' in df.columns:
                sub.update(df['BLOOMBERG_CODE'].dropna().to_dict())
        return sub

    def on_market_data_setting(self) -> None:
        self.book_mid = pd.Series(index=self._all_securities, dtype=float)
        mgr = self.global_subscription_service

        for isin in self.etf_isins:
            mgr.subscribe_bloomberg(isin, self._subscription_dict[isin], ["BID", "ASK"], {"interval": 1})
        for iid in self.drivers_list:
            mgr.subscribe_bloomberg(iid, self._subscription_dict[iid], ["BID", "ASK"], {"interval": 1})
        for iid in self.credit_futures_contracts:
            mgr.subscribe_bloomberg(iid, self._subscription_dict[iid], ["BID", "ASK"], {"interval": 1})
        for iid in self.irp_contracts_list + self.irs_contracts_list:
            mgr.subscribe_bloomberg(iid, self._subscription_dict[iid], ["LAST_PRICE"], {"interval": 1})
        for ccy in self.fx_list:
            mgr.subscribe_bloomberg(ccy, f"{ccy} Curncy", ["BID", "ASK"], {"interval": 1})

        self._seed_initial_prices()

    def _seed_initial_prices(self) -> None:
        yesterday_price = pd.concat([
            self.historical_prices.iloc[-1],
            self.fx_prices.iloc[-1]
        ])
        for isin, price in yesterday_price.items():
            if isin in self._all_securities:
                self.market_data.update(isin, {f: price for f in self.market_data.mid_key})

    # ── Historical data ───────────────────────────────────────────────────────

    def _setup_historical_data(self) -> None:
        days = self.holidays.get_business_days(start=self.start_date, end=self.end)
        snapshot_time = dt.time(16, 45)

        self._register_instruments()
        self._fetch_market_prices(days, snapshot_time)
        self._fetch_info_data()
        self._apply_overrides()
        self._build_adjuster()

        self.corrected_return = pd.DataFrame(
            index=self.historical_prices.index,
            columns=self.historical_prices.columns,
            dtype=float,
        )

    def _register_instruments(self) -> None:
        self.index_instruments = [self.market_api.build_instrument(i, type='INDEX', autocomplete=True) for i in
                                  self.irp_contracts_list + self.irs_contracts_list]
        self.cds_instruments = [self.market_api.build_instrument(i, type='CDXINDEX', autocomplete=True) for i in
                                self.cds_list]
        self.etf_instruments = [self.market_api.build_instrument(i, type='ETP', autocomplete=True) for i in
                                self.etf_isins]
        self.future_instruments = [self.market_api.build_instrument(i, type='FUTURE', autocomplete=True) for i in
                                   self.futures_list]
        self.fx_instruments = [self.market_api.build_instrument(i, type='CURRENCYPAIR', autocomplete=True) for i in
                               self.fx_list]

        self.instruments = {
            inst.id: inst
            for inst in [*self.index_instruments,
                         *self.etf_instruments,
                         *self.cds_instruments,
                         *self.future_instruments,
                         *self.fx_instruments]
        }
        for inst in self.instruments.values():
            self.market_api.register(inst)

    def _fetch_market_prices(self, days, snapshot_time) -> None:
        index_prices = self.API.market.get_daily_repo_rates(start=self.start_date, end=self.end,
                                                            id=self.irs_contracts_list)
        future_prices = self.API.market.get_daily_future(self.start_date, self.end, id=self.futures_list,
                                                         fallbacks=[{"source": "bloomberg"}])
        cds_prices = self.API.market.get_daily_cdx(self.start_date, self.end, id=self.cds_list)
        self.etf_prices = self.API.market.get_daily_etf(
            id=self.etf_isins, start=self.start_date, end=self.end,
            snapshot_time=snapshot_time, timeout=10,
            fallbacks=[{"source": "bloomberg", "market": mkt} for mkt in ["IM", "FP", "NA"]],
        ).reindex(days)
        self.fx_prices = self.API.market.get_daily_currency(
            id=self.fx_list, start=self.start_date, end=self.end,
            snapshot_time=snapshot_time,
            fallbacks=[{"source": "bloomberg"}],
        ).reindex(days)

        self.historical_prices = pd.concat(
            [self.etf_prices, cds_prices, index_prices, future_prices], axis=1
        ).interpolate("time")

    def _fetch_info_data(self) -> None:
        self.nav = self.API.info.get_etp_fields(
            start=self.start_date, isin=self.etf_isins,
            source="oracle", fields='NAV',
            fallbacks=[{"source": "bloomberg"}],
        )
        self.ter = self.API.info.get_ter(id=self.etf_isins) / 100
        self.ytm = self.API.info.get_etp_fields(
            'ytm', isin=self.etf_isins, source="timescale",
            start=self.start_date, end=self.end,
        )
        self.dividends = self.API.info.get_dividends(id=self.etf_isins, start=self.start_date)
        self.repo_per_currency = self.market_api.get_daily_repo_rates(start=self.start_date,
                                                                      end=self.end,
                                                                      currencies=['EUR', 'USD', 'JPY', 'CHF'])

    def _apply_overrides(self) -> None:
        self.fx_composition = self.API.info.get_fx_composition(self.etf_isins, fx_fxfwrd='fx')
        self.fx_forward = self.API.info.get_fx_composition(self.etf_isins, fx_fxfwrd="fxfwrd")

        for fx in [self.fx_composition, self.fx_forward]:
            for isin, hard_coding_currency in self.input_params.fx_hard_coding.items():
                fx[isin] = hard_coding_currency
            for isin, isin_proxy in self.input_params.fx_mapping.items():
                fx.loc[isin] = fx.loc[isin_proxy]

        for isin, mapping in self.input_params.get_ytm_mapping().items():
            self.ytm[isin] = self.ytm[mapping]

        self.fx_forward_prices = self.API.market.get_daily_fx_forward(
            quoted_currency=self.fx_forward.columns.tolist(),
            start=self.start_date, end=self.end,
        )

    def _build_adjuster(self) -> None:
        self.adjuster = (
            Adjuster(self.historical_prices, instruments=self.instruments)
            .add(TerComponent(self.ter))
            .add(FxSpotComponent(self.fx_composition, self.fx_prices))
            .add(FxForwardCarryComponent(self.fx_forward, self.fx_forward_prices, "1M", self.fx_prices))
            .add(DividendComponent(self.dividends, self.etf_prices, fx_prices=self.fx_prices))
            .add(YtmComponent(self.ytm)
                 # .add(SpecialtyEtfCarryComponent(self.repo_per_currency, self.fx_prices))
                 .add(RepoComponent(self.repo_per_currency,
                                    'currency',
                                    {i.id: i.currency for i in self.future_instruments})))
        )

    # ── Book management ───────────────────────────────────────────────────────

    def get_mid(self) -> pd.Series:
        if self.book_mid is not None:
            ir_secs = self.irp_contracts_list + self.irs_contracts_list
            non_ir = [s for s in self._all_securities if s not in ir_secs]
            last_book = self.market_data.get_data_field(field=["BID", "ASK"], securities=non_ir)
            last_price_ir = self.market_data.get_data_field(field=["LAST_PRICE"], securities=ir_secs)

            bid = last_book["BID"].replace({0: np.nan})
            ask = last_book["ASK"].replace({0: np.nan})
            spread = ask / bid - 1
            if missing := spread[spread.isna()].index.tolist():
                logging.warning(f"bid is zero for {', '.join(missing)}")
            is_outlier = (bid.isna() | ask.isna() | (spread > 0.015)) & last_book.index.isin(self.etf_isins)

            self.book_mid.update(last_book[~is_outlier].mean(axis=1))
            self.book_mid.update(last_price_ir)
        else:
            self.book_mid = self.market_data.get_mid()

        with self.adjuster.live_update(self.book_mid[self._all_securities], fx_prices=self.book_mid[self.fx_list]):
            self.corrected_returns = self.adjuster.get_clean_returns(cumulative=True).T
        self.book_storage.append(self.book_mid)
        return self.book_mid

    def wait_for_book_initialization(self):
        logging.info("Checking all subscription started")
        return True

    # ── Pricing models ────────────────────────────────────────────────────────

    def _setup_pricing_models(self) -> None:
        pc = self.input_params.pricing_config
        self.cluster_correction = self._calculate_cluster_correction(pc.hedge_ratios_cluster)
        self.brothers_correction = self._calculate_cluster_correction(pc.hedge_ratios_brothers)
        self.pricing_model_registry = PricingModelRegistry()

        def reg(name, instruments, model):
            self.pricing_model_registry.register(name=name, instruments=instruments, model=model)

        reg("th live cluster price", self.etf_isins, ClusterPricingModel(
            name="th live cluster price", beta=pc.hedge_ratios_cluster,
            returns=self.corrected_returns, forecast_aggregator=pc.forecast_aggregator_cluster,
            cluster_correction=self.cluster_correction,
        ))
        reg("th live brother price", self.etf_isins, ClusterPricingModel(
            name="th live brother price", beta=pc.hedge_ratios_brothers,
            returns=self.corrected_returns, forecast_aggregator=pc.forecast_aggregator_brother,
            cluster_correction=self.brothers_correction,
        ))
        reg("th live driver price", self.etf_isins, DriverPricingModel(
            name="th live driver price", beta=pc.hedge_ratios_drivers,
            returns=self.corrected_returns, forecast_aggregator=pc.forecast_aggregator_driver,
        ))

        cf = self.credit_futures_contracts
        cf_idx = cf + self.index_drivers
        reg("th live cluster credit futures price", cf_idx, ClusterPricingModel(
            name="th live cluster credit futures price",
            beta=pc.hedge_ratios_credit_futures_cluster.loc[cf_idx],
            returns=self.corrected_returns, forecast_aggregator=pc.forecast_aggregator_cluster,
            disable_warning=True,
        ))
        reg("th live brother credit futures price", cf, ClusterPricingModel(
            name="th live brother credit futures price",
            beta=pc.hedge_ratios_credit_futures_brothers.loc[cf],
            returns=self.corrected_returns, forecast_aggregator=pc.forecast_aggregator_brother,
            disable_warning=True,
        ))

        reg(
            name="th live nav price",
            instruments=self.etf_isins,
            model=NavPricingModel(
                name="th live nav price", target_variables=self.etf_isins,
                irs_data=self.irs_data, nav_data=self.nav,
            )
        )

        cfd = self.credit_futures_contracts_data
        cfd['REGION'] = cfd['REGION'].replace("", "US")
        proxy = pd.DataFrame(index=cf, columns=["Future Proxy", "IRP", "Expiry", "Proxy Expiry"])
        proxy["Future Proxy"] = cfd['PREVIOUS_CONTRACT']
        proxy["IRP"] = cfd.merge(self.irp_data.reset_index(), on="REGION")['INSTRUMENT_ID'].to_list()
        proxy["Expiry"] = cfd['EXPIRY_DATE']
        proxy["Proxy Expiry"] = cfd['PREVIOUS_CONTRACT_EXPIRY_DATE']
        self.credit_futures_proxy = proxy

        reg("th live spread credit futures price", cf, CreditFuturesCalendarSpreadPricingModel(
            name="th live spread credit futures price", target_variables=cf,
            variables_proxy=proxy, irp_manager=self.irp_manager,
        ))
        reg("th live ir credit futures price", cf, CreditFuturesInterestRatePricingModel(
            name="th live ir credit futures price", target_variables=cf,
            variables_proxy=proxy[['IRP', 'Expiry']], irp_manager=self.irp_manager,
        ))

    def calculate_theoretical_prices(self):
        self.pricing_model_registry.calculate_theoretical_prices(self.book_mid, self.corrected_returns)

    def _publish_prices(self) -> None:
        pm = self.pricing_model_registry
        rtt = self.round_series_to_tick
        export = self.gui_redis.export_message

        for key, name, do_round, dropna in [
            ("th_live_cluster_price", "th live cluster price", False, False),  # ← usa get_prices
            ("th_live_driver_price", "th live driver price", True, False),
            ("th_live_brother_price", "th live brother price", True, False),
            ("th_live_credit_futures_cluster_price", "th live cluster credit futures price", False, False),
            ("th_live_credit_futures_brother_price", "th live brother credit futures price", False, False),
            ("th_live_credit_futures_spread_price", "th live spread credit futures price", False, True),
            ("th_live_credit_futures_ir_price", "th live ir credit futures price", False, True),
        ]:
            prices = pm.get_prices(name)
            if do_round: prices = rtt(prices, TICK_SIZE)
            if dropna:   prices = prices.dropna()
            export(key, prices, skip_if_unchanged=True)

    def update_HF(self):
        self.get_mid()
        self.calculate_theoretical_prices()
        self._publish_prices()

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _calculate_cluster_correction(cluster_betas: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        cluster_betas = cluster_betas.sort_index(axis=1).sort_index(axis=0).copy()
        for label in cluster_betas.index:
            cluster_betas.loc[label, label] = 0
        cluster_threshold = threshold / (cluster_betas != 0).sum(axis=1)
        cluster_sizes = cluster_betas.gt(cluster_threshold, axis=0).sum(axis=1) + 1
        return cluster_sizes.where(cluster_sizes == 1, (cluster_sizes - 1) / cluster_sizes)

    @staticmethod
    def round_series_to_tick(series, tick_dict, default_tick=0.001):
        ticks = np.array([tick_dict.get(idx, default_tick) for idx in series.index])
        values = series.fillna(0).values.astype(float)
        return pd.Series(np.round(np.round(values / ticks) * ticks, 10), index=series.index).fillna(0)

    def stop(self):
        pass
