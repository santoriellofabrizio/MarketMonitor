import logging
from collections import deque
from typing import Tuple, Union

import numpy as np
import pandas as pd
from dateutil.utils import today
import datetime as dt

from market_monitor.publishers.redis_publisher import RedisMessaging
from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI
from user_strategy.utils import CustomBDay
from user_strategy.utils.pricing_models.DataFetching.PricesProviderFI import PricesProviderFI
from user_strategy.utils.InputParamsFIQuoting import InputParamsFIQuoting
from user_strategy.utils.pricing_models.PricingModel import ClusterPricingModel, DriverPricingModel, \
    CreditFuturesCalendarSpreadPricingModel, CreditFuturesInterestRatePricingModel, NavPricingModel
from user_strategy.utils.pricing_models.TheoreticalPriceManager import TheoreticalPriceManager
from user_strategy.utils.pricing_models.IRPManager import IRPManager
from user_strategy.utils.enums import TICK_SIZE


class EtfFiPriceEngine(StrategyUI):
    """
    A class for monitoring fixed income markets, inheriting functionality from StrategyUI.

    Responsibilities are separated into three areas:
    - Instrument subscription: _setup_instrument_universe, _build_subscription_dict, on_market_data_setting
    - Book management: get_mid and its helpers (_fetch_raw_book, _validate_book, _update_book_mid,
      _compute_corrected_returns)
    - Price propagation: _setup_pricing_models, calculate_theoretical_prices, _publish_prices, update_HF
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.corrected_returns: pd.DataFrame = pd.DataFrame()
        self.today: dt.date = today().date()
        self.yesterday: dt.date = (today() - CustomBDay).date()
        self.book_mid: pd.Series | None = None
        self.input_params = InputParamsFIQuoting(kwargs)
        self._cumulative_returns: bool = True
        self.book_storage: deque = deque(maxlen=self.input_params.book_storage_size)
        self.gui_redis = RedisMessaging()

        self._setup_instrument_universe()
        self._setup_pricing_models()
        self.on_start_strategy()

    # -------------------------------------------------------------------------
    # Subscription
    # -------------------------------------------------------------------------

    def _setup_instrument_universe(self) -> None:
        """Build all instrument lists, the full securities universe and the subscription dict."""
        dc = self.input_params.data_config

        self.etf_isins = dc.etf_isins
        self.drivers_data = dc.drivers
        self.drivers_list = self.drivers_data.index.to_list()

        self.credit_futures_contracts_data = dc.credit_futures_data
        self.credit_futures_contracts = self.credit_futures_contracts_data.index.tolist()
        self.index_drivers = dc.index_data.index.to_list()
        cutoff_date = max(self.credit_futures_contracts_data['EXPIRY_DATE']) + dt.timedelta(days=97)

        self.irs_data = dc.irs_data
        self.irs_contracts_list = self.irs_data.index.to_list()
        self.irp_data = dc.irp_data
        self.irp_manager = IRPManager(
            cutoff_date,
            self.irp_data,
            self.irs_data.loc[~self.irs_data.index.isin(["ESTR3M", "SOFR3M"])]
        )
        self.irp_contracts_data = self.irp_manager.get_contracts_list_data()
        self.irp_contracts_list = self.irp_contracts_data.index.to_list()

        self.trading_currency: pd.DataFrame = dc.trading_currency
        self.fx_list = dc.currency_exposure.columns.tolist()

        self._all_securities = (self.etf_isins + self.fx_list + self.drivers_list +
                                self.credit_futures_contracts + self.irp_contracts_list)
        self._subscription_dict = self._build_subscription_dict()

    def _build_subscription_dict(self) -> dict[str, str]:
        """Build Bloomberg subscription strings from instrument DataFrames.

        ETFs use the standard IM Equity format. All other instruments take
        their subscription string from the BLOOMBERG_CODE column already
        stored in each instrument DataFrame.
        FX pairs are excluded here and subscribed separately in _subscribe_fx.
        """
        sub: dict[str, str] = {}
        for isin in self.etf_isins:
            sub[isin] = f"{isin} IM Equity"
        for df in [self.drivers_data, self.credit_futures_contracts_data,
                   self.irp_contracts_data, self.irs_data]:
            if 'BLOOMBERG_CODE' in df.columns:
                sub.update(df['BLOOMBERG_CODE'].dropna().to_dict())
        return sub

    def on_market_data_setting(self) -> None:
        """Register securities and set up all Bloomberg subscriptions."""
        self.market_data.set_securities(self._all_securities, "market")
        self.book_mid = pd.Series(index=self._all_securities, dtype=float)
        bbg_sub_mgr = self.market_data.get_subscription_manager()
        self._subscribe_etf(bbg_sub_mgr)
        self._subscribe_index(bbg_sub_mgr)
        self._subscribe_futures(bbg_sub_mgr)
        self._subscribe_fx(bbg_sub_mgr)
        self._seed_initial_prices()

    def _subscribe_etf(self, bbg_sub_mgr) -> None:
        """Subscribe to Bloomberg BID/ASK for ETF instruments."""
        for isin in self.etf_isins:
            bbg_sub_mgr.subscribe_bloomberg(
                id=isin,
                subscription_string=self._subscription_dict[isin],
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

    def _subscribe_index(self, bbg_sub_mgr) -> None:
        """Subscribe to Bloomberg BID/ASK for index/driver instruments."""
        for instrument_id in self.drivers_list:
            bbg_sub_mgr.subscribe_bloomberg(
                id=instrument_id,
                subscription_string=self._subscription_dict[instrument_id],
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

    def _subscribe_futures(self, bbg_sub_mgr) -> None:
        """Subscribe to Bloomberg for credit futures (BID/ASK) and IR/IRP contracts (LAST_PRICE)."""
        for instrument_id in self.credit_futures_contracts:
            bbg_sub_mgr.subscribe_bloomberg(
                id=instrument_id,
                subscription_string=self._subscription_dict[instrument_id],
                fields=["BID", "ASK"],
                params={"interval": 1}
            )
        for instrument_id in self.irp_contracts_list + self.irs_contracts_list:
            bbg_sub_mgr.subscribe_bloomberg(
                id=instrument_id,
                subscription_string=self._subscription_dict[instrument_id],
                fields=["LAST_PRICE"],
                params={"interval": 1}
            )

    def _subscribe_fx(self, bbg_sub_mgr) -> None:
        """Subscribe to Bloomberg for FX currency pairs."""
        for currency in self.fx_list:
            bbg_sub_mgr.subscribe_bloomberg(
                id=currency,
                subscription_string=f"{currency} Curncy",
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

    def _seed_initial_prices(self) -> None:
        """Seed yesterday's closing prices into market_data to initialize the book."""
        yesterday_price = pd.concat([
            self.historical_prices.loc[self.yesterday],
            self.historical_fx.loc[self.yesterday]
        ])
        for isin, price in yesterday_price.items():
            if isin in self._all_securities:
                self.market_data.update(isin, {field: price for field in self.market_data.mid_key})

    # -------------------------------------------------------------------------
    # Historical data
    # -------------------------------------------------------------------------

    def on_start_strategy(self) -> None:
        """Fetch historical prices, initialize the prices provider and add the NAV pricing model."""
        relevant_columns = ['BLOOMBERG_CODE', 'PRICE_SOURCE_MARKET', 'MARKET_CODE']
        additional_contracts = pd.concat([
            self.credit_futures_contracts_data[relevant_columns],
            self.irp_contracts_data[relevant_columns]
        ])
        self.prices_provider = PricesProviderFI(
            etfs=self.etf_isins,
            input_params=self.input_params.data_config,
            subscription_dict=self._subscription_dict,
            instruments_to_download_eod=self.index_drivers + self.irs_contracts_list + self.irp_contracts_list,
            additional_contracts=additional_contracts,
            trading_currency=self.trading_currency
        )
        self.historical_prices: pd.DataFrame = self.prices_provider.get_hist_prices()
        self.historical_fx: pd.DataFrame = self.prices_provider.get_hist_fx_prices()
        self.irp_manager.save_historical_prices(self.historical_prices)
        self.nav_data: pd.DataFrame = self.prices_provider.get_nav_data()
        self._setup_nav_model()
        self.return_adjustments = self.prices_provider.get_adjustments(cumulative=self._cumulative_returns)

    def _setup_nav_model(self) -> None:
        """Add the NAV pricing model (must be called after historical data is loaded)."""
        self.theoretical_price_manager.add_pricing(
            dtype=float,
            name="th live nav price",
            instruments=self.etf_isins,
            model=NavPricingModel(
                name="th live nav price",
                target_variables=self.etf_isins,
                irs_data=self.irs_data,
                nav_data=self.nav_data,
            )
        )

    # -------------------------------------------------------------------------
    # Book management
    # -------------------------------------------------------------------------

    def get_mid(self) -> pd.Series:
        """
        Fetch, validate and update mid-prices, then compute corrected returns.

        Returns:
            pd.Series: Current mid-prices for all securities.
        """
        if self.book_mid is not None:
            last_book, last_price_ir = self._fetch_raw_book()
            last_valid_book = self._validate_book(last_book)
            self._update_book_mid(last_valid_book, last_price_ir)
        else:
            self.book_mid = self.market_data.get_mid()
        self._compute_corrected_returns()
        self.book_storage.append(self.book_mid)
        return self.book_mid

    def _fetch_raw_book(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Retrieve raw BID/ASK and LAST_PRICE snapshots from market_data."""
        non_ir_securities = [
            sec for sec in self._all_securities
            if sec not in self.irp_contracts_list + self.irs_contracts_list
        ]
        last_book = self.market_data.get_data_field(field=["BID", "ASK"], securities=non_ir_securities)
        last_price_ir = self.market_data.get_data_field(
            field=["LAST_PRICE"],
            securities=self.irp_contracts_list + self.irs_contracts_list
        )
        return last_book, last_price_ir

    def _validate_book(self, last_book: pd.DataFrame) -> pd.DataFrame:
        """Filter ETF outliers: zero bids, missing prices, or bid-ask spread > 1.5%.

        Returns:
            pd.DataFrame: Book with outlier rows removed.
        """
        last_bid = last_book["BID"].replace({0: np.nan})
        last_ask = last_book["ASK"].replace({0: np.nan})
        spread = last_ask / last_bid - 1
        if len(missing_book := spread[spread.isna()].index):
            logging.warning(f"bid is zero for {', '.join(missing_book)}")
        is_outlier = (
            (last_bid.isna() | last_ask.isna() | (spread > 0.015))
            & last_book.index.isin(self.etf_isins)
        )
        return last_book[~is_outlier]

    def _update_book_mid(self, last_valid_book: pd.DataFrame, last_price_ir: pd.DataFrame) -> None:
        """Update self.book_mid with validated BID/ASK mid-prices and IR last prices."""
        self.book_mid.update(last_valid_book.mean(axis=1))
        self.book_mid.update(last_price_ir)

    def _compute_corrected_returns(self) -> None:
        """Compute corrected returns: live returns + FX correction + historical adjustments."""
        self.corrected_returns = (
            self.get_live_returns()
            .add(self.get_live_fx_return_correction().T, fill_value=0)
            .add(self.return_adjustments.T, fill_value=0)
        )

    def get_live_returns(self) -> pd.Series:
        """
        Get live ETF and driver returns relative to yesterday's closing prices.

        Returns:
            pd.Series: Transposed live returns.
        """
        all_returns: pd.Series = self.book_mid / self.historical_prices - 1
        return all_returns.T

    def get_live_fx_return_correction(self) -> pd.DataFrame:
        """
        Calculate live FX return correction.

        Returns:
            pd.DataFrame: FX live correction.
        """
        fx_book: pd.Series = self.book_mid[self.input_params.data_config.currencies_EUR_ccy]
        return self.prices_provider.get_fx_correction(fx_book, cumulative=self._cumulative_returns)

    def wait_for_book_initialization(self):
        logging.info("Checking all subscription started")
        return True

    # -------------------------------------------------------------------------
    # Pricing models and price propagation
    # -------------------------------------------------------------------------

    def _setup_pricing_models(self) -> None:
        """Initialize TheoreticalPriceManager with all models except NAV (needs historical data)."""
        pc = self.input_params.pricing_config

        self.cluster_correction: pd.Series = self._calculate_cluster_correction(pc.hedge_ratios_cluster)
        self.brothers_correction: pd.Series = self._calculate_cluster_correction(pc.hedge_ratios_brothers)

        self.theoretical_price_manager = TheoreticalPriceManager()

        self.theoretical_price_manager.add_pricing(
            dtype=float,
            name="th live cluster price",
            instruments=self.etf_isins,
            model=ClusterPricingModel(
                name="th live cluster price",
                beta=pc.hedge_ratios_cluster,
                returns=self.corrected_returns,
                forecast_aggregator=pc.forecast_aggregator_cluster,
                cluster_correction=self.cluster_correction,
            )
        )
        self.theoretical_price_manager.add_pricing(
            dtype=float,
            name="th live brother price",
            instruments=self.etf_isins,
            model=ClusterPricingModel(
                name="th live brother price",
                beta=pc.hedge_ratios_brothers,
                returns=self.corrected_returns,
                forecast_aggregator=pc.forecast_aggregator_brother,
                cluster_correction=self.brothers_correction,
            )
        )
        self.theoretical_price_manager.add_pricing(
            dtype=float,
            name="th live driver price",
            instruments=self.etf_isins,
            model=DriverPricingModel(
                name="th live driver price",
                beta=pc.hedge_ratios_drivers,
                returns=self.corrected_returns,
                forecast_aggregator=pc.forecast_aggregator_driver,
            )
        )
        self.theoretical_price_manager.add_pricing(
            dtype=float,
            name="th live cluster credit futures price",
            instruments=self.credit_futures_contracts + self.index_drivers,
            model=ClusterPricingModel(
                name="th live cluster credit futures price",
                beta=pc.hedge_ratios_credit_futures_cluster.loc[
                    self.credit_futures_contracts + self.index_drivers
                ],
                returns=self.corrected_returns,
                forecast_aggregator=pc.forecast_aggregator_cluster,
                disable_warning=True,
            )
        )
        self.theoretical_price_manager.add_pricing(
            dtype=float,
            name="th live brother credit futures price",
            instruments=self.credit_futures_contracts,
            model=ClusterPricingModel(
                name="th live brother credit futures price",
                beta=pc.hedge_ratios_credit_futures_brothers.loc[self.credit_futures_contracts],
                returns=self.corrected_returns,
                forecast_aggregator=pc.forecast_aggregator_brother,
                disable_warning=True,
            )
        )

        self.credit_futures_proxy = pd.DataFrame(
            index=self.credit_futures_contracts,
            columns=["Future Proxy", "IRP"]
        )
        self.credit_futures_proxy["Future Proxy"] = self.credit_futures_contracts_data['PREVIOUS_CONTRACT']
        self.credit_futures_contracts_data['REGION'] = self.credit_futures_contracts_data['REGION'].replace("", "US")
        self.credit_futures_proxy["IRP"] = (
            self.credit_futures_contracts_data
            .merge(self.irp_data.reset_index(), left_on="REGION", right_on="REGION")
            ['INSTRUMENT_ID'].to_list()
        )
        self.credit_futures_proxy['Expiry'] = self.credit_futures_contracts_data['EXPIRY_DATE']
        self.credit_futures_proxy['Proxy Expiry'] = self.credit_futures_contracts_data['PREVIOUS_CONTRACT_EXPIRY_DATE']

        self.theoretical_price_manager.add_pricing(
            dtype=float,
            name="th live spread credit futures price",
            instruments=self.credit_futures_contracts,
            model=CreditFuturesCalendarSpreadPricingModel(
                name="th live spread credit futures price",
                target_variables=self.credit_futures_contracts,
                variables_proxy=self.credit_futures_proxy,
                irp_manager=self.irp_manager,
            )
        )
        self.theoretical_price_manager.add_pricing(
            dtype=float,
            name="th live ir credit futures price",
            instruments=self.credit_futures_contracts,
            model=CreditFuturesInterestRatePricingModel(
                name="th live ir credit futures price",
                target_variables=self.credit_futures_contracts,
                variables_proxy=self.credit_futures_proxy[['IRP', 'Expiry']],
                irp_manager=self.irp_manager,
            )
        )

    def calculate_theoretical_prices(self):
        self.theoretical_price_manager.calculate_theorical_prices(self.book_mid, self.corrected_returns)

    def _publish_prices(self) -> None:
        """Export all theoretical prices and book mid to Redis."""
        pm = self.theoretical_price_manager
        rtt = self.round_series_to_tick
        now = dt.datetime.now().isoformat()

        self.gui_redis.export_message(
            "th_live_cluster_price",
            rtt(pm.get_theoretical_prices("th live cluster price"), TICK_SIZE),
            skip_if_unchanged=True
        )
        self.gui_redis.export_message(
            "th_live_driver_price",
            rtt(pm.get_theoretical_prices("th live driver price"), TICK_SIZE),
            skip_if_unchanged=True
        )
        self.gui_redis.export_message(
            "th_live_brother_price",
            rtt(pm.get_theoretical_prices("th live brother price"), TICK_SIZE),
            skip_if_unchanged=True
        )
        self.gui_redis.export_message(
            "th_live_credit_futures_cluster_price",
            pm.get_theoretical_prices("th live cluster credit futures price"),
            skip_if_unchanged=True
        )
        self.gui_redis.export_message(
            "th_live_credit_futures_brother_price",
            pm.get_theoretical_prices("th live brother credit futures price"),
            skip_if_unchanged=True
        )
        self.gui_redis.export_message(
            "th_live_credit_futures_spread_price",
            pm.get_theoretical_prices("th live spread credit futures price").dropna(),
            skip_if_unchanged=True
        )
        self.gui_redis.export_message(
            "th_live_credit_futures_ir_price",
            pm.get_theoretical_prices("th live ir credit futures price").dropna(),
            skip_if_unchanged=True
        )
        self.gui_redis.export_message("mid", self.book_mid, skip_if_unchanged=True)
        self.gui_redis.export_message("time_now", now)

    def update_HF(self, *args, **kwargs) -> Union[dict, Tuple]:
        """
        Main update loop called periodically. Refreshes the book, recalculates
        theoretical prices and publishes them to Redis.
        """
        self.get_mid()
        self.calculate_theoretical_prices()
        self._publish_prices()

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_cluster_correction(cluster_betas: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Calculate the cluster correction factor for each subcluster.

        Returns:
            pd.Series: Series with correction factors for each ISIN.
        """
        # sort_index ensures brothers matrix is comparable with clusters matrix
        cluster_betas = cluster_betas.sort_index(axis=1).sort_index(axis=0)
        for label in cluster_betas.index:
            cluster_betas.loc[label, label] = 0
        cluster_threshold: pd.Series = threshold / (cluster_betas != 0).sum(axis=1)
        cluster_sizes = cluster_betas.gt(cluster_threshold, axis=0).sum(axis=1) + 1
        return cluster_sizes.where(cluster_sizes == 1, (cluster_sizes - 1) / cluster_sizes)

    @staticmethod
    def round_series_to_tick(series, tick_dict, default_tick=0.001):
        """Round a Series to the tick size specified per instrument."""
        ticks = np.array([tick_dict.get(idx, default_tick) for idx in series.index])
        values = series.fillna(0).values.astype(float)
        rounded_values = np.round(np.round(values / ticks) * ticks, 10)
        return pd.Series(rounded_values, index=series.index).fillna(0)

    def stop(self):
        pass
