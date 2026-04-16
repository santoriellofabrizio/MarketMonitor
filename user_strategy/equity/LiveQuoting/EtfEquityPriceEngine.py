import os
import sqlite3
import time as sleep_time

from collections import deque
from contextlib import contextmanager
from datetime import datetime, time, date
import numpy as np
import pandas as pd
from dateutil.utils import today

from sfm_data_provider.analytics.adjustments.adjuster import Adjuster
from sfm_data_provider.analytics.adjustments.dividend import DividendComponent
from sfm_data_provider.analytics.adjustments.fx_forward_carry import FxForwardCarryComponent
from sfm_data_provider.analytics.adjustments.fx_spot import FxSpotComponent
from sfm_data_provider.analytics.adjustments.ter import TerComponent
from sfm_data_provider.core.holidays.holiday_manager import HolidayManager


from datetime import datetime, time
from typing import Optional
import logging

from sfm_data_provider.interface.bshdata import BshData

from market_monitor.strategy.strategy_ui.StrategyUI import StrategyUI
from user_strategy.equity.LiveQuoting.InputParamsQuoting import InputParamsQuoting
from user_strategy.equity.LiveQuoting.price_publisher import PricePublisherHub
from user_strategy.equity.LiveQuoting.pricing_engine import PricingModelRegistry
from user_strategy.equity.LiveQuoting.utils import filter_outliers, round_series_to_tick
from user_strategy.utils.pricing_models.AggregationFunctions import EwmaOutlier

from user_strategy.utils.pricing_models.PricingModel import ClusterPricingModel
from user_strategy.utils.bloomberg_subscription_utils.SubscriptionManager import SubscriptionManager

logger = logging.getLogger(__name__)

DB_ANAGRPHIC_PATH = r"V:\EquityETF\etf_equity_anagraphic_db.sqlite"


class EtfEquityPriceEngine(StrategyUI):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.logger = logger
        self.db_path = kwargs.get("path_db", None)
        self._init_universe(kwargs)
        self.publisher = PricePublisherHub.from_config(kwargs, self.isin_to_ticker)
        self._init_historical_data(kwargs)
        self._init_bloomberg(kwargs)
        self.alpha: float = 1.0

    # =========================================================================
    # Inizializzazione
    # =========================================================================

    def _init_universe(self, kwargs: dict) -> None:
        """Carica l'universo ETF, le valute e i mapping ISIN/ticker."""
        self.number_of_days = kwargs.get("number_of_days", 5)
        self.number_of_days_intraday = kwargs.get("number_of_days_intraday", 3)
        self.failed_isin = []

        self.API = BshData(config_path=kwargs.get("bshdata_config_path"))

        isins_etf_equity = self.API.general.get(
            fields=["etp_isins"], segments=["IM"], currency="EUR",
            underlying="EQUITY", source="oracle",
        )["etp_isins"]

        self.reference_tick_size = self.API.info.get_etp_fields(
            isin=isins_etf_equity, fields=["REFERENCE_TICK_SIZE"], source="bloomberg",
        )["REFERENCE_TICK_SIZE"].to_dict()

        db_path = kwargs["path_db"]
        with sqlite3.connect(db_path) as conn:
            self.isin_to_ticker = pd.read_sql(
                "SELECT ISIN, TICKER FROM vw_isin_ticker", conn
            ).set_index("ISIN")["TICKER"].to_dict()
            self.active_isin = pd.read_sql(
                "SELECT ISIN, MARKET_STATUS FROM market_status", conn
            ).set_index("ISIN")["MARKET_STATUS"].to_dict()

        self.currencies = [
            f"EUR{c}" for c in
            ['USD', 'GBP', 'CHF', 'AUD', 'DKK', 'HKD', 'NOK',
             'PLN', 'SEK', 'CNY', 'JPY', 'CNH', 'CAD', 'INR', 'BRL']
        ]

        self.ticker_to_isin = {ticker: isin for isin, ticker in self.isin_to_ticker.items()}
        self.input_params = InputParamsQuoting(**kwargs)

        # Filtra ETF attivi dall'universo beta
        self.etfs = self._filter_active_etps(kwargs.get("isin_list", []))

        self.instruments = list({*self.etfs, *self.currencies})
        self._isins_etf_equity = list(isins_etf_equity)

        # Stato runtime
        self.mid_eur: Optional[pd.Series] = None
        self.book_mid_threshold = 0.5
        self.book_eur: pd.DataFrame = pd.DataFrame()
        self.book_storage: deque = deque(maxlen=3)
        self.bloomberg_subscription_manager = SubscriptionManager(self.etfs,
                                                                  kwargs.get('bloomberg_subscription_config_path'))
        self.position: Optional[pd.Series] = None
        self.return_to_publish: list = [1, 2, 3, 4, 5, 6, 7, 8]
        self.today = pd.Timestamp.today().normalize()
        self.yesterday = HolidayManager().previous_business_day(self.today, market='ETFP')

    def _filter_active_etps(self, isin_list: list) -> list:
        """
        Filtra gli ETF attivi dall'universo delle matrici beta.

        Args:
            isin_list: Lista opzionale di ISIN da includere (whitelist)

        Returns:
            Lista ordinata di ETF attivi
        """
        raw_etfs = list({
            *self.input_params.beta_cluster.index,
            *self.input_params.beta_cluster.columns,
            *self.input_params.beta_cluster_index.index,
            *self.input_params.beta_cluster_index.columns,
        })

        etp_type = self.API.info.get_etp_fields(isin=raw_etfs,
                                                fields="ETP_TYPE")["ETP_TYPE"].to_dict()

        leverage = self.API.info.get_etp_fields(isin=raw_etfs, source='bloomberg',
                                                fields="FUND_LEVERAGE")["FUND_LEVERAGE"].to_dict()

        valid_etfs = []
        for etf in raw_etfs:
            if isin_list and etf not in isin_list:
                continue
            if self.active_isin.get(etf, "NOT_PRESENT") != "ACTV":
                logger.warning(f"{etf} ({self.ticker_to_isin.get(etf)}) is not ACTV.")
                continue

            if etp_type.get(etf, "ETP") != "ETF":
                logger.warning(f"{etf} ({self.ticker_to_isin.get(etf)}) is not an ETP.")
                continue

            if leverage.get(etf, "N") == "Y":
                logger.warning(f"{etf} ({self.ticker_to_isin.get(etf)}) is leveraged.")
                continue

            valid_etfs.append(etf)

        return sorted(valid_etfs)

    def _init_historical_data(self, kwargs: dict) -> None:
        """Carica prezzi storici, adjusters e modelli di pricing."""

        self.holidays = HolidayManager()
        start = self.holidays.subtract_business_days(today(), self.number_of_days)
        start_intraday = self.holidays.subtract_business_days(today(), self.number_of_days_intraday)
        end = self.holidays.previous_business_day(today())
        days = self.holidays.get_business_days(start=start, end=end)
        snapshot_time = time(16, 45)

        fx_composition = self.API.info.get_fx_composition(
            self.etfs, fx_fxfwrd='fx', reference_date=date(2026, 4, 2))
        fx_forward = self.API.info.get_fx_composition(
            self.etfs, fx_fxfwrd="fxfwrd", reference_date=date(2026, 4, 2))

        for isin, isin_proxy in kwargs.get("fx_mapping", {}).items():
            fx_composition.loc[isin] = fx_composition.loc[isin_proxy]

        self.fx_prices = self.API.market.get_daily_currency(
            id=self.currencies, start=start, end=end, snapshot_time=snapshot_time,
            fallbacks=[{"source": "bloomberg"}],
        ).reindex(days)

        self.etf_prices = self.API.market.get_daily_etf(
            id=self.etfs, start=start, end=end, snapshot_time=snapshot_time, timeout=10,
            fallbacks=[{"source": "bloomberg", "market": mkt} for mkt in ["IM", "FP", "NA"]],
        ).reindex(days)

        self.fx_prices_intraday = filter_outliers(
            self.API.market.get_intraday_fx(
                id=self.currencies, start=start_intraday, end=end, frequency="15m",
                fallbacks=[{"source": "bloomberg"}])
            .between_time("10:00", "17:00"))

        self.etf_prices_intraday = filter_outliers(
            self.API.market.get_intraday_etf(
                id=self.etfs, start=start_intraday, end=end, frequency="15m",
                source='timescale',
                fallbacks=[{"source": "bloomberg", "market": mkt} for mkt in ["IM", "FP", "NA"]])
            .between_time("10:00", "17:00"), name="etf_intraday")

        self.etf_prices = self.etf_prices.interpolate("time")
        self.etf_prices_intraday = self.etf_prices_intraday.interpolate("time")

        fx_forward_prices = self.API.market.get_daily_fx_forward(
            quoted_currency=fx_forward.columns.tolist(), start=start, end=end)
        dividends = self.API.info.get_dividends(id=self.etfs, start=start)
        ter = self.API.info.get_ter(id=self.etfs) / 100

        self.adjuster = (
            Adjuster(self.etf_prices)
            .add(TerComponent(ter))
            .add(FxSpotComponent(fx_composition, self.fx_prices))
            .add(FxForwardCarryComponent(fx_forward, fx_forward_prices, "1M", self.fx_prices))
            .add(DividendComponent(dividends, self.etf_prices, fx_prices=self.fx_prices))
        )
        self.intraday_adjuster = (
            Adjuster(self.etf_prices_intraday)
            .add(FxSpotComponent(fx_composition, self.fx_prices_intraday))
            .add(DividendComponent(dividends, self.etf_prices_intraday,
                                   fx_prices=self.fx_prices_intraday))
        )

        self.corrected_return = pd.DataFrame(
            index=self.etf_prices.index, columns=self.etf_prices.columns, dtype=float)
        self.corrected_return_intraday = pd.DataFrame(
            index=self.etf_prices_intraday.index, columns=self.etf_prices_intraday.columns,
            dtype=float)

        # Modelli di pricing via registry
        beta_cluster = self._prepare_beta_matrix(self.input_params.beta_cluster)
        beta_cluster_index = self._prepare_beta_matrix(self.input_params.beta_cluster_index)

        self.models = PricingModelRegistry()
        # self.models.register("cluster", ClusterPricingModel(
        #     beta=beta_cluster,
        #     returns=self.corrected_return,
        #     forecast_aggregator=self.input_params.forecast_aggregator_cluster,
        #     cluster_correction=self._calculate_cluster_correction(beta_cluster, 0),
        #     name="theoretical_live_cluster_price",
        # ), self.corrected_return)

        self.models.register("intraday", ClusterPricingModel(
            beta=beta_cluster,
            returns=self.corrected_return_intraday,
            forecast_aggregator=EwmaOutlier(**self.kwargs['pricing']['cluster']['ewma_outlier']),
            cluster_correction=self._calculate_cluster_correction(beta_cluster, 0),
            name="theoretical_live_cluster_price",
        ), self.corrected_return_intraday)

        # self.models.register("index_cluster", ClusterPricingModel(
        #     beta=beta_cluster_index,
        #     returns=self.corrected_return,
        #     forecast_aggregator=self.input_params.forecast_aggregator_cluster,
        #     cluster_correction=self._calculate_cluster_correction(beta_cluster_index, 0),
        #     name="theoretical_live_index_cluster_price",
        # ), self.corrected_return)

    def _init_bloomberg(self, kwargs: dict) -> None:
        """Configura la sottoscrizione Bloomberg e pubblica i ritorni storici iniziali."""
        self.all_etf_plus_securities = list(set(self.currencies + self._isins_etf_equity))
        self.bloomberg_subscription_config_path = kwargs.get("bloomberg_subscription_config_path")
        logger.info("=" * 70)
        logger.info("Price analytics: http://localhost:3000/d/etf-price-monitor-v2/")
        logger.info("=" * 70)

        self.publisher.publish_static_returns(self.adjuster, self.return_to_publish)

    # =========================================================================
    # Command handling (via Redis pub/sub — task "command_listener")
    # Usage: redis-cli PUBLISH engine:commands '{"action": "reload_beta"}'
    # =========================================================================

    # EtfEquityPriceEngine.on_command — sostituire il metodo esistente

    def on_command(self, action: str, payload: dict) -> None:
        """Gestisce comandi ricevuti via Redis pub/sub.

        Esempi:
            redis-cli PUBLISH engine:commands '{"action": "reload_beta"}'
            redis-cli PUBLISH engine:commands '{"action": "update_forecaster", "model": "cluster", "type": "ewma_outlier", "params": {"halflife": 3, "outlier_std": 2.5}}'
            redis-cli PUBLISH engine:commands '{"action": "update_forecaster", "model": "all", "type": "trimmed_mean", "params": {"perc_outlier": 0.1}}'
        """
        if action == "reload_beta":
            logger.warning("Beta reload triggered via command channel")
            self._reload_beta_matrices()
            self.models.predict_all(self.mid_eur)
            logger.warning("Beta reload completed successfully")

        elif action == "set_alpha":
            try:
                val = float(payload.get("value", 1.0))
                self.alpha = max(0.0, min(1.0, val))
                logger.warning(f"Alpha updated to {self.alpha:.2f}")
            except (TypeError, ValueError) as e:
                logger.error(f"Invalid alpha value: {e}")

        elif action == "set_outlier_std":
            try:
                from user_strategy.utils.pricing_models.AggregationFunctions import EwmaOutlier
                val = float(payload.get("value", 3.0))
                if val <= 0:
                    raise ValueError("outlier_std must be positive")
                updated = []
                for name in self.models.model_names:
                    entry = self.models._entries.get(name)
                    if entry and hasattr(entry.model, "forecast_aggregator"):
                        fc = entry.model.forecast_aggregator
                        if isinstance(fc, EwmaOutlier):
                            fc.outlier_threshold = val
                            updated.append(name)
                logger.warning(f"outlier_std updated to {val} on models: {updated}")
            except (TypeError, ValueError) as e:
                logger.error(f"Invalid outlier_std value: {e}")

        elif action == "set_halflife":
            try:
                from user_strategy.utils.pricing_models.AggregationFunctions import EwmaOutlier, Ewma
                val = float(payload.get("value", 10.0))
                if val <= 0:
                    raise ValueError("halflife must be positive")
                updated = []
                for name in self.models.model_names:
                    entry = self.models._entries.get(name)
                    if entry and hasattr(entry.model, "forecast_aggregator"):
                        fc = entry.model.forecast_aggregator
                        if isinstance(fc, (EwmaOutlier, Ewma)):
                            fc.halflife = val
                            updated.append(name)
                logger.warning(f"halflife updated to {val} on models: {updated}")
            except (TypeError, ValueError) as e:
                logger.error(f"Invalid halflife value: {e}")

        elif action == "debug_isin":
            self._handle_debug_isin(payload)

        elif action == "stop":
            self.stop()

        else:
            logger.warning(f"Unknown command: '{action}'")

    def _handle_update_forecaster(self, payload: dict) -> None:
        """Costruisce e applica un nuovo ForecastAggregator dai parametri del payload."""
        from user_strategy.utils.pricing_models.AggregationFunctions import forecast_aggregation

        forecaster_type = payload.get("type")
        params = payload.get("params", {})
        model_name = payload.get("model", "all")  # "all" aggiorna tutti i modelli

        if forecaster_type not in forecast_aggregation:
            logger.error(
                f"Unknown forecaster type '{forecaster_type}'. "
                f"Valid options: {list(forecast_aggregation.keys())}"
            )
            return

        try:
            forecaster = forecast_aggregation[forecaster_type](**params)
            logger.warning(F"Forecaster type '{forecaster_type}' UPDATED!")
        except TypeError as e:
            logger.error(f"Invalid params for '{forecaster_type}': {e}")
            return

        target_models = self.models.model_names if model_name == "all" else [model_name]

        for name in target_models:
            if name not in self.models:
                logger.warning(f"Model '{name}' not found, skipping")
                continue
            self.models.update_forecaster(name, forecaster)

        logger.warning(
            f"Forecaster updated → type={forecaster_type}, "
            f"params={params}, models={target_models}"
        )

    def _handle_debug_isin(self, payload: dict) -> None:
        """Genera un file Excel di debug per un singolo ISIN.

        Esempio:
            redis-cli PUBLISH engine:commands '{"action": "debug_isin", "isin": "IE00B4L5Y983", "model": "cluster"}'

        Il file Excel contiene 4 sheet:
            - beta:         driver attivi (beta != 0) con il loro peso
            - breakdown:    output di intraday_adjuster.get_breakdown() per l'ISIN
            - clean_returns: corrected_return_intraday per l'ETF e i suoi driver
            - predictions:  predict_prices del modello scelto per l'ISIN
        """
        raw_id = payload.get("isin")
        if not raw_id:
            logger.error("debug_isin: 'isin' mancante nel payload")
            return

        isin = self.as_isin(raw_id)
        if isin not in self.etfs:
            logger.error(f"debug_isin: ISIN '{isin}' non trovato nell'universo ETF attivo")
            return

        if self.mid_eur is None or self.mid_eur.isna().all():
            logger.error("debug_isin: mid_eur non ancora inizializzato")
            return

        model_name = payload.get("model", "cluster")
        if model_name not in self.models:
            logger.error(f"debug_isin: modello '{model_name}' non trovato. "
                         f"Validi: {self.models.model_names}")
            return

        entry = self.models._entries[model_name]
        model = entry.model

        # ── Step 1: beta dei driver (beta != 0) ──────────────────────────────
        if isin not in model.beta.index:
            logger.error(f"debug_isin: ISIN '{isin}' non nella beta del modello '{model_name}'")
            return

        beta_row = model.beta.loc[isin]
        active_beta = beta_row[beta_row != 0].rename("beta").to_frame()
        drivers = active_beta.index.tolist()

        # ── Step 2: breakdown intraday adjuster ──────────────────────────────
        try:
            breakdown_df = self.intraday_adjuster.get_breakdown(ticker=isin)
        except Exception as e:
            logger.error(f"debug_isin: errore in get_breakdown: {e}")
            breakdown_df = pd.DataFrame()

        # ── Step 3: clean returns per ETF + driver ───────────────────────────
        returns_T = self.corrected_return_intraday.T  # index=timestamps, cols=ISINs
        available_cols = [c for c in [isin] + drivers if c in returns_T.columns]
        clean_returns_df = returns_T[available_cols]

        # ── Step 4: predict_prices del modello per l'ISIN ────────────────────
        try:
            predictions_all = model.predict_prices(self.mid_eur, entry.returns_source.T)
            if isin in predictions_all.columns:
                predictions_df = predictions_all[[isin]]
            else:
                predictions_df = pd.DataFrame()
                logger.warning(f"debug_isin: ISIN '{isin}' non nelle predizioni del modello")
        except Exception as e:
            logger.error(f"debug_isin: errore in predict_prices: {e}")
            predictions_df = pd.DataFrame()

        # ── Step 5: salva Excel ───────────────────────────────────────────────
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ".cache/debug"
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"{isin}_{model_name}_{ts}.xlsx")

        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                active_beta.to_excel(writer, sheet_name="beta", index=True)
                if not breakdown_df.empty:
                    breakdown_df.to_excel(writer, sheet_name="breakdown", index=True)
                if not clean_returns_df.empty:
                    clean_returns_df.to_excel(writer, sheet_name="clean_returns", index=True)
                if not predictions_df.empty:
                    predictions_df.to_excel(writer, sheet_name="predictions", index=True)
            logger.info(f"debug_isin: file salvato in '{file_path}'")
        except Exception as e:
            logger.error(f"debug_isin: errore nel salvataggio Excel: {e}")

    # =========================================================================
    # Market data setup
    # =========================================================================

    def on_market_data_setting(self) -> None:
        """Imposta i securities e le sottoscrizioni Bloomberg."""
        self.mid_eur = pd.Series(index=self.all_etf_plus_securities)
        self.book_eur = pd.DataFrame(columns=["BID", "ASK"])
        self.market_data.set_securities(self.all_etf_plus_securities)

        bloomberg_subscriptions = self.bloomberg_subscription_manager.get_subscription_dict()
        currency_info = self.bloomberg_subscription_manager.get_currency_informations()
        self.market_data.currency_information = currency_info

        for isin in self.etfs:
            self.global_subscription_service.subscribe_bloomberg(
                id=isin,
                subscription_string=bloomberg_subscriptions.get(isin, isin),
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

        for ccy in self.currencies:
            self.global_subscription_service.subscribe_bloomberg(
                id=ccy,
                subscription_string=f"{ccy} CURNCY",
                fields=["BID", "ASK"],
                params={"interval": 1}
            )

    def wait_for_book_initialization(self):
        """Attende l'inizializzazione del book e gestisce strumenti con dati mancanti."""
        while datetime.today().time() < time(9, 5):
            return False

        while not self.wait_for_bloomberg_initialization():
            sleep_time.sleep(1)

        return True

    def wait_for_bloomberg_initialization(self):
        while self.market_data.get_pending_subscriptions("bloomberg"):
            sleep_time.sleep(1)

        for sub in self.global_subscription_service.get_failed_subscriptions():
            isin = sub.get("id")
            if isin in self.etfs:
                ticker = self.isin_to_ticker.get(isin)
                self.global_subscription_service.subscribe_bloomberg(isin, f"{ticker} IM EQUITY", ["BID", "ASK"])

        sleep_time.sleep(5)

        for sub in self.global_subscription_service.get_failed_subscriptions():
            isin = sub.get("id")
            logger.warning(f"failed subscription: '{isin} IM EQUITY' ({self.isin_to_ticker.get(isin)})")
            self.failed_isin.append(isin)

        return True

    # =========================================================================
    # Loop ad alta frequenza
    # =========================================================================

    def update_HF(self):
        if datetime.today().time() < time(17, 30):

            with _timer("get_mid", self.logger):
                self.get_mid()

            with _timer("predict_all", self.logger):
                predictions = self.models.predict_all(self.mid_eur) or pd.DataFrame()

            if self.alpha < 1.0:
                with _timer("alpha_blend", self.logger):
                    for key in ("cluster", "intraday", "index_cluster"):
                        pred = predictions.get(key)
                        if pred is not None:
                            predictions[key] = self.alpha * pred + (1.0 - self.alpha) * self.mid_eur

            with _timer("round_prices", self.logger):
                # Recuperiamo le serie gestendo il caso in cui la chiave non esista o sia None
                idx_series = predictions.get("index_cluster")
                if idx_series is None:
                    idx_series = pd.Series(dtype=float)

                clust_series = predictions.get("cluster")
                if clust_series is None:
                    clust_series = pd.Series(dtype=float)

                intraday_series = predictions.get("intraday")
                if intraday_series is None:
                    intraday_series = pd.Series(dtype=float)

                # Calcolo dei prezzi normalizzati
                normalized_prices = {
                    # Arrotondamento con riempimento dei None/NaN a 0
                    'live_idx': round_series_to_tick(idx_series.fillna(0), self.reference_tick_size),

                    'live_clust': round_series_to_tick(clust_series.fillna(0), self.reference_tick_size),

                    'intraday': round_series_to_tick(intraday_series.fillna(0), self.reference_tick_size),

                    # Gestione del mid_eur se fosse None
                    'mid': round_series_to_tick(
                        (self.mid_eur if self.mid_eur is not None else pd.Series([0])).fillna(0),
                        self.reference_tick_size),

                    # Protezione per la divisione: se etf_prices è None o vuoto, restituisce 0 o NaN
                    'normalized_mid': (self.mid_eur / self.etf_prices.iloc[-1]) if (
                            self.etf_prices is not None and not self.etf_prices.empty) else 0,
                }

            with _timer("publish_gui", self.logger):
                self.publisher.publish_prices_to_gui(normalized_prices)

            with _timer("publish_ts", self.logger):
                self.publisher.publish_to_timeseries(normalized_prices, datetime.now(), predictions)

    def update_LF(self):
        """Aggiornamento a bassa frequenza: pubblica i ritorni intraday storici."""
        self.publisher.publish_lf_data(self.intraday_adjuster)
        mid = self.mid_eur
        etfs, fx = mid[self.etfs], mid[self.currencies]
        self.intraday_adjuster.append_update(prices=etfs, fx_prices=fx)

    # =========================================================================
    # Pricing
    # =========================================================================

    def get_mid(self) -> pd.Series:
        last_mid = self.market_data.get_mid()
        self.book_eur = self.market_data.get_data_field(["BID", "ASK"])

        if self.mid_eur is not None:
            safe_last_book = last_mid.replace(0, np.nan)
            is_outlier = (
                    last_mid.isna()
                    | (last_mid == 0)
                    | ((self.mid_eur / safe_last_book - 1).abs() > self.book_mid_threshold)
            )
            valid_entries = last_mid[~is_outlier]
            self.mid_eur.loc[[i for i in valid_entries.index if i in self.mid_eur.index]] = valid_entries
        else:
            self.mid_eur = last_mid

        with _timer("get_mid.adjuster_daily", self.logger):
            with self.adjuster.live_update(fx_prices=last_mid[self.currencies], prices=last_mid):
                self.corrected_return = self.adjuster.get_clean_returns(cumulative=True).T
                last_return = self.corrected_return.iloc[:, -1]

        with _timer("get_mid.adjuster_intraday", self.logger):
            with self.intraday_adjuster.live_update(fx_prices=last_mid[self.currencies], prices=last_mid):
                self.corrected_return_intraday = self.intraday_adjuster.get_clean_returns(cumulative=True).T
                last_return_intraday = self.corrected_return_intraday.iloc[:, -1]

        self.models.set_returns_source("cluster", self.corrected_return)
        self.models.set_returns_source("index_cluster", self.corrected_return)
        self.models.set_returns_source("intraday", self.corrected_return_intraday)

        with _timer("get_mid.publish_returns", self.logger):
            self.publisher.publish_returns(last_return, last_return_intraday)

        self.book_storage.append(last_mid)
        return last_mid

    def _reload_beta_matrices(self):
        """Ricarica e aggiorna le matrici beta per tutti i modelli."""
        self.input_params.load_inputs_db(db_path=self.db_path)

        beta_cluster = self._prepare_beta_matrix(self.input_params.beta_cluster)
        beta_cluster_index = self._prepare_beta_matrix(self.input_params.beta_cluster_index)

        if beta_cluster.empty or beta_cluster_index.empty:
            raise ValueError("One or both beta matrices are empty after filtering")

        for name, beta in [("cluster", beta_cluster), ("intraday", beta_cluster),
                           ("index_cluster", beta_cluster_index)]:
            correction = self._calculate_cluster_correction(beta, 0)
            self.models.update_beta(name, beta, correction)

        logger.info(f"  - Cluster beta shape: {beta_cluster.shape}, "
                    f"density: {(beta_cluster != 0).sum().sum() / beta_cluster.size:.2%}")
        logger.info(f"  - Index cluster beta shape: {beta_cluster_index.shape}, "
                    f"density: {(beta_cluster_index != 0).sum().sum() / beta_cluster_index.size:.2%}")

    def _prepare_beta_matrix(self, beta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra, normalizza e converte ticker->ISIN per una matrice beta.

        Args:
            beta_df: Matrice beta grezza (con ticker)

        Returns:
            Matrice beta filtrata, normalizzata e con ISIN
        """
        filtered = beta_df.loc[
            beta_df.index.intersection(self.etfs),
            beta_df.columns.intersection(self.etfs)
        ]

        if filtered.empty:
            logger.warning("Filtered beta matrix is empty")
            return pd.DataFrame()

        filtered = filtered.dropna(how='all', axis=0).dropna(how='all', axis=1)

        row_sums = filtered.sum(axis=1)

        invalid_rows = (row_sums == 0) | row_sums.isna()
        if invalid_rows.any():
            logger.warning(f"Beta matrix has {invalid_rows.sum()} rows with zero/NaN sum: "
                           f"{filtered.index[invalid_rows].tolist()}")
            filtered = filtered[~invalid_rows]
            row_sums = row_sums[~invalid_rows]

        normalized = filtered.div(row_sums, axis=0).fillna(0)

        result = (normalized
                  .rename(self.ticker_to_isin, axis=1)
                  .rename(self.ticker_to_isin, axis=0))

        logger.debug(f"Beta matrix prepared: shape {result.shape}, "
                     f"non-zero elements: {(result != 0).sum().sum()}")

        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    @property
    def instruments(self) -> list:
        return self._instruments

    @instruments.setter
    def instruments(self, value: list):
        self._instruments = [self.as_isin(_id) for _id in set(value) if self.as_isin(_id)]

    @instruments.getter
    def instruments(self) -> list[str]:
        return self._instruments

    def as_isin(self, _id: str | list[str]) -> list[str] | str:
        if isinstance(_id, str):
            return self.ticker_to_isin.get(_id, _id)
        return [self.ticker_to_isin.get(el, el) for el in _id]

    @staticmethod
    def round_series_to_tick(series, tick_dict, default_tick=0.001):
        """Arrotonda una Series ai tick specificati per ciascun strumento."""
        if series is None:
            return series
        if isinstance(tick_dict, pd.Series):
            tick_dict = tick_dict.to_dict()
        ticks = np.array([tick_dict.get(idx, default_tick) for idx in series.index]) / 2
        values = series.fillna(0).values.astype(float)
        rounded_values = np.round(np.round(values / ticks) * ticks, 10)
        return pd.Series(rounded_values, index=series.index).fillna(0)

    @staticmethod
    def filter_outliers(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
        Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)
        if logger.isEnabledFor(logging.DEBUG) and outliers.any().any():
            out_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_parquet(os.path.join(out_dir, f"{name}_raw_{ts}.parquet"))
            outliers.to_parquet(os.path.join(out_dir, f"{name}_outliers_{ts}.parquet"))
            logger.debug(f"Outliers in {name}: {outliers.sum().sum()} values, saved to {out_dir}")
        df[outliers] = np.nan
        return df

    @staticmethod
    def _calculate_cluster_correction(cluster_betas: pd.DataFrame,
                                      threshold: float = 0.5) -> pd.Series:
        """
        Calcola il fattore di correzione per ciascun sottocluster.

        Returns:
            pd.Series: fattori di correzione per ISIN.
        """
        if cluster_betas.empty:
            return pd.Series(dtype=float)

        cluster_betas = cluster_betas.sort_index(axis=1).sort_index(axis=0)

        for etf in cluster_betas.index:
            if etf in cluster_betas.columns:
                cluster_betas.loc[etf, etf] = 0

        non_zero_counts = (cluster_betas != 0).sum(axis=1)

        cluster_threshold = pd.Series(index=cluster_betas.index, dtype=float)
        cluster_threshold[non_zero_counts > 0] = threshold / non_zero_counts[non_zero_counts > 0]
        cluster_threshold[non_zero_counts == 0] = 0

        cluster_sizes = cluster_betas.gt(cluster_threshold, axis=0).sum(axis=1) + 1

        return cluster_sizes.where(cluster_sizes == 1, (cluster_sizes - 1) / cluster_sizes)


@contextmanager
def _timer(label: str, log: logging.Logger = logger):
    t0 = sleep_time.perf_counter()
    yield
    elapsed_ms = (sleep_time.perf_counter() - t0) * 1000
    log.debug(f"[TIMER] {label}: {elapsed_ms:.1f} ms")
