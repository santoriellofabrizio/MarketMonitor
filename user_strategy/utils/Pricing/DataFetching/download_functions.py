__all__ = ["download_daily_prices_fx",
           "memoryFixedIncome",
           "download_daily_prices",
           "get_price_for_day_time",
           "download_repo",
           "download_yas",
           "download_ter",
           "download_dividends_currency",
           "download_dividends",
           "download_fx_forward_composition",
           "process_downloaded_prices"]

import logging
from datetime import timedelta, time, date
from typing import Optional, List, Dict

import pandas as pd

from pandas import DatetimeIndex, DataFrame, Series
from tqdm import tqdm
from xbbg import blp
from xbbg.blp import bdh, bdib, bdp

from sfm_dbconnections.DbConnectionParameters import DbConnectionParameters, TimescaleConnectionParameters
from sfm_timescaledb_queries.QueryTSMarkets import QueryTSMarkets
from sfm_timescaledb_queries.callable_functions import download_daily_fairvalues, download_daily_fairvalues_currency
from user_strategy.FixedIncomeETF import memoryFixedIncome
from user_strategy.utils import memoryPriceProvider

logger = logging.getLogger()


def _check_cache_validity(result):
    return True


from datetime import datetime

def _check_cache_validity_only_today(cache_metadata):
    """
    Callback per validare la cache.
    La cache è valida solo se la data corrente coincide con quella di creazione.
    """
    # Ottieni la data di oggi
    today = datetime.now().date()

    # Ottieni la data di creazione dalla metadata della cache
    # Supponendo che 'cache_metadata' contenga un campo 'time' come Unix timestamp
    cache_creation_timestamp = cache_metadata.get("time", None)

    if cache_creation_timestamp:
        # Converti il timestamp in un oggetto datetime
        cache_creation_date = datetime.fromtimestamp(cache_creation_timestamp).date()

        # Confronta solo la data (non l'ora)
        return cache_creation_date == today

    # Se non c'è una data di creazione, invalida la cache
    return False


def process_downloaded_prices(prices: DataFrame,
                              array_date: list,
                              col_name,
                              instruments: list | None = None) -> DataFrame:
    """
    Pivot and check for missing data in the downloaded prices.
    """
    prices: pd.DataFrame = prices.pivot_table(index="Data", columns=col_name, values="mid")
    # noinspection PyTypeChecker,PyTypeHints
    prices.index: DatetimeIndex = pd.to_datetime(prices.index).date
    # Check for missing data
    missing_data = prices.index.symmetric_difference(array_date)
    if len(missing_data) > 0:
        logger.warning(f"WARNING: {missing_data} not found on Timescale for Prices")
        for date in missing_data:
            prices.loc[date] = None  # Fill missing dates with None
    if instruments:
        missing_prices = prices.columns.symmetric_difference(instruments)
        if len(missing_data) > 0:
            logger.warning(f"WARNING: {missing_data} not found on Timescale for Prices")
            for instr in missing_prices:
                prices[instr] = None
    return prices.sort_index(ascending=False)


def download_daily_prices(array_date, isins, price_snipping_time, market="EURONEXT", desc=f"Downloading prices"):
    # Utilizza tqdm per creare una barra di avanzamento per il ciclo
    return pd.concat([
        download_single_data_prices(data, isins, price_snipping_time, market)
        for data in tqdm(array_date, desc=desc, unit="date")
    ])


def download_daily_prices_fx(array_date, currencies_EUR_ccy, price_snipping_time):
    # Utilizza tqdm per creare una barra di avanzamento per il ciclo
    return pd.concat([
        download_single_data_fx(data, currencies_EUR_ccy, price_snipping_time)
        for data in tqdm(array_date, desc="Downloading daily FX prices Oracle", unit="date")
    ])


@memoryPriceProvider.cache
def get_price_for_day_time(ticker, day: date, snipping_time: time, download_eod: bool) -> Optional[float]:
    """
        Get the price for a given ticker at a specific date and time.

        Args:
            ticker: Ticker symbol to fetch the price for.
            day: Date to fetch the price on.
            snipping_time: Time to fetch the price at.
            download_eod: Download directly the instrument at end of day without requesting intraday data

        Returns:
            float: The fetched price.
        """

    # Fetch bid and ask prices
    def get_closest_price(ticker, field, day_time, config) -> Optional[float]:

        data = blp.bdib(ticker, dt=day, session="marketsnip", interval=30, typ=field, config=config)
        if data.empty: data = blp.bdib(ticker, dt=day, session="allday", interval=30, typ=field, config=config)
        if data.empty:
            logger.warning(f"Could not find an intraday price for {ticker} in {day} ({field})")
            return None
        try:
            result = (data.
                      tz_convert("Europe/Rome").
                      loc[:day_time, ticker]["open"].
                      asof(day_time))
        except IndexError:
            result = data.tz_convert("Europe/Rome").loc[:, ticker]["open"].iloc[-1]
            logger.warning(f"Cannot find price snip for {ticker}. Using last price available")

        logging.info(f"{ticker} snipped at {data.tz_convert('Europe/Rome').index.asof(day_time)},"
                        f" price:{result} {field}")
        return result

    logger = logging.getLogger()
    day_time = datetime.combine(day, snipping_time)

    day_time_tz_aware, time_zone, start_time, end_time = get_time_zone(ticker, day_time)
    config = pd.DataFrame([{'tz': time_zone,
                            'marketsnip': [start_time, end_time],
                            'allday': ["10:00", "17:15"]}], index=[ticker])

    # Create configuration for price snipping
    # Aggiungere la logica per BID e ASK
    if not download_eod:
        bid = get_closest_price(ticker, "BID", day_time_tz_aware, config)
        ask = get_closest_price(ticker, "ASK", day_time_tz_aware, config)
        # Check if prices were found
        if ask is not None and bid is not None:
            fairprice = (bid + ask) / 2
            if fairprice > 0:
                return fairprice  # Return valid fair price

    # If no price found in the snipping window, fall back to historical data
    try:
        bdh_snip = bdh(ticker, flds="PX_LAST",
                       start_date=day.strftime("%Y%m%d"),
                       end_date=(day + timedelta(1)).strftime("%Y%m%d")).iloc[0, 0]
    except Exception as e:
        logger.warning(f"Cannot find price {ticker} (dismissed ?), return N/A: {e}")
        bdh_snip = None
    if bdh_snip is not None:
        if not download_eod:
            logger.warning(f"Cannot find price for specific snipping time for {ticker}, using bdh: {bdh_snip}")
        return bdh_snip  # Return fallback price
    else:
        logger.error(f"Cannot find price for {ticker}. NA value")
        return None


@memoryPriceProvider.cache(cache_validation_callback=_check_cache_validity_only_today)
def download_single_data_fx(data, currencies, snipping_time):
    params = DbConnectionParameters
    query_: QueryTSMarkets = QueryTSMarkets(
        params.get_timescale_parameter(TimescaleConnectionParameters.HOST),
        params.get_timescale_parameter(TimescaleConnectionParameters.PORT),
        params.get_timescale_parameter(TimescaleConnectionParameters.DB_NAME),
        params.get_timescale_parameter(TimescaleConnectionParameters.USERNAME),
        params.get_timescale_parameter(TimescaleConnectionParameters.PASSWORD)
    )
    return download_daily_fairvalues_currency(
        array_date=[data],
        array_currency=currencies,
        fairvalue_time=snipping_time,
        query_ts=query_)


@memoryPriceProvider.cache(cache_validation_callback=_check_cache_validity_only_today)
def download_single_data_prices(data, isins, snipping_time, market="EURONEXT"):
    params = DbConnectionParameters
    query_: QueryTSMarkets = QueryTSMarkets(
        params.get_timescale_parameter(TimescaleConnectionParameters.HOST),
        params.get_timescale_parameter(TimescaleConnectionParameters.PORT),
        params.get_timescale_parameter(TimescaleConnectionParameters.DB_NAME),
        params.get_timescale_parameter(TimescaleConnectionParameters.USERNAME),
        params.get_timescale_parameter(TimescaleConnectionParameters.PASSWORD)
    )
    return download_daily_fairvalues(
        array_date=[data],
        market_isin_dictionary={market: isins},
        fairvalue_time=snipping_time,
        currency='EUR',
        query_ts=query_)


def get_time_zone(instrument: str, day_time, start_time_bar: int = 15, end_time_bar: int = 18):
    import pytz
    utc_time = pytz.timezone("Europe/Rome").localize(day_time)
    start_time, end_time = utc_time.replace(hour=start_time_bar), utc_time.replace(hour=end_time_bar)
    if instrument[:2] in ["FV", "TY", "US", "WN", "TU", "JB", "ES"]:
        timezone_str = "America/New_York"
    else:
        timezone_str = 'Europe/Rome'
    timezone = pytz.timezone(timezone_str)
    return (utc_time.astimezone(timezone),
            timezone_str,
            start_time.astimezone(timezone).strftime("%H:%M"),
            end_time.astimezone(timezone).strftime("%H:%M"))


@memoryPriceProvider.cache
def download_dividends(isins: List[str], **kwargs) -> pd.DataFrame:
    if not isins: return pd.DataFrame()
    market_of_instruments = kwargs.get("market_of_instruments", None) or {}
    bbg_subscription = [f"{isin} {market_of_instruments.get(isin, 'IM')} EQUITY"
                        for isin in isins]
    print("Downloading dividends...")
    if kwargs.get("last_or_all", "last").lower() == 'last':
        div = _download_last_dividends(bbg_subscription)
    elif kwargs.get("last_or_all").lower() == 'all':
        div = _download_all_dividends(bbg_subscription, **kwargs)
    else:
        print(f"dividend mode can only be 'last' or 'all', not {kwargs.get('last_or_all')}")
        return pd.DataFrame()
    print("..done..")
    return div


def _download_all_dividends(subscription, **kwargs):
    try:
        dividends = blp.bds(subscription, "DVD_HIST_ALL",
                            DVD_START_DT=(datetime.today()
                                          - timedelta(days=kwargs.get("n_days", 60))).strftime("%YYYY%mm%dd"),
                            timeout=1)
        if dividends.empty:
            print("No dividend found")
            return pd.DataFrame()
    except:
        print("Error while downloading last dividend")
        return pd.DataFrame()
    return dividends


def _download_last_dividends(subscription):
    try:
        dividends = bdp(subscription, flds=["DVD_SH_LAST", "DVD_EX_DT"])
        dividends.columns = ["dividend_amount", "ex_date"]
        if dividends.empty:
            print("No dividend found")
            return pd.DataFrame()
    except:
        print("Error while downloading last dividend")
        return pd.DataFrame()
    return dividends


@memoryPriceProvider.cache
def download_dividends_currency(isins, market_of_instruments: Optional[Dict[str, str]] = None):
    if not isins: return pd.DataFrame()
    if market_of_instruments is None: market_of_instruments = {}
    print("Downloading dividends currency...")
    bbg_subscription = [f"{isin} {market_of_instruments.get(isin, 'IM')} EQUITY"
                        for isin in isins]
    from xbbg import blp
    while (tries := 0) <= 3:
        try:
            dividend_ccy = blp.bdp(bbg_subscription, "dvd_crncy")
            dividend_ccy = dividend_ccy["dvd_crncy"]
            print("Done...")
            return dividend_ccy
        except Exception as e:
            logger.error(f"error downloading dividends currency: {e}")
            raise Exception
        finally:
            tries += 1


@memoryPriceProvider.cache
def download_ter(isins: list[str], market_of_instruments: Optional[Dict] = None) -> Series:
    from xbbg import blp
    if market_of_instruments is None: market_of_instruments = {}
    bbg_subscription = [f"{isin} {market_of_instruments.get(isin, 'IM')} EQUITY" for isin in isins]
    TER = blp.bdp(bbg_subscription, "fund_expense_ratio")["fund_expense_ratio"] / 100
    TER.index = [idx.split(" ")[0] for idx in TER.index]
    return TER


@memoryPriceProvider.cache(cache_validation_callback=_check_cache_validity_only_today)
def download_yas(isins, yas_mapping):
    from xbbg.blp import bdp
    yas = bdp(yas_mapping.unique(), "yas_bond_yld")["yas_bond_yld"] / 100
    try:
        yas_df = yas.loc[yas_mapping[isins]]
        yas_df.index = isins
        return yas_df
    except KeyError as e:
        logger.critical(f"No yas found on BBG:  {e}. \n Hard Code missing values in Input file")
        raise KeyboardInterrupt


def download_repo(isins: List[str], repo_mapping: pd.Series):
    from xbbg.blp import bdp
    logger.info(f"Downloading REPO's for {', '.join(isins)}")
    try:
        repos_df = (bdp(repo_mapping.unique(), "px_last")["px_last"] / 100).loc[repo_mapping[isins]]
        repos_df.index = isins
        logger.info(f"Downloaded REPO's")
        return repos_df
    except KeyError as e:
        logger.critical(f"No REPO found on BBG: {e}. \n Hard code missing values in Input file.")
        raise KeyboardInterrupt


@memoryPriceProvider.cache
def download_fx_forward_ticker_date(ticker, date):
    """
    this function downloads the fx forward price at 16:35.
    :param ticker:
    :param date:
    :return:
    """
    result = bdib(ticker, date, ref="EquityLondon", session="post", interval=1)
    try:
        return result[ticker, "open"].iloc[-1]
    except Exception as e:
        logger.warning(f"{ticker} - {date} - not found, price is assumed to be 0")
        return 0


def download_fx_forward_composition(fx_forward: List[str], date_range: List[datetime.date]):
    prices_fx_frwd = pd.DataFrame(0.0, index=date_range, columns=fx_forward)
    for fx_frwd in fx_forward:
        prices_fx_frwd.loc[:, fx_frwd] = [download_fx_forward_ticker_date(fx_frwd, date.strftime("%Y%m%d"))
                                          for date in date_range]

    return prices_fx_frwd


