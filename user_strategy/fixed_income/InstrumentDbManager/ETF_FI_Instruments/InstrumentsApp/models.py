from typing import List

from django.db import models
from django.core.exceptions import ValidationError
CURRENCY = {
    'PLN', 'CHF', 'AUD', 'NOK', 'GBP', 'CZK', 'DKK', 'INR', 'HKD', 'EUR', 'SEK',
    'UAH', 'RON', 'ISK', 'RUB', 'HRK', 'CAD', 'USD', 'JPY', 'HUF', 'ILS', 'JMD',
    'ARS', 'BRL', 'CLP', 'CNH', 'COP', 'DOP', 'EGP', 'IDR', 'FJD', 'KES', 'KRW',
    'KZT', 'MXN', 'MYR', 'NGN', 'NPR', 'PKR', 'PHP', 'QAR', 'SAR', 'SGD', 'THB',
    'TWD', 'VND', 'ZAR', 'NZD', 'TRY', 'CNY', 'AED', 'KWD', 'UYU','PEN','RSD'}


class MyModel(models.Model):
    objects = models.Manager()

    class Meta:
        abstract = True

    @classmethod
    def validate_fields(cls, kwargs):
        """Ensure provided values match predefined finite sets."""
        required_fields = getattr(cls, "REQUIRED_FIELDS", {})  # Defined in child models
        missing_fields = [fld for fld in required_fields if fld not in kwargs]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")

    @classmethod
    def validate_keys(cls, kwargs):
        """Ensure all unique_together fields or primary key fields are present."""
        unique_together = getattr(cls._meta, "unique_together", None)
        if unique_together is None:
           primary_keys = [field.name for field in cls._meta.fields if field.primary_key]
           required_keys = set(primary_keys)
        else:
            required_keys = unique_together[0]

        missing_keys = [key for key in required_keys if key not in kwargs]
        if missing_keys:
            raise ValidationError(f"Missing required keys: {', '.join(missing_keys)}")

    @classmethod
    def validate_choices(cls, kwargs):
        """Ensure provided values match predefined finite sets."""
        field_choices = getattr(cls, "FIELD_CHOICES", {})  # Defined in child models
        for field, choices in field_choices.items():
            if field in kwargs and kwargs[field] not in choices:
                raise ValidationError(f"Invalid value for {field}: {kwargs[field]}. Allowed: {choices}")



class Instruments(MyModel):
    REQUIRED_FIELDS = ('isin', 'exchange', 'trading_currency', 'fund_currency', 'ticker')
    VALID_EXCHANGES = {"Xetra", "ETF Plus", "LSE"}  # Allowed exchanges
    FIELD_CHOICES = {"exchange": VALID_EXCHANGES, "trading_currency": CURRENCY, "fund_currency": CURRENCY}  # Mapping of fields to allowed values

    isin = models.TextField(db_column='ISIN')
    exchange = models.TextField(db_column='Exchange')
    trading_currency = models.TextField(db_column='Trading Currency')
    fund_currency = models.TextField(db_column='Fund Currency', blank=True, null=True)
    ticker = models.TextField(db_column='Ticker', blank=True, null=True)

    def __str__(self):
        return (f"ISIN: {self.isin}, Exchange: {self.exchange}, Trading Currency: {self.trading_currency}, "
                f"Fund Currency: {self.fund_currency}, Ticker: {self.ticker}")

    class Meta:
        db_table = 'Instruments'
        unique_together = (('isin', 'exchange', 'trading_currency'),)


class Brothers(MyModel):
    REQUIRED_FIELDS = ('isin',)
    isin = models.TextField(db_column='ISIN', primary_key=True)
    ticker = models.TextField(db_column='Ticker', blank=True, null=True)
    cluster = models.IntegerField(db_column='Cluster', blank=True, null=True)
    weight = models.FloatField(db_column='Weight', blank=True, null=True)

    class Meta:
        db_table = 'Brothers'

    def __str__(self):
        """Return a string representing the cluster and its members."""
        # Get all members of the same cluster
        cluster_members = Brothers.objects.filter(cluster=self.cluster)
        # Create a list of strings for each member's ISIN and weight
        member_strs = [f"Ticker: {member.ticker}, Weight: {member.weight}" for member in cluster_members]
        return f"Cluster {self.cluster}, members: " + " | ".join(member_strs)

    def save(self, *args, **kwargs):
        """Override save method to handle weight distribution when a new member is added to a cluster."""
        clusters = sorted(list(Brothers.objects.values_list('cluster', flat=True).distinct()))
        isins = Brothers.objects.values_list('isin', flat=True)
        if self.isin in isins:
            raise ValidationError(
                f"Isin {self.isin} already added, cluster is {Brothers.objects.filter(isin=self.isin)[0]}. "
                f"To change cluster delete and add again")
        cluster_list: List[int] = []
        if not self.cluster or self.cluster not in clusters:
            if clusters:
                self.cluster = max(clusters) + 1
            else:
                self.cluster = 1
        else:
            for cluster in clusters:
                brothers_in_cluster = Brothers.objects.filter(cluster=cluster)
                brothers_number = brothers_in_cluster.count()
                if self.cluster == cluster:
                    brothers_number += 1

                weight = 0
                if brothers_number >= 2:
                    weight = 1 / (brothers_number - 1)

                cluster_to_insert = 1
                if cluster_list:
                    cluster_to_insert = max(cluster_list) + 1

                brothers_in_cluster.update(weight=weight, cluster=cluster_to_insert)
                cluster_list.append(cluster_to_insert)
                if self.cluster == cluster:
                    self._cluster = cluster_to_insert
                    self._weight = weight

            self.cluster = self._cluster
            self.weight = self._weight

        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        clusters = sorted(list(Brothers.objects.values_list('cluster', flat=True).distinct()))
        isins = Brothers.objects.values_list('isin', flat=True)
        if self.isin not in isins:
            raise ValidationError(f"Isin {self.isin} not present in table")
        cluster_deleted = False
        self.cluster = Brothers.objects.get(isin=self.isin).cluster
        for cluster in clusters:
            cluster_to_insert = cluster
            if cluster_deleted:
                cluster_to_insert -= 1
                brothers_in_cluster = Brothers.objects.filter(cluster=cluster)
                brothers_in_cluster.update(cluster=cluster_to_insert)
            if self.cluster == cluster:
                brothers_in_cluster = Brothers.objects.filter(cluster=cluster)
                brothers_number = brothers_in_cluster.count()
                if brothers_number <= 2:
                    weight = 0
                else:
                    weight = 1 / (brothers_number - 2)
                brothers_in_cluster.update(weight=weight)

                if brothers_number == 1:
                    cluster_deleted = True


        super().delete(*args, **kwargs)


class Clusters(MyModel):
    isin = models.TextField(db_column='ISIN', primary_key=True)
    ticker = models.TextField(db_column='Ticker')
    cluster = models.TextField(db_column='Cluster')
    subcluster = models.IntegerField(db_column='Subcluster')

    class Meta:
        db_table = 'Clusters'

    def __str__(self):
        return f"ISIN: {self.isin}, Ticker: {self.ticker}, Cluster: {self.cluster}, Subcluster: {self.subcluster}"


class Ter(MyModel):
    isin = models.TextField(db_column='ISIN', primary_key=True)
    hard_coding = models.FloatField(db_column='Hard Coding', null=True, blank=True)

    class Meta:
        db_table = 'Ter'

    def __str__(self):
        return f"ISIN: {self.isin}, Hard Coding: {self.hard_coding}"


class Drivers(MyModel):
    name = models.TextField(db_column='Name', primary_key=True)
    subscription_ts = models.TextField(db_column='Subscription Ts', null=True, blank=True)
    market_ts = models.TextField(db_column='Market Ts', null=True, blank=True)
    subscription_bbg =models.TextField(db_column='Subscription BBG')

    class Meta:
        db_table = 'Drivers'

    def __str__(self):
        return (f"name: {self.name}, Subscription ts: {self.subscription_ts}, Market ts: {self.market_ts}, "
                f"subscription BBG: {self.subscription_bbg}")


class YtmMapping(MyModel):
    isin = models.TextField(db_column='ISIN', primary_key=True)
    name = models.TextField(db_column='Name', default=None)
    subscription = models.TextField(db_column='Subscription', default=None)
    mapping_isin = models.TextField(db_column='Mapping ISIN', null=True, blank=True)
    mapping_name = models.TextField(db_column='Mapping Name', null=True, blank=True)
    mapping_subscription = models.TextField(db_column='Mapping Subscription', null=True, blank=True)
    hard_coding = models.FloatField(db_column='Hard Coding', null=True, blank=True)
    repo_rate =models.TextField(db_column='Repo Rate', null=True, blank=True)
    instrument_type =models.TextField(db_column='Instrument Type', null=True, blank=True)

    class Meta:
        db_table = 'YtmMapping'

    def __str__(self):
        return (f"ISIN: {self.isin}, Name: {self.name}, Subscription: {self.subscription}, "
                f"Mapping ISIN: {self.mapping_isin}, Mapping Name: {self.mapping_name},"
                f" Mapping Subscription: {self.mapping_subscription}, Hard Coding: {self.hard_coding}, "
                f"Repo Rate: {self.repo_rate}, Instrument type {self.instrument_type}")


class CurrencyFactorMultiplier(MyModel):
    currency_exchange = models.TextField(db_column='Currency Exchange', primary_key=True)
    multiplier = models.FloatField(db_column='Multiplier', null=True, blank=True)

    class Meta:
        db_table = 'CurrencyFactorMultiplier'

    def __str__(self):
        return f"Currency Exchange: {self.currency_exchange}, Multiplier: {self.multiplier}"


class HedgeRatioDriver(MyModel):
    target_isin = models.TextField(db_column='Target_ISIN')
    target_ticker = models.TextField(db_column='Target_Ticker')
    driver = models.TextField(db_column='Driver')
    beta = models.FloatField(db_column='Beta')

    class Meta:
        db_table = 'HedgeRatioDriver'
        unique_together = (('target_isin', 'driver'),)

    def __str__(self):
        return f"Target ISIN: {self.target_isin}, Target Ticker: {self.target_ticker}, Driver: {self.driver}, Beta: {self.beta} "


class HedgeRatioCluster(MyModel):
    target_isin = models.TextField(db_column='Target ISIN')
    target_ticker = models.TextField(db_column='Target Ticker')
    driver_isin = models.TextField(db_column='Driver ISIN')
    driver_ticker = models.TextField(db_column='Driver Ticker')
    beta = models.FloatField(db_column='Beta')

    class Meta:
        db_table = 'HedgeRatioCluster'
        unique_together = (('target_isin', 'driver_isin'),)

    def __str__(self):
        return (f"Target ISIN: {self.target_isin}, Target Ticker: {self.target_ticker}, Driver Isin: {self.driver_isin},"
                f" Driver Ticker: {self.driver_ticker}, Beta: {self.beta} ")

class CurrencyExposureManual(MyModel):
    isin = models.TextField(db_column='ISIN')
    ticker = models.TextField(db_column='Ticker', blank=True, null=True)
    currency = models.TextField(db_column='Currency')
    weight = models.FloatField(db_column='Weight', blank=True, null=True)
    hedged = models.TextField(db_column='Hedged', blank=True, null=True)

    class Meta:
        db_table = 'CurrencyExposureManual'
        unique_together = (('isin', 'currency'),)

    def __str__(self):
        return f"ISIN: {self.isin}, Ticker: {self.ticker}, Currency: {self.currency}, Weight: {self.weight}, Hedged: {self.hedged}"

