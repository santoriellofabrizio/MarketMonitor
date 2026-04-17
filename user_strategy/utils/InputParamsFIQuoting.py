import logging
from user_strategy.utils.pricing_models.AggregationFunctions import forecast_aggregation
from user_strategy.utils.InputParamsFI import InputParamsFI

logger = logging.getLogger()


class InputParamsFIQuoting(InputParamsFI):
    """
    InputParamsFI variant for live quoting.

    Differences from the parent:
    - Default price_snipping_time_string: "17:00:00"  (parent default: None)
    - Default number_of_days: 10                       (parent default: None)
    - set_forecast_aggregation_func also initializes the "brother" aggregator
    """

    def __init__(self, params, **kwargs):
        quoting_defaults = {
            "price_snipping_time_string": "17:00:00",
            "number_of_days": 10,
        }
        # YAML config in `params` takes precedence over quoting defaults
        super().__init__({**quoting_defaults, **params}, **kwargs)

    def set_forecast_aggregation_func(self, kwargs: dict) -> None:
        """Extends parent to also initialize the 'brother' forecast aggregator."""
        super().set_forecast_aggregation_func(kwargs)
        try:
            params = kwargs["brother"]
            self.forecast_aggregator_brother = forecast_aggregation[params["forecast_aggregation"]](
                **params[params["forecast_aggregation"]]
            )
        except KeyError:
            self.logger.critical(
                f"forecast aggregator for brother not implemented. available: {forecast_aggregation}"
            )
            raise KeyboardInterrupt
