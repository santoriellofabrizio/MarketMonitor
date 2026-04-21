from abc import ABC
from typing import List


class InputParams(ABC):

    def set_forecast_aggregation_func(self, kwargs: dict, keys: List[str]) -> None:
        """Instantiate forecast aggregators for each key and store them as attributes.

        Args:
            kwargs: Pricing config dict keyed by aggregator name.
            keys: List of aggregator names to initialise (e.g. ["cluster", "driver"]).
        """
        from user_strategy.utils.pricing_models.AggregationFunctions import forecast_aggregation
        import logging
        logger = logging.getLogger(__name__)

        for key in keys:
            try:
                params = kwargs[key]
                setattr(
                    self,
                    f"forecast_aggregator_{key}",
                    forecast_aggregation[params["forecast_aggregation"]](
                        **params[params["forecast_aggregation"]]
                    ),
                )
            except KeyError:
                logger.critical(
                    f"forecast aggregator for '{key}' not implemented. "
                    f"available: {list(forecast_aggregation.keys())}"
                )
                raise KeyboardInterrupt
