from user_strategy.utils.pricing_models.DataFetching.DataPreprocessor import DataPreprocessor


class DataPreprocessorEquity(DataPreprocessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
