class InputParamsNAV:

    def __init__(self, callback_attributes: dict | None = None) -> None:
        self._callback_attributes: dict[str, callable] = callback_attributes or {}
        self._isin_to_check: list = []

    @property
    def isin_to_check(self) -> list:
        return self._isin_to_check

    @isin_to_check.setter
    def isin_to_check(self, isin_to_check: list) -> None:
        if "isin_to_check" in self._callback_attributes:
            self._callback_attributes["isin_to_check"](isin_to_check)
        self._isin_to_check = isin_to_check

    @isin_to_check.getter
    def isin_to_check(self) -> list:
        return self._isin_to_check
