from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict, Set, NamedTuple
from datetime import datetime

class BaseMarketSymbolData(NamedTuple):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class BaseMarketData(NamedTuple):
    time: datetime
    entries: Dict[str, BaseMarketSymbolData] # contains BaseMarketSymbolData(value) per symbol(key)

class BaseMarketDataSource(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        """Initializes the data source"""
        pass

    @abstractmethod
    def __next__(self):
        """Return prices of financial instruments during next timestamp

        Returns:
            BaseMarketData: prices of financial instruments
        """
        pass

    @abstractmethod
    def get_data_at(self, time):
        """Return prices of financial instruments at a given timestamp if possible

        Returns:
            BaseMarketData: prices of financial instruments
        """
        pass

    @abstractproperty
    def symbols(self):
        """Return set of symbols available from source

        Returns:
            Set[str]: set of instruments symbols
        """
        pass