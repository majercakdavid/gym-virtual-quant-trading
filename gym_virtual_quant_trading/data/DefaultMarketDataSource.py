import csv
import os

from collections import OrderedDict
from datetime import datetime
from typing import OrderedDict, Set

import gym_virtual_quant_trading.data as data

class DefaultMarketData(data.BaseMarketData):
    pass

class DefaultMarketSymbolData(data.BaseMarketSymbolData):
    pass

class DefaultMarketDataSource(data.BaseMarketDataSource):
    def __init__(self):
        super(DefaultMarketDataSource, self).__init__()
        
        self._data          = {}                        # type: OrderedDict[datetime, DefaultMarketData]
        self._symbols       = set()                     # type: Set[str]
        self._data_iterator = iter(self._data.items())  # type: dict_items
        
        default_csv_data_path = os.path.join(os.path.dirname(__file__), 'all_stocks_5yr.csv')
        with open(default_csv_data_path, newline='') as csv_contents:
            self._load_data(csv.DictReader(csv_contents))

    def __next__(self):
        """Gets the next data in time from file

        Returns:
            DefaultMarketData: instruments prices in next time step
        """
        return next(self._data_iterator)[1]

    def get_data_at(self, time):
        """Gets market data at given time

        Returns:
            DefaultMarketData: market data at given time
        """
        return self._data[time]

    def symbols(self):
        return self._symbols

    def _load_data(self, csv_dict_reader):
        
        data = {}

        # row example:
        #   'date':'2013-02-08', 'open':'15.07', 'high':'15.12', 'low':'14.63', 'close':'14.75', 'volume':'8407500', 'Name':'AAL'
        for row in csv_dict_reader:
            # Collect all symbols in the given file
            self._symbols.add(row['Name'])

            if row['date'] not in data:
                data[row['date']] = DefaultMarketData(time = row['date'], entries = {})

            if row['Name'] not in data[row['date']].entries:
                data[row['date']].entries[row['Name']] = DefaultMarketSymbolData(
                    symbol  = row['Name'],
                    open    = .0 if not row['open'] else float(row['open']),
                    high    = .0 if not row['high'] else float(row['high']),
                    low     = .0 if not row['low'] else float(row['low']),
                    close   = .0 if not row['close'] else float(row['close']),
                    volume  = int(row['volume']),
                )

        self._data = OrderedDict(sorted(data.items()))
        self._data_iterator = iter(self._data.items())