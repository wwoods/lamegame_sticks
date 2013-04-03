from datetime import datetime
from mongokit import Document, OR

class StockDay(Document):
    __collection__ = 'stock_day'

    structure = {
            'ticker': unicode,
            'date': datetime,
            'open': OR(int, float),
            'high': OR(int, float),
            'low': OR(int, float),
            'close': OR(int, float),
            'volume': OR(int, float),
            'adjClose': OR(int, float),
    }

    validators = {
            'ticker': lambda x: x == x.upper()
    }

    def validate(self, *args, **kwargs):
        # Assign unique ID per day so we don't get duplicates
        self['_id'] = self['ticker'] + '_' + self['date'].strftime("%Y%m%d")
        super(StockDay, self).validate(*args, **kwargs)
