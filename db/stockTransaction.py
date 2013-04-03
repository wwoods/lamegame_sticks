
from datetime import datetime
from mongokit import Document, OR

class StockTransaction(Document):
    __collection__ = 'stock_transaction'

    structure = {
            'ticker': unicode,
            'quantity': int,
            'price': OR(int, float),
            'transactionPrice': OR(int, float),
            'date': datetime
    }

    default_values = {
            'transactionPrice': 3.95,
    }

    validators = {
            'ticker': lambda x: x == x.upper()
    }
