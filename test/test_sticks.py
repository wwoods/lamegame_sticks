
import datetime
import mock
from mongokit import ValidationError
from unittest import TestCase

import config
config.mongo_database = 'test_' + config.mongo_database
from sticks import db, add_stock, load_stock

class FakedArgs(object):
    pass



def FakeArgs(**kwargs):
    r = mock.Mock()
    o = FakedArgs()
    baseKwargs = dict(ticker = u'AAA', quantity = 1, price = 2,
            date = datetime.datetime.utcnow())
    o.__dict__.update(baseKwargs)
    o.__dict__.update(kwargs)
    r.parse_args.return_value = o
    return r


class TestSticks(TestCase):
    @classmethod
    def setUpClass(cls):
        # Clear all transactions prior to this class
        db.StockDay.collection.remove()
        db.StockTransaction.collection.remove()


    def test_addStock_basic(self):
        before = list(db.StockTransaction.find())
        add_stock(FakeArgs())
        after = list(db.StockTransaction.find())
        self.assertEqual(1, len(after) - len(before))


    def test_addStock_lowercaseTicker(self):
        with self.assertRaises(ValidationError):
            add_stock(FakeArgs(ticker = u'qqq'))


    def test_loadStock_works(self):
        date1 = datetime.datetime(year = 2013, month = 4, day = 3)
        date2 = datetime.datetime(year = 2013, month = 4, day = 2)
        fakeRows = [
                # NOTE - data should be all strings, since that's how it comes
                # back except for the date.
                { 'Date': date1, 'Open': '1', 'High': '2', 'Low': '3',
                    'Close': '4', 'Volume': '5', 'Adj Close': '6' },
                { 'Date': date2, 'Open': '7', 'High': '8', 'Low': '3',
                    'Close': '4', 'Volume': '5', 'Adj Close': '6' },
        ]
        with mock.patch('util.stockLoader.StockLoader.loadStock',
                mock.Mock(return_value = fakeRows)):
            load_stock(FakeArgs(ticker = u'QQQ'))
            # We should have 2 stock days with ticker QQQ
            days = list(db.StockDay.find())
            self.assertEqual(2, len(days))
            self.assertEqual('QQQ', days[0]['ticker'])

            # Loading again should overwrite those days due to _id hacking,
            # not add 2 more
            load_stock(FakeArgs(ticker = u'QQQ'))
            newDays = list(db.StockDay.find())
            self.assertEqual(2, len(newDays))
