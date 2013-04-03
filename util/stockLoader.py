
from cStringIO import StringIO
import csv
import datetime
import urllib2

class StockLoader(object):
    @classmethod
    def loadStock(cls, ticker, toDate = 'now', fromDate = '20090101'):
        """Yields a series of dicts with info on the given stock."""
        # Months are ranged 0-11
        if toDate == 'now':
            toDate = datetime.date.today().isoformat().replace('-', '')

        url = 'http://ichart.finance.yahoo.com/table.csv?s={symbol}&d={toMonth}&e={toDay}&f={toYear}&g=d&a={fromMonth}&b={fromDay}&c={fromYear}&ignore=.csv'
        url = url.format(
                symbol = ticker,
                toYear = toDate[0:4],
                toMonth = int(toDate[4:6]) - 1,
                toDay = toDate[6:8],
                fromYear = fromDate[0:4],
                fromMonth = int(fromDate[4:6]) - 1,
                fromDay = fromDate[6:8],
        )
        data = urllib2.urlopen(url).read()
        dataIo = StringIO(data)
        r = csv.reader(dataIo)
        headers = r.next()
        while True:
            try:
                row = r.next()
            except StopIteration:
                break
            # First entry is date
            row[0] = datetime.datetime.strptime(row[0], "%Y-%m-%d")
            yield dict(zip(headers, row))
