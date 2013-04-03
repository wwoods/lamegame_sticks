#! /usr/bin/python

import argparse
import config
import datetime
import logging
import sys

config = config.__dict__
from db import Db
db = Db(config)

commands = {}
def command(n):
    commands[n.__name__] = n
    return n


@command
def add_stock(parser):
    parser.add_argument('ticker', type = unicode)
    parser.add_argument('quantity', type = int)
    parser.add_argument('price', type = float)
    parser.add_argument('--date', default = datetime.datetime.utcnow())
    args = parser.parse_args()

    st = db.StockTransaction()
    st.update(ticker = args.ticker, quantity = args.quantity,
            price = args.price, date = args.date)
    st.save()


@command
def load_stock(parser):
    from util.stockLoader import StockLoader
    parser.add_argument('--ticker', type = unicode)
    args = parser.parse_args()

    tickers = []
    if args.ticker:
        tickers.append(args.ticker)
    else:
        for ticker in db.StockTransaction.collection.distinct('ticker'):
            tickers.append(ticker)

    for ticker in tickers:
        logging.info("Loading {}".format(ticker))
        for stockDay in StockLoader.loadStock(ticker):
            sd = db.StockDay()
            sd.update(ticker = ticker, date = stockDay['Date'],
                    open = float(stockDay['Open']),
                    high = float(stockDay['High']),
                    low = float(stockDay['Low']),
                    close = float(stockDay['Close']),
                    volume = float(stockDay['Volume']),
                    adjClose = float(stockDay['Adj Close']))
            sd.save()


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)

    parser = argparse.ArgumentParser(description = "lg stock handling")
    parser.add_argument('command',
            help = "One of: " + ", ".join(commands.keys()))
    if len(sys.argv) > 1:
        commands[sys.argv[1]](parser)
    else:
        # Help at this point
        parser.print_help()
