#! /usr/bin/python

import argparse
import config
import datetime
import logging
import numpy as np
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
        fromDate = db.StockDay.find_one({ 'ticker': ticker },
                sort = [ ('date', -1) ])['date']
        if fromDate is None:
            fromDate = '20090101'
        else:
            fromDate = fromDate.strftime("%Y%m%d")
        for stockDay in StockLoader.loadStock(ticker, fromDate = fromDate):
            sd = db.StockDay()
            sd.update(ticker = ticker, date = stockDay['Date'],
                    open = float(stockDay['Open']),
                    high = float(stockDay['High']),
                    low = float(stockDay['Low']),
                    close = float(stockDay['Close']),
                    volume = float(stockDay['Volume']),
                    adjClose = float(stockDay['Adj Close']))
            sd.save()


@command
def profit(parser):
    parser.add_argument('--ticker', type = unicode)
    parser.add_argument('--as-of')
    args = parser.parse_args()

    tickers = []
    if args.ticker:
        tickers.append(args.ticker.upper())
    else:
        tickers.extend(t
                for t in db.StockTransaction.collection.distinct('ticker'))

    if args.as_of:
        args.as_of = datetime.datetime.strptime(args.as_of, "%Y-%m-%d")
    else:
        args.as_of = datetime.datetime.utcnow()

    profit = 0.0
    totalPrice = 0.0
    for ticker in tickers:
        sharesAndValues = []
        for st in db.StockTransaction.find({ 'ticker': ticker },
                sort = [ ('date', 1) ]):
            if st['quantity'] > 0:
                sharesAndValues.append([ st['quantity'], st['price'],
                        st['transactionPrice'] ])
            else:
                raise NotImplementedError()
        sharesCount = sum([ s[0] for s in sharesAndValues ])
        sharesPrice = sum([ s[0] * s[1] + s[2] for s in sharesAndValues ])
        tProfit = db.StockDay.find_one(
                { 'ticker': ticker, 'date': { '$lte': args.as_of } },
                sort = [ ('date', -1) ])['close'] * sharesCount - sharesPrice
        logging.info("{}: {} ({:.2f}%)".format(ticker, tProfit, 
                100.0 * tProfit / sharesPrice))
        profit += tProfit
        totalPrice += sharesPrice
    logging.info("Total: {} ({:.2f}%)".format(profit, 
            100.0 * profit / totalPrice))


@command
def test(parser):
    parser.add_argument('ticker', type = unicode)
    parser.add_argument('--days', type = int, default = 0)
    parser.add_argument('--gap', type = int, default = 21, help = "How long to "
            "predict in the future, in human days")
    parser.add_argument('--input-days', type = int, default = 10)
    parser.add_argument('--ipython', action = 'store_true')

    args = parser.parse_args()
    args.ticker = args.ticker.upper()

    daysKwargs = {}
    if args.days:
        daysKwargs['limit'] = args.days + args.input_days + args.gap
    days = db.StockDay.find({ 'ticker': args.ticker }, sort = [('date', -1)],
            **daysKwargs)
    days = list(days)

    allInputs = []
    allOutputs = []
    for i in range(len(days)):
        # i is indexed
        outputDay = days[i]
        gapStart = outputDay['date'].date() - datetime.timedelta(
                days = args.gap)
        inputDays = []
        seenFirst = False
        for d in days[i + 1:]:
            if not seenFirst:
                if d['date'].date() <= gapStart:
                    seenFirst = True
                    inputDays.append(d)
            elif len(inputDays) < args.input_days:
                inputDays.append(d)
            else:
                break

        if len(inputDays) != args.input_days:
            # Past end of file, we have no more training cases
            break

        # What we're predicting
        outputs = [ outputDay['low'] / inputDays[0]['high'] - 1.0 ]

        # What we're using for it
        inputs = []
        inputs.append(1.0)
        # Start with first day's absolutes
        inputs.append(inputDays[0]['low'] * 0.01)
        inputs.append(inputDays[0]['high'] * 0.01)
        # Put in other days' absolutes and the time between them and the next
        # data point
        for i, iday in enumerate(inputDays):
            #inputs.append(iday['low'] * 0.01)
            #inputs.append(iday['high'] * 0.01)
            #inputs.append(iday['open'] * 0.01)
            inputs.append(iday['close'] * 0.01)
            #inputs.append(iday['adjClose'] * 0.01)
            inputs.append(iday['volume'] / 100000000.0)
            if i + 1 < len(inputDays):
                inputs.append(iday['high'] / inputDays[i + 1]['low'] - 1.0)
                inputs.append((iday['date'].date()
                        - inputDays[i + 1]['date'].date()).days)

        allInputs.append(inputs)
        allOutputs.append(outputs)

    realInputs = np.array(allInputs)
    realOutputs = np.array(allOutputs)

    from sklearn import linear_model
    from lgSmarts.sklearnNn import nn_model
    def try_model(m, **kwargs):
        clf = m(**kwargs)
        clf.fit(realInputs, realOutputs)
        print(clf.score(realInputs, realOutputs))
        return clf

    def small_try(m, **kwargs):
        clf = m(**kwargs)
        clf.fit(realInputs[:20], realOutputs[:20])
        print("Sample: {}".format(clf.score(realInputs[:20], realOutputs[:20])))
        print("Overall: {}".format(clf.score(realInputs, realOutputs)))
        return clf

    if args.ipython:
        aa = [ [ 0, 0 ], [ 1, 2 ], [ 3, 4 ] ]
        bb = [ [ 0 ], [ 0.5 ], [ 0.75 ] ]
        n = nn_model()
        n.fit(aa, bb)
        print(n.score(aa, bb))
        import IPython
        IPython.embed()
    else:
        small_try(nn_model)


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)

    parser = argparse.ArgumentParser(description = "lg stock handling")
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        parser.add_argument('command')
        commands[sys.argv[1]](parser)
    else:
        # Help at this point
        parser.add_argument('command',
                help = "One of: " + ", ".join(commands.keys()))
        parser.print_help()
