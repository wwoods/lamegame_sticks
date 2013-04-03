
from mongokit import Connection

models = []
from stockDay import StockDay
models.append(StockDay)
from stockTransaction import StockTransaction
models.append(StockTransaction)

def Db(config):
    c = Connection(config['mongo_host'], port = config['mongo_port'])
    c.register(models)
    return c[config['mongo_database']]
