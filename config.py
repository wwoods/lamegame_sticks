
mongo_host = 'localhost'
mongo_port = None
mongo_database = 'lg_sticks'

try:
    from config_local import *
except ImportError:
    pass
