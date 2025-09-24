from .binance_ws import BinanceWS
from .crypto_connector import *
from .stock_connector import *

try:  # Optional NOWNodes support
    from .nownodes_ws import *  # noqa: F401,F403
except RuntimeError:
    pass
except ImportError:
    pass

SOURCES = {
    "binance": BinanceWS,
    # Add other sources as needed
}
