import logging
from .tao import Tao
from .results import TaoResults
from .eval import TaoEval

logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%m/%d %H:%M:%S",
    level=logging.WARN,
)

__all__ = ["Tao", "TaoResults", "TaoEval"]
