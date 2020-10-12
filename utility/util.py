import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from enum import Enum
from datetime import datetime


class Color(Enum):
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class DataSetName(Enum):
    CIFAR10 = (1, 10)
    CIFAR100 = (2, 100)
    MNIST = (3, 10)
    FashionMNIST = (4, 10)

def get_current_time():
    dt = datetime.now()
    t = dt.strftime("%Y_%m_%d_%H_%M_%S")
    return t


def get_current_date():
    dt = datetime.now()
    t = dt.strftime("%m_%d")
    return t
