import numpy as np
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP
def sec2ind(s, sr):
    '''
    Convert timestamp to sample

    :param s: float, timestamp in seconds
    :param sr: float, sampling rate
    :return: int: sample that corresponds to the timestamp
    '''
    return int(Decimal(s * sr).quantize(0, ROUND_HALF_UP))