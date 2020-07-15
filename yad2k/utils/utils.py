"""Miscellaneous utility functions."""

from functools import reduce
import cv2
import numpy as np


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    # 先将参数传递给funcs中的第一个函数，然后将第一个函数的return 传递给funcs中的第二个函数，以此类推
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')
