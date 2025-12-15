import numpy as np 
import pandas as pd 
import functools
import logging
import time
from standard_transform import minnie_transform_nm
from morph_package.constants import NAVSKEL_FOLDER, OVERLAPS_FOLDER, RESOLUTION
from functools import wraps

"""
This is a decorator that can be used to automatically convert coordinates of microns 
API tables to um, every field here will be converted if it exists. 
Supported fields are:

- pt_position 
- pre_pt_position 
- post_pt_position 
- ctr_pt_position 

"""
def convert_coordinates(func):

    transform = minnie_transform_nm()
    def _transform_position(pos):
        return transform.apply(np.array(pos) * RESOLUTION)
    
    def wrapper(*args, **kwargs):
        table = func(*args, **kwargs)
        
        if 'pt_position' in table.keys():
            table['pt_position'] = table['pt_position'].apply(_transform_position)
        if 'pre_pt_position' in table.keys():
            table['pre_pt_position'] = table['pre_pt_position'].apply(_transform_position)
        if 'post_pt_position' in table.keys():
            table['post_pt_position'] = table['post_pt_position'].apply(_transform_position)
        if 'ctr_pt_position' in table.keys():
            table['ctr_pt_position'] = table['ctr_pt_position'].apply(_transform_position)
        
        return table 
    return wrapper 




def initalize_navskel_folder(func):
    """
    Decorator: ensure NAVSKEL_FOLDER exists before running the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not NAVSKEL_FOLDER.exists():
            NAVSKEL_FOLDER.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper


def initalize_overlaps_folder(func):
    """
    Decorator: ensure OVERLAPS_FOLDER exists before running the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not OVERLAPS_FOLDER.exists():
            OVERLAPS_FOLDER.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper



def _tictoc():
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0


def logged(level: int = logging.DEBUG):
    """
    Decorator factory.
    Logs entry/exit at `level`, exceptions at ERROR with stack trace.
    Uses the decorated function's module logger.
    """
    def deco(fn):
        log = logging.getLogger(fn.__module__)  # key point: per-module logger

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if log.isEnabledFor(level):
                log.log(level, "→ %s(args=%d, kwargs=%s)", fn.__name__, len(args), list(kwargs.keys()))
            toc = _tictoc()
            try:
                out = fn(*args, **kwargs)
                if log.isEnabledFor(level):
                    log.log(level, "← %s (%.3fs)", fn.__name__, toc())
                return out
            except Exception:
                log.exception("✖ %s failed (%.3fs)", fn.__name__, toc())
                raise

        return wrapper
    return deco