import numpy as np 
import pandas as pd 
from standard_transform import minnie_transform_nm
from morph_package.constants import NAVSKEL_FOLDER, OVERLAPS_FOLDER

_RESOLUTION = np.array([4,4,40])
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
        return transform.apply(np.array(pos) * _RESOLUTION)
    
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



def initalize_navskel_folder(func, *args, **kwargs):
    """
    A decorator to ensure the navskel folder exists before executing the function.
    """
    def wrapper(*args, **kweargs):
        return func(*args, **kweargs)
    
    if not NAVSKEL_FOLDER.exists():
        NAVSKEL_FOLDER.mkdir(parents=True, exist_ok=True)
    return wrapper


def initalize_overlaps_folder(func, *args, **kwargs):
    """
    A decorator to ensure the overlaps folder exists before executing the function.
    """
    def wrapper(*args, **kweargs):
        return func(*args, **kweargs)
    
    if not NAVSKEL_FOLDER.exists():
        OVERLAPS_FOLDER.mkdir(parents=True, exist_ok=True)
    return wrapper
