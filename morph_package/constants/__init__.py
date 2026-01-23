from pathlib import Path
import numpy as np 
from caveclient import CAVEclient 

from importlib.resources import files

_pkg_root = files(__package__ or __name__.split(".", 1)[0])  
PKG_DIR = Path(_pkg_root).resolve().parent   

NAVSKEL_FOLDER = PKG_DIR / "data" / "navskel_data"
OVERLAPS_FOLDER = PKG_DIR / "data" / "overlaps_data"

API_VERSION = 1621
API_TOKEN  = "8bb19d9702fb74f6d6d01bfb54b85ba7"

RESOLUTION = np.array([4,4,40])

 
CLIENT  = CAVEclient("minnie65_public",auth_token=API_TOKEN)
# we set the default client version 
CLIENT.version = API_VERSION

def set_api_version(version: int):
    global API_VERSION
    global CLIENT
    API_VERSION = version
    CLIENT.version = API_VERSION
    


LAYER_BOUNDARIES = {
    "l1" : [0, 62.83360145],
    "l23": [62.83360145, 227.2999053],
    "l4" : [227.2999053, 362.8920013],
    "l5" : [362.8920013, 504.76536649],
    "l6" : [504.76536649, 714.45315768],
    }

    





