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
    
    





