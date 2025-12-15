from pathlib import Path 
from importlib.resources import files
from morph_package.proximities import *
from morph_package.microns_api.skeletons import * 
from morph_package.constants import set_api_version 
from morph_package.workflows.overlap import workflow_overlap_generation
 
"""
def test_get_skeleton():
    test_root_id = 864691136913261297
    
    skeleton = get_skeleton(test_root_id)
    


def test_proximity():
    test_root_id = 864691136913261297
    test_root_id2 = 864691135731649465
    
    
    print(len(compute_proxmities(get_skeleton(test_root_id), get_skeleton(test_root_id2))))
"""



def test_neuron_overlaps():
    # we set the api version to 1621 to ensure compatibility with the test data
    set_api_version(1621)
    ex_sst = 864691135561699041
    ex_l23 = 864691135774053371
    workflow_overlap_generation(axon_pt_root_ids=[ex_sst], dend_pt_root_ids=[ex_l23])
    pkg_root = files("morph_package")
    expected = pkg_root / "data" / "overlaps_data" / f"pre{ex_sst}_post{ex_l23}.pkl"
    assert expected.exists(), "Overlap File was not created"
  
    
    
    