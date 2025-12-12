from Microns.MicronsBaselineModel.morph_package.microns_api.skeletons import *
from morph_package.proximities import *



def test_get_skeleton():
    test_root_id = 864691136913261297
    
    skeleton = get_skeleton(test_root_id)
    


def test_proximity():
    test_root_id = 864691136913261297
    test_root_id2 = 864691135731649465
    
    
    print(len(compute_proxmities(get_skeleton(test_root_id), get_skeleton(test_root_id2))))
    



    
    