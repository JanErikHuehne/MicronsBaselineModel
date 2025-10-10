from caveclient import CAVEclient 
from cloudvolume import CloudVolume
import numpy as np 
from .utils import convert_coordinates




_TOKEN = "8bb19d9702fb74f6d6d01bfb54b85ba7"
_VERSION = 1507
CLIENT  = CAVEclient("minnie65_public",auth_token=_TOKEN)
CLIENT.version = _VERSION
_RESOLUTION = np.array([4,4,40])


@convert_coordinates
def get_synapases(pre_pt_root_id, post_pt_root_id=None):
    synapse_table_name = CLIENT.info.get_datastack_info()["synapse_table"]
    if post_pt_root_id:
        syn_df = CLIENT.materialize.query_table(synapse_table_name, filter_equal_dict={'pre_pt_root_id' : pre_pt_root_id}, desired_resolution=_RESOLUTION)
        syn_df = syn_df[syn_df['post_pt_root_id'] == post_pt_root_id]
    else:
        syn_df = CLIENT.materialize.query_table(synapse_table_name, filter_equal_dict={'pre_pt_root_id' : pre_pt_root_id}, desired_resolution=_RESOLUTION)
    synapses = syn_df[['ctr_pt_position', 'size', 'pre_pt_root_id', 'post_pt_root_id']]
    synapses = synapses[synapses['size'] > 800]
    return synapses