from caveclient import CAVEclient 
from cloudvolume import CloudVolume
import numpy as np 
from .utils import convert_coordinates
import time
import random



_TOKEN = "8bb19d9702fb74f6d6d01bfb54b85ba7"
_VERSION = 1507
CLIENT  = CAVEclient("minnie65_public",auth_token=_TOKEN)
CLIENT.version = _VERSION
_RESOLUTION = np.array([4,4,40])




@convert_coordinates
def get_synapases(pre_pt_root_id=None, post_pt_root_id=None, max_retries=10, base_wait=120):
    """Query synapses with automatic retry when the API is temporarily unavailable."""
    synapse_table_name = CLIENT.info.get_datastack_info()["synapse_table"]
    if pre_pt_root_id is None and post_pt_root_id is None:
        raise ValueError("pre_pt_root_id or post_pt_root_id must be provided.")
    attempt = 0
    while True:
        try:
            # Try to query the API
            if post_pt_root_id and pre_pt_root_id:
                syn_df = CLIENT.materialize.query_table(
                    synapse_table_name,
                    filter_equal_dict={'pre_pt_root_id': pre_pt_root_id},
                    desired_resolution=_RESOLUTION
                )
                syn_df = syn_df[syn_df['post_pt_root_id'] == post_pt_root_id]
            elif post_pt_root_id:
                syn_df = CLIENT.materialize.query_table(
                    synapse_table_name,
                    filter_equal_dict={'post_pt_root_id': post_pt_root_id},
                    desired_resolution=_RESOLUTION
                )
            elif pre_pt_root_id:
                syn_df = CLIENT.materialize.query_table(
                    synapse_table_name,
                    filter_equal_dict={'pre_pt_root_id': pre_pt_root_id},
                    desired_resolution=_RESOLUTION
                )

            # Filter and return
            synapses = syn_df[['ctr_pt_position', 'size', 'pre_pt_root_id', 'post_pt_root_id']]
            synapses = synapses[synapses['size'] > 800]
            return synapses

        except Exception as e:
            attempt += 1
            wait_time = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 3)
            print(f"[get_synapases] Attempt {attempt} failed for pre {pre_pt_root_id}: {e}")
            if attempt >= max_retries:
                print(f"[get_synapases] Giving up after {max_retries} retries.")
                raise
            print(f"Waiting {wait_time:.1f}s before retrying...")
            time.sleep(wait_time)

@convert_coordinates
def get_post_synapses_exclude(post_pt_root_id, exlude_pre_ids):
   
    syn_df = CLIENT.materialize.tables.synapse_target_predictions_ssa(post_pt_root_id=post_pt_root_id).query()
    syn_df = syn_df[~syn_df['pre_pt_root_id'].isin(exlude_pre_ids)]
    syn_df = syn_df[~syn_df['pre_pt_root_id'].isin([post_pt_root_id])]
    return syn_df


@convert_coordinates
def get_post_synapses_include(post_pt_root_id, include_pre_ids):
    syn_df = CLIENT.materialize.tables.synapse_target_predictions_ssa(post_pt_root_id=post_pt_root_id).query()
    syn_df = syn_df[syn_df['pre_pt_root_id'].isin(include_pre_ids)]
    syn_df = syn_df[~syn_df['pre_pt_root_id'].isin([post_pt_root_id])]
    return syn_df