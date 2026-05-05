import numpy as np 
import pandas as pd
from cloudvolume import CloudVolume
from scipy.spatial import cKDTree
from functools import wraps
from collections.abc import MutableMapping
from typing import Union, Iterable, Optional
from morph_package.constants import CLIENT
from morph_package.microns_api.utils import convert_coordinates
import warnings

class NeuronTypes(MutableMapping):
    """
    Wrapper around a neuron-types dataframe with filtering utilities.
    """
    def __init__(
        self,
        types_df,
        ty: str,
        *,
        status_df=None,
        ax_status_map=None,
        den_status_map=None
    ):
        super().__init__()
        self.data = types_df
        self.type = ty

        # cache proofreading status (allow injection to avoid repeated queries)
        if ax_status_map is None or den_status_map is None:
            if status_df is None:
                status_df = CLIENT.materialize.query_table("proofreading_status_and_strategy")
            self._status = status_df
            self.ax_status_map = dict(zip(status_df["pt_root_id"], status_df["strategy_axon"]))
            self.den_status_map = dict(zip(status_df["pt_root_id"], status_df["strategy_dendrite"]))
        else:
            self._status = status_df
            self.ax_status_map = ax_status_map
            self.den_status_map = den_status_map

    # MutableMapping interface 
    def __getitem__(self, key): return self.data[key]
    def __setitem__(self, key, value): self.data[key] = value
    def __delitem__(self, key): del self.data[key]
    def __iter__(self): return iter(self.data)
    def __len__(self): return len(self.data)

    def filter_by_type(self, cell_type: Union[str, list]):
        if isinstance(cell_type, str):
            cell_type = [cell_type]
        self.data = self.data[self.data["cell_type"].isin(cell_type)].reset_index(drop=True)

    def filter_by_position(self, pt_position: Union[list, np.ndarray], radius: Union[float, int]):
        """
        Filter to cells within horizontal radius of pt_position (uses x,z; ignores y).
        """
        p = np.asarray(pt_position).reshape(-1)
        cx, cz = float(p[0]), float(p[2])

        coords = np.vstack(self.data["pt_position"].values)
        dist_sq = (coords[:, 0] - cx) ** 2 + (coords[:, 2] - cz) ** 2
        inside = dist_sq <= float(radius) ** 2
        self.data = self.data.loc[inside].reset_index(drop=True)

    def filter_axon_extended(self):
        valid = {"axon_fully_extended", "axon_partially_extended"}
        mask = self.data["pt_root_id"].map(self.ax_status_map).isin(valid)
        self.data = self.data.loc[mask].reset_index(drop=True)

    def filter_dend_extended(self):
        valid = {"dendrite_extended"}
        mask = self.data["pt_root_id"].map(self.den_status_map).isin(valid)
        self.data = self.data.loc[mask].reset_index(drop=True)

    def filter_proofread(self, which: str = "axon"):
        """
        which: 'axon' | 'dendrite' | 'both' | 'either'
        """
        if which == "axon":
            self.filter_axon_extended()
            return
        if which == "dendrite":
            self.filter_dend_extended()
            return

        ax_valid = {"axon_fully_extended", "axon_partially_extended"}
        den_valid = {"dendrite_extended"}

        ax_ok = self.data["pt_root_id"].map(self.ax_status_map).isin(ax_valid)
        den_ok = self.data["pt_root_id"].map(self.den_status_map).isin(den_valid)

        if which == "both":
            mask = ax_ok & den_ok
        elif which == "either":
            mask = ax_ok | den_ok
        else:
            raise ValueError("which must be one of: 'axon', 'dendrite', 'both', 'either'")

        self.data = self.data.loc[mask].reset_index(drop=True)

        
    
        
@convert_coordinates
def get_manual_types():
    
    """
    Subset of nucleus detections in a 100 um column are manually classified by anatomist at the Allen Institute into categories 
    of cell subclasses, fist distinguishing cells into classes of non-neuronal, excitatory and inhibitory; then into subclasses.
    
    23P 
    4P 
    5P-IT
    5P-ET 
    5P-NP 
    6P-IT 
    6P-CT
    BC
    BPC
    MC 
    Unsure
    """
    name = "allen_v1_column_types_slanted_ref"
    return NeuronTypes(CLIENT.materialize.query_table(name), 'manual-type')


@convert_coordinates
def get_ctypes():
    """
    Predictions from soma/nucleus features, sometimes confuses layer 5 inhibitory neurons as being excitatory. 
    
    23P 
    4P 
    5P-IT
    5P-ET 
    5P-NP
    6P-IT 
    6P-CT 
    BC
    BPC
    MC
    NGC 
    OPC 
    astrocypte
    microglia 
    percicyte 
    oligo 
    """
    name = "aibs_metamodel_celltypes_v661"
    return NeuronTypes(CLIENT.materialize.query_table(name), 'c-type')

@convert_coordinates
def get_proofread():
    return CLIENT.materialize.query_table("proofreading_status_and_strategy")

@convert_coordinates
def get_mtypes():
    """
    Excitatory neurons and inhibitory neurons were distinguished with the soma-nucleus model and sublcasses 
    are assigned based on a data-driven clustering of the neuronal features. Inhibitory neurons were classified on 
    how they distributed their synaptic output onto target cells, while excitatory neurons were classified on a collection 
    of dendritic features. 
    
    L2a L2b
    L3a L3b L3c
    L4a L4b L4c 
    L5a L5b L5ET LTNP
    L6a L6b L6c L6CT L6wm 
    
    PTC Periosomatic Targetting, Corresponds to basket cells
    DTC Dendrite Targetting, Most SST cells would be DTc
    STC Sparsly Targetting - Neuroglia cells and L1 interneurons 
    ITC Inhibitory Targetting Cells, Mostly VIP cells
    """
    name = "aibs_metamodel_mtypes_v661_v2"
    return NeuronTypes(CLIENT.materialize.query_table(name), 'm-type')



def get_column_neurons(
    *,
    center=np.array([666.61100865, -196.67104429, 856.69494]),
    radius=75,
    cell_types: Optional[Union[str, list[str]]] = None,
    proofread: str = "axon",   # 'axon'|'dendrite'|'both'|'either'|None
    source: str = "manual",    # 'manual' or 'ctype' 
    verbose: bool = True,
):
    # choose source table
    if source == "manual":
        base = get_manual_types()

        ty = "manual-type"
    elif source == "ctype":
        base = get_ctypes()

        ty = "c-type"
    elif source =='dtype':
        base = get_mtypes()
        ty = "m-type"
    else:
        raise ValueError("source must be 'manual' or 'ctype'")

    status_df = CLIENT.materialize.query_table("proofreading_status_and_strategy")
    ax_map = dict(zip(status_df["pt_root_id"], status_df["strategy_axon"]))
    den_map = dict(zip(status_df["pt_root_id"], status_df["strategy_dendrite"]))

   

    if verbose:
        print(f"--- get_column_neurons --- start: {len(base)}")

    base.filter_by_position(np.asarray(center), radius)
    if verbose:
        print(f"--- get_column_neurons --- in radius: {len(base)}")

    if proofread is not None:
        base.filter_proofread(proofread)
        if verbose:
            print(f"--- get_column_neurons --- proofread({proofread}): {len(base)}")

    
    # Some pt_root_ids were classified as bad because of missing axons or other inconsistencies, we 
    # need to filter them out here, list to be extended in the future 

    EXCLUDE = {
    864691135060769435, 864691136043337814, 864691137021996142,
    864691135942086401, 864691135989595395, 864691135928339726,
    864691135997241770, 864691136275106238, 864691136331104234,
    864691135774202363, 864691135106433613, 864691135274736401,
    864691135571427462, 864691135339982310,
    864691135662059248, 864691136313406781, 864691135510593289,
    864691136620192653, 864691135308295238, 864691135732377785,
    864691135503439170, 864691136125985318, 864691135694189759,
    864691135939209988
        }
    base.data = base.data[~base.data["pt_root_id"].isin(EXCLUDE)].reset_index(drop=True)
   
     
    # We also filter out bad_ids from the refusal list 
    bad_ids = CLIENT.skeleton.get_refusal_list()
    bad_ids =bad_ids[bad_ids['DATASTACK_NAME'] == 'minnie65_phase3_v1']['ROOT_ID'].values.tolist()
    bad_ids = set(bad_ids)
    base.data = base.data[~base.data["pt_root_id"].isin(bad_ids)].reset_index(drop=True)
    
    if verbose:
            print(f"--- get_column_neurons --- filtered: {len(base)}")
    # default: return the full filtered set
    if cell_types is None:
        return base

    # else return per-type subsets
    if isinstance(cell_types, str):
        cell_types = [cell_types]

    out = {}
    for ct in cell_types:
        df_ct = base.data[base.data["cell_type"] == ct].reset_index(drop=True)
        out[ct] = NeuronTypes(df_ct, ty, status_df=status_df, ax_status_map=ax_map, den_status_map=den_map)
        if verbose:
            print(f"--- get_column_neurons --- {ct}: {len(out[ct])}")

    return out



class InputClassifier():
    def __init__(self, include_partner_classes=None):
        self.include_partner_classes = include_partner_classes
        self.tau_dom = 0.5 # dominance fraction 
        self.delta = 0.2 # dominance margin over 2nd 
        self.tau_mix = 0.35 # minium for b oth in a mixed pair 
        self.eps = 0.15 # Closeness of top two 
        self.tau_low = 0.2 # all other must be samll of a mixed pair 
        self.tau_tri = 0.25 # for a triplet all three must at least be this 
        self.min_syn = 1 # minimum synapses to consider
    
    
    def classify(self, xs, mode='type'):
        """
        # Docstring for InputClassifier.classify    
        Classify input synapse categories into 'dominant', 'mixed', 'triplet', or 'unspecific' based on predefined thresholds.
        Parameters:
        xs (iterable): 1D array-like input of synapse categories.
        mode (str): Classification mode, when type cs must contain neuron types, when "raw" xs contains pt_root_id values 
        which will be mapped to neuron types using get_manual_types.

        """
        
        assert mode == 'type', "InputClassifier.classify: only mode='type' is currently supported"
        
        try: 
            xs = np.array(xs)
            assert len(xs.shape) == 1
        except AssertionError as e: 
            raise ValueError("InputClassifier.classify: input xs must be 1D") from e
        except Exception as e: 
            raise ValueError("InputClassifier.classify: unable to convert xs, aborting") from e
        
        if xs.size < self.min_syn:
            warnings.warn("InputClassifier.classify: number of synapses below minimum, returning 'NaN'")
            return "NaN"

        
        values, counts = np.unique(xs, return_counts=True)
        count_map = dict(zip(values, counts))
        filtered_counts = np.array([count_map.get(cat, 0) for cat in self.include_partner_classes])
        if np.sum(filtered_counts) <= self.min_syn:
            warnings.warn("InputClassifier.classify: number of synapses from included partner classes  below minimum, returning 'NaN'")
            return "NaN"
        else:
            filtered_counts = filtered_counts / np.sum(filtered_counts)
        idx_sorted = np.argsort(filtered_counts)[::-1]
        

        first_val = filtered_counts[idx_sorted[0]]
        second_val = filtered_counts[idx_sorted[1]] 
        third_val = filtered_counts[idx_sorted[2]]
        rest = 1 - (first_val + second_val + third_val)
        
        if first_val > self.tau_dom and (first_val - second_val) > self.delta:
            return "D-({})".format(self.include_partner_classes[idx_sorted[0]])
        elif (first_val > self.tau_mix) and (second_val > self.tau_mix) and (abs(first_val - second_val) < self.eps) and (rest < self.tau_low):
            return "M-({})-({})".format(self.include_partner_classes[idx_sorted[0]], self.include_partner_classes[idx_sorted[1]])
        elif (first_val > self.tau_tri) and (second_val > self.tau_tri) and (third_val > self.tau_tri):
            return "T-({})-({})-({})".format(self.include_partner_classes[idx_sorted[0]], self.include_partner_classes[idx_sorted[1]], self.include_partner_classes[idx_sorted[2]])
        else:
            return "unspecific"
        
    def batch_classify(self, xs, mode='type'):
        assert mode =='type', 'InputClassifier.batch_classify: only mode="type" is currently supported'
        results = []
        for x in xs:
            res = self.classify(x, mode=mode)
            results.append(res)
        return results
        
        
        
    
def get_exc_neurons(column_neurons, fine_grained=True):
    """
    Get excitatory neurons in the column, optionally split into fine-grained subclasses. Returns a dict of neuron type to pt_root_id array, and a full array of all excitatory pt_root_ids.
    """
    
    l2a_neurons = column_neurons[column_neurons['cell_type'].isin(['L2a'])]['pt_root_id'].values
    l2b_neurons = column_neurons[column_neurons['cell_type'].isin(['L2b'])]['pt_root_id'].values
    l2c_neurons = column_neurons[column_neurons['cell_type'].isin(['L2c'])]['pt_root_id'].values

    l3a_neurons = column_neurons[column_neurons['cell_type'].isin(['L3a'])]['pt_root_id'].values
    l3b_neurons = column_neurons[column_neurons['cell_type'].isin(['L3b'])]['pt_root_id'].values

    l4a_neurons = column_neurons[column_neurons['cell_type'].isin(['L4a'])]['pt_root_id'].values
    l4b_neurons = column_neurons[column_neurons['cell_type'].isin(['L4b'])]['pt_root_id'].values
    l4c_neurons = column_neurons[column_neurons['cell_type'].isin(['L4c'])]['pt_root_id'].values

    l5et_neurons = column_neurons[column_neurons['cell_type'].isin(['L5ET'])]['pt_root_id'].values
    l5np_neurons = column_neurons[column_neurons['cell_type'].isin(['L5NP'])]['pt_root_id'].values
    l5a_neurons = column_neurons[column_neurons['cell_type'].isin(['L5a'])]['pt_root_id'].values
    l5b_neurons = column_neurons[column_neurons['cell_type'].isin(['L5b'])]['pt_root_id'].values

    l6shorta_neurons = column_neurons[column_neurons['cell_type'].isin(['L6short-a'])]['pt_root_id'].values
    l6shortb_neurons = column_neurons[column_neurons['cell_type'].isin(['L6short-b'])]['pt_root_id'].values
    l6talla_neurons = column_neurons[column_neurons['cell_type'].isin(['L6tall-a'])]['pt_root_id'].values
    l6tallb_neurons = column_neurons[column_neurons['cell_type'].isin(['L6tall-b'])]['pt_root_id'].values
    l6tallc_neurons = column_neurons[column_neurons['cell_type'].isin(['L6tall-c'])]['pt_root_id'].values
            
    if fine_grained:
        return {
            'L2a': l2a_neurons,
            'L2b': l2b_neurons,
            'L2c': l2c_neurons,
            'L3a': l3a_neurons,
            'L3b': l3b_neurons,
            'L4a': l4a_neurons,
            'L4b': l4b_neurons,
            'L4c': l4c_neurons,
            'L5ET': l5et_neurons,
            'L5NP': l5np_neurons,
            'L5a': l5a_neurons,
            'L5b': l5b_neurons,
            'L6short-a': l6shorta_neurons,
            'L6short-b': l6shortb_neurons,
            'L6tall-a': l6talla_neurons,
            'L6tall-b': l6tallb_neurons,
            'L6tall-c': l6tallc_neurons
        }, np.concatenate([l2a_neurons, l2b_neurons, l2c_neurons, l3a_neurons, l3b_neurons, l4a_neurons, l4b_neurons, l4c_neurons, l5et_neurons, l5np_neurons, l5a_neurons, l5b_neurons, l6shorta_neurons, l6shortb_neurons, l6talla_neurons, l6tallb_neurons, l6tallc_neurons])
    else:
        return {
            'L2': np.concatenate([l2a_neurons, l2b_neurons, l2c_neurons]),
            'L3': np.concatenate([l3a_neurons, l3b_neurons]),
            'L4': np.concatenate([l4a_neurons, l4b_neurons, l4c_neurons]),
            'L5ET': l5et_neurons,
            'L5NP': l5np_neurons,
            'L5': np.concatenate([l5a_neurons, l5b_neurons]),
            'L6short': np.concatenate([l6shorta_neurons, l6shortb_neurons]),
            'L6tall': np.concatenate([l6talla_neurons, l6tallb_neurons, l6tallc_neurons])
        }, np.concatenate([l2a_neurons, l2b_neurons, l2c_neurons, l3a_neurons, l3b_neurons, l4a_neurons, l4b_neurons, l4c_neurons, l5et_neurons, l5np_neurons, l5a_neurons, l5b_neurons, l6shorta_neurons, l6shortb_neurons, l6talla_neurons, l6tallb_neurons, l6tallc_neurons])