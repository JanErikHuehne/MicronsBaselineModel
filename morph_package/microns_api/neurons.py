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
    name = "aibs_metamodel_celltypes_v661"
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
            xs = np.array(xs, dtype=float)
            assert len(xs.shape) == 1
        except AssertionError as e: 
            raise ValueError("InputClassifier.classify: input xs must be 1D") from e
        except Exception as e: 
            raise ValueError("InputClassifier.classify: unable to convert xs, aborting") from e
        
        if xs.size < self.min_syn:
            warnings.warn("InputClassifier.classify: number of synapses below minimum, returning 'unspecific'")
            return "unspecific"

        
        values, counts = np.unique(xs, return_counts=True)
        count_map = dict(zip(values, counts))
        filtered_counts = np.array([count_map.get(cat, 0) for cat in self.include_partner_classes])
        if np.sum(filtered_counts) <= self.min_syn:
            warnings.warn("InputClassifier.classify: number of synapses from included partner classes  below minimum, returning 'unspecific'")
            return "unspecific"
        else:
            filtered_counts = filtered_counts / np.sum(filtered_counts)
            
        idx_sorted = np.argsort(filtered_counts)[::-1]
        
      
        first_val = counts[idx_sorted[0]]
        second_val = counts[idx_sorted[1]] 
        third_val = counts[idx_sorted[2]]
        rest = 1 - (first_val + second_val + third_val)
        
        if first_val > self.tau_dom and (first_val - second_val) > self.delta:
            return "D-{}".format(self.include_partner_classes[idx_sorted[0]])
        elif (first_val > self.tau_mix) and (second_val > self.tau_mix) and (abs(first_val - second_val) < self.eps) and (rest < self.tau_low):
            return "M-{}-{}".format(self.include_partner_classes[idx_sorted[0]], self.include_partner_classes[idx_sorted[1]])
        elif (first_val > self.tau_tri) and (second_val > self.tau_tri) and (third_val > self.tau_tri):
            return "T-{}-{}-{}".format(self.include_partner_classes[idx_sorted[0]], self.include_partner_classes[idx_sorted[1]], self.include_partner_classes[idx_sorted[2]])
        else:
            return "unspecific"
        
        
        
    
            
        
        
        