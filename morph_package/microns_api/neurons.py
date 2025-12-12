from caveclient import CAVEclient 
from cloudvolume import CloudVolume
import numpy as np 
from scipy.spatial import cKDTree
from collections.abc import MutableMapping
from typing import Union

from .utils import convert_coordinates
_TOKEN = "8bb19d9702fb74f6d6d01bfb54b85ba7"
_VERSION = 1507
CLIENT  = CAVEclient("minnie65_public",auth_token=_TOKEN)


class NeuronTypes(MutableMapping):
    """
    This is a wrapper class for neuron types df from the microns dataset to expose additional filtering methods. 
    """
    def __init__(self,types_dict, ty):
        super().__init__()
        self.data = types_dict
        self.type = ty 

        # Storing the proofreading status in here 
        self._status = CLIENT.materialize.query_table("proofreading_status_and_strategy")
        self.ax_status_map = dict(zip(self._status['pt_root_id'], self._status['strategy_axon']))
        self.den_status_map = dict(zip(self._status['pt_root_id'], self._status['strategy_dendrite']))
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def filter_by_type(self,  cell_type:Union[str,list]):
        if cell_type == str:
            cell_type = [cell_type]
        self.data = self.data[self.data['cell_type'].isin(cell_type)]
        
        return None
    def filter_by_position(self, pt_position: Union[list, np.ndarray], radius:Union[float, int]):
        """
        This filter method will filter out all the cells that are within a horizontal radius 
        of a specific coordinates. Only the first and third coordinates are used, the y coordinates (depth)
        will be ignored.  
        """
        pt_position = np.array(pt_position)
        coords = np.vstack(self.data['pt_position'].values)
        cx, cz = pt_position[0,0], pt_position[0,2]
        dist_sq = (coords[:,0] - cx) ** 2 + (coords[:,2] - cz) ** 2
        inside_circle = dist_sq <= radius ** 2
        self.data = self.data.loc[inside_circle].reset_index(drop=True)
        
        return None
    
    def filter_axon_extended(self):
        valid_statuses = {'axon_fully_extended', 'axon_partially_extended'}
        mask = self.data['pt_root_id'].map(self.ax_status_map).isin(valid_statuses)
        self.data = self.data.loc[mask].reset_index(drop=True)
       
    
    def filter_dend_extended(self):
        valid_statuses = {'dendrite_extended'}
        mask = self.data['pt_root_id'].map(self.den_status_map).isin(valid_statuses)
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



def column_neurons():
    """
    This method uses the interface to filter out proofread p23, mc and bc in the proofread column 
    of the allen microns dataset.
    """
    
    l23_neurons = get_manual_types()
    print("--- Neuron Interface --- column_neurons : Filtering L23 neurons, initally found neurons: ", len(l23_neurons))
    l23_neurons.filter_by_type(['23P'])
    print("--- Neuron Interface --- column_neurons : L23 neurons: ", len(l23_neurons))
    l23_neurons.filter_by_position(np.array([[666.61100865, -196.67104429,  856.69494 ]]), 75)
    print("--- Neuron Interface --- column_neurons : L23 neurons within radius: ", len(l23_neurons))
    l23_neurons.filter_axon_extended()
    print("--- Neuron Interface --- column_neurons : L23 with axon proofread: ", len(l23_neurons))
    mc_neurons = get_manual_types()
    print("--- Neuron Interface --- column_neurons : Filtering MC neurons, initally found neurons: ", len(mc_neurons))
    mc_neurons.filter_by_type(['MC'])
    print("--- Neuron Interface --- column_neurons : MC neurons: ", len(mc_neurons))
    mc_neurons.filter_by_position(np.array([[666.61100865, -196.67104429,  856.69494 ]]), 75)
    print("--- Neuron Interface --- column_neurons : MC neurons within radius: ", len(mc_neurons))
    mc_neurons.filter_axon_extended()
    print("--- Neuron Interface --- column_neurons : MC with axon proofread: ", len(mc_neurons))
    bc_neurons = get_manual_types()
    print("--- Neuron Interface --- column_neurons : Filtering BC neurons, initally found neurons: ", len(bc_neurons))
    bc_neurons.filter_by_type(['BC'])
    print("--- Neuron Interface --- column_neurons : BC neurons: ", len(bc_neurons))
    bc_neurons.filter_by_position(np.array([[666.61100865, -196.67104429,  856.69494 ]]), 75)
    print("--- Neuron Interface --- column_neurons : BC neurons within radius: ", len(bc_neurons))
    bc_neurons.filter_axon_extended()
    print("--- Neuron Interface --- column_neurons : BC with axon proofread: ", len(bc_neurons))
    vip_neurons = get_manual_types()
    print("--- Neuron Interface --- column_neurons : Filtering L23 neurons, initally found neurons: ", len(vip_neurons))
    vip_neurons.filter_by_type(['BPC'])
    print("--- Neuron Interface --- column_neurons : L23 neurons: ", len(vip_neurons))
    vip_neurons.filter_by_position(np.array([[666.61100865, -196.67104429,  856.69494 ]]), 75)
    print("--- Neuron Interface --- column_neurons : L23 neurons within radius: ", len(vip_neurons))
    vip_neurons.filter_axon_extended()
    
    return [l23_neurons, mc_neurons, bc_neurons]