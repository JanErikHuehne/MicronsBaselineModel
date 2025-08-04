import numpy as np 
import caveclient
from standard_transform import minnie_transform_nm

RESOLUTION = np.array([4, 4, 40])




class MinnieData:
    
    def __init__(self, auth_token='8bb19d9702fb74f6d6d01bfb54b85ba7', versin=1300):
        self.client = caveclient.CAVEclient("minnie65_public",auth_token=auth_token)
        self.client.version = versin
        self.resolution = np.array([4,4,40])
        
        self.transform = minnie_transform_nm()
        self._stypes_ext = self._fetch_and_transform('aibs_metamodel_celltypes_v661')
        #print(self._stypes_ext.head())
        self._stypes = self._fetch_and_transform('allen_v1_column_types_slanted_ref')
        self._mtypes = self._fetch_and_transform('allen_column_mtypes_v2')
        #print(self._mtypes.head())
        #print(self._stypes.head())
        #self._stypes_ext = self._fetch_and_transform('aibs_metamodel_celltypes_v661')
        self._nuc = self._fetch_and_transform('nucleus_detection_v0')
        self._mtypes_ext = self._fetch_and_transform('aibs_metamodel_mtypes_v661_v2')
        self._status = self.client.materialize.query_table("proofreading_status_and_strategy")
        self.status_map = dict(zip(self._status['pt_root_id'], self._status['strategy_axon']))
        self.den_status_map = dict(zip(self._status['pt_root_id'], self._status['strategy_dendrite']))
    def _fetch_and_transform(self, table_name):
        df = self.client.materialize.query_table(table_name).copy()
        df['pt_position'] = df['pt_position'].apply(self._transform_position)
        return df
    
    def _transform_position(self, pos):
        return self.transform.apply(np.array(pos) * self.resolution)
    
    def _filter_cells(self, df, types):
        return df[df['cell_type'].isin(types)][['pt_root_id', 'pt_position']]
    
    def _filter_fully_extended(self, pt_root_ids):
        return [
            rid for rid in pt_root_ids
            if self.status_map.get(rid) in ['axon_fully_extended', 'axon_partially_extended']
        ]
    
    def _filter_fully_extended_dend(self, pt_root_ids):
        return [
            rid for rid in pt_root_ids
            if self.den_status_map.get(rid) == 'dendrite_extended'
        ]
        
        
    @property
    def mtypes(self):
        return self._mtypes
    
    @property
    def mtypes_extended(self):
        return self._mtypes_ext
    
    def position(self, pt_root_ids):
        return self._mtypes_ext[self._mtypes_ext['pt_root_id'].isin(pt_root_ids)]['pt_position'].values
    
    def status(self, pt_root_ids):
        return self._status[ self._status['pt_root_id'].isin(pt_root_ids)]['strategy_axon'].values
    def get_cells_mtype(self, layer, extended=False):
        df = self._mtypes_ext if extended else self._mtypes
        layer_map = {
            'l2': ['L2a', 'L2b', 'L2c'],
            'l3': ['L3a', 'L3b'],
            'l4': ['L4a', 'L4b', 'L4c'],
            'l5': ['L5a', 'L5b', 'L5ET', 'L5NP'],
            'l6': ['L6tall-a', 'L6tall-b', 'L6tall-c', 'L6short-a', 'L6short-b', 'L6wm'],
            'dtc': ['DTC'],
            'ptc': ['PTC'],
        }
        
        
        types = layer_map.get(layer.lower())
        
        if not types:
            raise ValueError(f"Unknown layer: {layer}")
        return self._filter_cells(df, types)
    
    def get_cells_stype(self, layer, extended=False):
        df = self._stypes_ext if extended else self._stypes
        layer_map = {
            'l23': ['23P'],
            'l4':  ['4P'],
            'l5et':  ['5P-ET'],
            'l5it':  ['5P-IT'],
            'l5np':  ['5P-NP'],
            'l6it':  ['6P-IT'],
            'l6ct':  ['6P-CT'],
            'bc':  ['BC'],
            'vip': [ 'BPC'],
            'sst':[ 'MC'],
           
        }
        
        
        types = layer_map.get(layer.lower())
        
        if not types:
            raise ValueError(f"Unknown layer: {layer}")
        return self._filter_cells(df, types)
    
    
    def get_fully_dendrite(self, ids):
        return self._filter_fully_extended_dend(ids)
    def get_fully_extended(self, ids):
        return self._filter_fully_extended(ids)
    
    
    def filter_nuc_neurons(self, pt_root_ids):
        filtered = []
        for pt_root_id in pt_root_ids:
            if len(self._nuc[self._nuc['pt_root_id'] == pt_root_id]) == 1:
                filtered.append(pt_root_id)
            else:
                print(self._nuc[self._nuc['pt_root_id'] == pt_root_id])
        return filtered
    def get_dorkenwald_neurons_l23(self):
        
        lost_neurons = [864691134919349258,
                        864691134941217635,
                        864691135059995035,
                        864691135066700868,
                        864691135257638831,
                        864691135335733481,
                        864691135368459001,
                        864691135464044445,
                        864691135473351730,
                        864691135519105930,
                        864691135565870679,
                        864691135577012766,
                        864691135577051678,
                        864691135594774315,
                        864691135694281919,
                        864691135726939455,
                        864691135953669667,
                        864691136135954059,
                        864691136389921911,
                        864691136424566831,
                        864691136445340035]
        l23_neurons = self.get_cells_mtype(layer='l2', extended=True)
        
        lost_neurons_in = True
        for lost_neuron in lost_neurons:
            if lost_neuron not in l23_neurons['pt_root_id'].values:
                lost_neurons_in = False
                
        print("Lost neurons in inital l23 neurons:", lost_neurons_in)
        l23_pos = np.vstack(l23_neurons['pt_position'].values)
        radius = 75  # 75 µm in mm
        
        x_coords = l23_pos[:, 0]
        z_coords = l23_pos[:, 2]
        
        
        
        p1 =[681505.5 , 141139.95, 856694.94]
        p2 = [768661.44, 1137334.5 ,  856694.94]

        p1 = self.transform.apply([p1])
        p2 = self.transform.apply([p2])
        print(p1, p2)
        # Circle center in XZ-plane
        cx, cz = p1[0,0], p1[0,2]
        dist_sq = (x_coords - cx) ** 2 + (z_coords - cz) ** 2
        
        # Compare to squared radius (to avoid unnecessary sqrt)
        inside_circle = dist_sq <= radius ** 2  # Boolean array
        l23_valid = l23_neurons['pt_root_id'].values[inside_circle]
        l23_faxon = self.get_fully_extended(l23_valid)
        #l23_fdend  = self.get_fully_dendrite(l23_valid)
        return {'pre': l23_faxon, 'post': self.filter_nuc_neurons(l23_valid)}
    
"""     
def setup():
    return_dict = {}
    client=caveclient.CAVEclient("minnie65_public",auth_token='8bb19d9702fb74f6d6d01bfb54b85ba7')
    materialization = 1412
    client.version = materialization
    
    return_dict['client'] = client 
    return_dict['mtypes'] = client.materialize.query_table('allen_column_mtypes_v2')
    return_dict['mtypes_extended'] = client.materialize.query_table('aibs_metamodel_mtypes_v661_v2')

    return_dict['status'] = client.materialize.query_table('proofreading_status_and_strategy')
    # Given resolution and transform
    resolution = np.array([4, 4, 40])
    transf = minnie_transform_nm()
    
    # Define a helper function to scale and transform a position
    def transform_position(pos):
        # Ensure pos is a numpy array (if it's not already)
        pos_arr = np.array(pos)
        pos_scaled = pos_arr * resolution
        return transf.apply(pos_scaled)
        # Apply transformation to each 'pt_position' in both dataframes
    return_dict['mtypes']['pt_position'] = return_dict['mtypes']['pt_position'].apply(transform_position)
    return_dict['mtypes_extended']['pt_position'] = return_dict['mtypes_extended']['pt_position'].apply(transform_position)
    
    
    
    return_dict['l23']  = return_dict['mtypes'][return_dict['mtypes']['cell_type'].isin(['L2a','L2b', 'L2c', 'L3a', 'L3b']) ]['pt_root_id'].values
    return_dict['l23ex']  = return_dict['mtypes_extended'][return_dict['mtypes_extended']['cell_type'].isin(['L2a','L2b', 'L2c', 'L3a', 'L3b']) ]['pt_root_id'].values
    fully_extended_l23 = []
    for l23 in return_dict['l23']:

        if return_dict['status'][ return_dict['status']['pt_root_id'] == l23]['strategy_axon'].values[0] in ['axon_fully_extended']:
            fully_extended_l23.append(l23)
    fully_dendrite_l23 = []
    for l23 in return_dict['l23ex']:
        if return_dict['status'][ return_dict['status']['pt_root_id'] == l23]['strategy_axon'].values and return_dict['status'][ return_dict['status']['pt_root_id'] == l23]['strategy_axon'].values[0] in ['axon_fully_extended']:
            fully_dendrite_l23.append(l23)
            
    return_dict['fully_extended_l23'] = fully_extended_l23
    return_dict['fully_dendrite_l23'] = fully_dendrite_l23
    print(len(fully_dendrite_l23))
    return_dict['l2']  = return_dict['mtypes'][return_dict['mtypes']['cell_type'].isin(['L2a','L2b', 'L2c']) ]['pt_root_id'].values
    return_dict['l3']  = return_dict['mtypes'][return_dict['mtypes']['cell_type'].isin(['L3a', 'L3b']) ]['pt_root_id'].values
    return_dict['l4']  = return_dict['mtypes'][return_dict['mtypes']['cell_type'].isin(['L4a','L4b', 'L4c']) ]['pt_root_id'].values
    return_dict['l4ex']  = return_dict['mtypes_extended'][return_dict['mtypes_extended']['cell_type'].isin(['L4a','L4b', 'L4c']) ]['pt_root_id'].values
    
    fully_extended_l4 = []
    for l4 in return_dict['l4']:

        if return_dict['status'][ return_dict['status']['pt_root_id'] == l4]['strategy_axon'].values[0] in ['axon_fully_extended']:
            fully_extended_l4.append(l4)
    return_dict['fully_extended_l4'] = fully_extended_l4
    
    fully_dendrite_l4 = []
    for l4 in return_dict['l4ex']:
        if return_dict['status'][ return_dict['status']['pt_root_id'] == l4]['strategy_axon'].values and return_dict['status'][ return_dict['status']['pt_root_id'] == l4]['strategy_axon'].values[0] in ['axon_fully_extended']:
            fully_dendrite_l4.append(l4)
            
    return_dict['fully_dendrite_l4'] = fully_dendrite_l4
    
    
    return_dict['l5']  = return_dict['mtypes'][return_dict['mtypes']['cell_type'].isin(['L5a','L5b', 'L5ET', 'L5NP']) ]['pt_root_id'].values
    return_dict['l6']  = return_dict['mtypes'][return_dict['mtypes']['cell_type'].isin(['L6tall-a','L6tall-b', 'L6tall-c', 'L6short-a', 'L6short-b','L6wm']) ]['pt_root_id'].values
    return_dict['dtc']  = return_dict['mtypes'][return_dict['mtypes']['cell_type'].isin(['DTC']) ]['pt_root_id'].values
    return_dict['ptc']  = return_dict['mtypes'][return_dict['mtypes']['cell_type'].isin(['PTC']) ]['pt_root_id'].values
    
   
    return_dict['l2ex']  = return_dict['mtypes_extended'][return_dict['mtypes_extended']['cell_type'].isin(['L2a','L2b', 'L2c']) ]['pt_root_id'].values
    return_dict['l3ex']  = return_dict['mtypes_extended'][return_dict['mtypes_extended']['cell_type'].isin(['L3a', 'L3b']) ]['pt_root_id'].values
    return_dict['l4ex']  = return_dict['mtypes_extended'][return_dict['mtypes_extended']['cell_type'].isin(['L4a','L4b', 'L4c']) ]['pt_root_id'].values
    return_dict['l5ex']  = return_dict['mtypes_extended'][return_dict['mtypes_extended']['cell_type'].isin(['L5a','L5b', 'L5ET', 'L5NP']) ]['pt_root_id'].values
    return_dict['l6ex']  = return_dict['mtypes_extended'][return_dict['mtypes_extended']['cell_type'].isin(['L6tall-a','L6tall-b', 'L6tall-c', 'L6short-a', 'L6short-b','L6wm']) ]['pt_root_id'].values
    return_dict['dtcex']  = return_dict['mtypes_extended'][return_dict['mtypes_extended']['cell_type'].isin(['DTC']) ]['pt_root_id'].values
    return_dict['ptcex']  = return_dict['mtypes_extended'][return_dict['mtypes_extended']['cell_type'].isin(['PTC']) ]['pt_root_id'].values


    dtc_dat = return_dict['mtypes'][return_dict['mtypes']['cell_type'].isin(['DTC']) ][['pt_root_id', 'pt_position']].values
    dtc_positions = np.array(dtc_dat[:,1].tolist())
    valid_upper_dtc = np.argwhere(dtc_positions[:,1] < 300)[:,0]
    valid_upper_dtc = return_dict['dtc'][valid_upper_dtc]
    
    return_dict['upper_dtc'] = valid_upper_dtc
    for dt in return_dict['upper_dtc']:
        print(return_dict['status'][ return_dict['status']['pt_root_id'] == dt]['strategy_axon'].values[0])
    return return_dict




info = setup()
"""