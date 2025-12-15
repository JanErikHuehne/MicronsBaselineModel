import numpy as np 
import navis 
import networkx as nx 
from scipy.spatial import cKDTree
from morph_package.microns_api.synapses import get_synapases
from morph_package.microns_api.skeletons import map_synapses

class OverlapColletion():
    def __init__(self,overlaps, axon_tree, dend_tree, axon_pt_root_id, dend_pt_root_id, **kwargs):
        self.axon_pt_root_id = axon_pt_root_id
        self.dend_pt_root_id = dend_pt_root_id
        self.axon_tree = axon_tree
        self.dend_tree = dend_tree
        self.synapses = get_synapases(axon_pt_root_id, dend_pt_root_id)
        if not self.synapses.empty:
            self.synapses = map_synapses(axon_tree, self.synapses)
        else:
            self.synapses = []
        self.overlaps = [Overlap(o[0],o[1], self) for o in overlaps]
       
        
    def get_overlaps(self):
        """
        Returns a list of overlap lengths. The length of an overlap is determined on the presynaptic 
        side. For more information see morph_pckage.proximities.Overlap.get_overlap_length method.
        """
        return [c.total_length for c in self.overlaps]
    
    def post_soma_distances(self):
        """
        Returns a list of distances from the soma to the closest postsynaptic node in each overlap group.
        """
        return [c.post_soma_distance for c in self.overlaps]
    
    
    def synapses_in_overlaps(self):
        if len(self.synapses) > 0: 
            total_overlap_syn = 0
            for o in self.overlaps:
                total_overlap_syn += len(o.synapses)
            return total_overlap_syn
        else:
            return 0
class Overlap():
    def __init__(self,axon_nodes, dend_nodes, collection):
        self.collection = collection
        self.axon_nodes = axon_nodes
        self.dend_nodes = dend_nodes
        # We extract the length of the overlaping sections at init time
        self.total_length = None
        self.get_overlap_length()
        self.post_soma_distance = None
        # We extract the postsynaptic soma distance at init time
        self.get_post_soma_distance()
        self.synapses = None
        # We extract the associated synapses at init time
        self.get_associated_synapses()
        
    def get_overlap_length(self):
        """
        Compute total axonal cable length involved in each proximity group (presynaptic perperspective).

        Parameters
        ----------
        tn : TreeNeuron-like object
            Must have:
                - tn.nodes: DataFrame with columns ['node_id', 'x', 'y', 'z']
                - tn.edges: ndarray of shape (n_edges, 2) with [parent_id, child_id]
        overlap_groups : list of list of int
            Groups of node IDs identified as close to postsynaptic targets.

        Returns
        -------
        overlap_info : list of overlap lengths
        """
        # Map node_id → coordinates
        tn = self.collection.axon_tree
        node_coords = tn.nodes.set_index('node_id')[['x', 'y', 'z']].astype(float)
        node_coords = node_coords.to_dict('index')

        
    
        group_set = set(self.axon_nodes)
        total_length = 0.0

        # iterate through all edges
        for parent, child in tn.edges:
            if parent in group_set and child in group_set:
                p_pos = np.array(list(node_coords[parent].values()))
                c_pos = np.array(list(node_coords[child].values()))
                total_length += np.linalg.norm(p_pos - c_pos)

        self.total_length = total_length

    
    def get_post_soma_distance(self):
        soma_node = self.collection.dend_tree.root
        distances = []
        for d_node in self.dend_nodes:
            d_geo = navis.dist_between(self.collection.dend_tree, soma_node, d_node) 
            distances.append(d_geo)
            
        self.post_soma_distance = min(distances)
    
    
    def get_associated_synapses(self):
        """
        Returns a DataFrame of synapses associated with this overlap object.
        """
        if len(self.collection.synapses) == 0:
            self.synapses = None
            return
        syns = self.collection.synapses
        mask = syns['nearest_node_id'].isin(self.axon_nodes)
        self.synapses = syns[mask]

        
        
def compute_proxmities(nv_pre_tree, nv_post_tree, threshold_um=5, **kwargs):
    """
    I still need add the pre and postsynaptic radii here to substract them away from the distances
    calculated
    """
    dend_r = nv_post_tree.nodes['radius'].values.astype(float)
    dend_nodes = nv_post_tree.nodes['node_id'].values
    overlap_list = []
    dend_pos = nv_post_tree.nodes[['x', 'y', 'z']].values.astype(float)
    ax_dend_dict = {}
    for _,ax_node in nv_pre_tree._get_nodes().iterrows():
        ax_node_pos = ax_node[['x', 'y', 'z']].values[np.newaxis, ...].astype(float)
        ax_node_id = ax_node['node_id']
        ax_r = float(ax_node['radius'])
        center_dists = np.linalg.norm(dend_pos - ax_node_pos, axis=1).astype(float)
        surface_dists = center_dists - (ax_r + dend_r)
        min_d = surface_dists.min()

        close_mask = surface_dists <= threshold_um
        if np.any(close_mask):
            overlap_list.append({'ax_node_id': ax_node_id, 'min_dist': min_d})
            ax_dend_dict[ax_node_id] = set(int(d) for d in dend_nodes[close_mask])
    
    ax_graph = nv_pre_tree.graph
    groups = [l['ax_node_id'] for l in overlap_list]
    finished_groups = []
    subgraph = ax_graph.subgraph(groups)
    finished_groups = [list(int(cc) for cc in c) for c in nx.connected_components(subgraph.to_undirected())]
    dendrite_groups = []
    for group in finished_groups:
        dend_set = set()
        for ax_node in group:
            dend_set.update(ax_dend_dict[ax_node])
        dendrite_groups.append(dend_set)
        
    return OverlapColletion([(a,list(d)) for a,d in zip(finished_groups, dendrite_groups)],nv_pre_tree, nv_post_tree, **kwargs)
        
   
    

    