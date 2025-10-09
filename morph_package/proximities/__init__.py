from scipy.spatial import cKDTree
import numpy as np 
import navis 
import networkx as nx 
def compute_proxmities(nv_pre_tree, nv_post_tree, threshold_um=5):
    """
    I still need add the pre and postsynaptic radii here to substract them away from the distances
    calculated
    """
    dend_r = nv_post_tree.nodes['radius'].values.astype(float)
    overlap_list = []
    dend_pos = nv_post_tree.nodes[['x', 'y', 'z']].values.astype(float)
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
    
    
    ax_graph = nv_pre_tree.graph
    groups = [l['ax_node_id'] for l in overlap_list]
    finished_groups = []
    subgraph = ax_graph.subgraph(groups)
    finished_groups = [list(c) for c in nx.connected_components(subgraph.to_undirected())]
    print(len(finished_groups))
    print(list(len(c) for c in finished_groups))
    
    
    """
    # Not sure if this correct 
    pre_pts = pre_skel["vertices"] #[pre_skel["compartment"] == 2] / 1000.0
    post_pts = post_skel["vertices"]#[(post_skel["compartment"] == 1) | (post_skel["compartment"] == 3)] / 1000.0
    
    
    pre_tree = cKDTree(pre_pts)
    post_tree = cKDTree(post_pts)
    found_compartments = 0
    while True:
        # Compute nearest pair (min distance)
        dist, pre_idx = post_tree.query(pre_pts)
        
        min_dist, min_dist_id = np.min(dist),np.argmin(dist)
        # We found another proximity compartment
        
        if min_dist < threshold_um: 
            found_compartments += 1
        
            
            # We extract the compartment_pre_id and its distance 
            compartment_pre_id = min_dist_id
            compartment_min_distance = dist 
            
            # Exclude all points within the exclude radius
            # First we need to calculate the distance of the remaining pre_pts 
            # to the comparment point
            dists_to_found = np.linalg.norm(pre_pts - pre_pts[min_dist_id], axis=1)
            
            contain_mask = dists_to_found > exclude_radius_um
            pre_pts = pre_pts[contain_mask]
            #print(f"{len(pre_pts)} points remaing after compartment removing")
        else:
            break
    print('Found {} compartments'.format(found_compartments))
    """
    