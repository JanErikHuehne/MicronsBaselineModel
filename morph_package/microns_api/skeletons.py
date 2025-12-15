import numpy as np 
import navis
import pandas as pd
import logging
import pickle 
from cloudvolume import CloudVolume
from tqdm import tqdm 
from scipy.spatial import cKDTree
from standard_transform import minnie_transform_nm
from morph_package.microns_api.utils import initalize_navskel_folder, logged
from morph_package.constants import NAVSKEL_FOLDER,CLIENT, NAVSKEL_FOLDER


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)




CV = CloudVolume(CLIENT.info.segmentation_source(), progress=False, use_https=True)


@logged()
def get_mesh(root_id):
    return CV.mesh.get(root_id)[root_id]

@logged()
def get_skeleton(root_id):
    return CLIENT.skeleton.get_skeleton(root_id)

@logged()
def get_skeletons(root_ids, batch_size=5):
    """Retrieve skeletons in batches (API allows up to 5 IDs per call)."""
    results = {}
    
    # ensure iterable and remove NaN if needed
    clean_ids = root_ids
    
    for i in tqdm(range(0, len(clean_ids), batch_size)):
        batch = clean_ids[i:i + batch_size]
        try:
            
            res = CLIENT.skeleton.get_bulk_skeletons(batch)

            results.update(res)
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed for IDs {batch}: {e}")
    
    return results

@logged()
def convert_skeleton(sk):
    transform = minnie_transform_nm()
    def _transform_position(pos):
        return transform.apply(np.array(pos))
    sk['vertices'] = _transform_position(np.array(sk['vertices']))
    
    # Convert  the radius from nm to um 
    sk['radius']  = np.array(sk['radius']) /1000.0
    
    return sk

@logged()
def batch_convert_skeletons(sks):
    """
    sks is a dict of keys root_id and values skeletons
    """
    csks = {}
    for id,sk in sks.items():
        csks[id] =convert_skeleton(sk)
    return csks

@logged()
@initalize_navskel_folder
def load_navis_skeletons(ids):
    cks = {}
    non_ccached = []
    for id in ids: 
        if not (NAVSKEL_FOLDER / f"{id}.pkl").exists():
           non_ccached.append(id)
        else: 
            with open(NAVSKEL_FOLDER / f"{id}.pkl", 'rb') as f:
                 cks[int(id)] = pickle.load(f)
            
    nc_cks = batch_get_navis_skeletons(batch_convert_skeletons(get_skeletons(non_ccached)))
    return  {**cks, **nc_cks}

@logged()
@initalize_navskel_folder
def batch_get_navis_skeletons(sks):
    csks = {}
    for id,sk in sks.items():
        nv_ksel =get_navis_skeleton(sk)
        with open(NAVSKEL_FOLDER / f"{id}.pkl", 'wb') as f:
            pickle.dump(nv_ksel, f)
        #navis.write_swc(nv_ksel, NAVSKEL_FOLDER / f"{id}.swc")
        """
        with open(NAVSKEL_FOLDER / f"{id}.pkl", 'wb') as f:
            pickle.dump(nv_ksel, f, protocol=pickle.HIGHEST_PROTOCOL)
        """
        csks[int(id)] = nv_ksel
    return csks

@logged()
def map_synapses(tn:navis.TreeNeuron, synapses):
    
    vertices = np.vstack(tn.nodes[['x','y','z']].values)
    syn_positions = np.vstack(synapses['ctr_pt_position'].values)
    sk_tree  = cKDTree(vertices)
    dist, ids = sk_tree.query(syn_positions, k=1)  # nearest neighbor for each synapse
    ids = tn.nodes.iloc[ids]['node_id'].values
    synapses = synapses.copy()
    synapses['nearest_node_id'] = ids
    synapses['distance_to_node'] = dist
    return synapses

@logged()
def get_navis_skeleton(skeleton):
    """
    Convert a skeleton with compartments and radii into a navis.TreeNeuron.
    Expects:
        skeleton['vertices']      : Nx3 array
        skeleton['edges']         : Mx2 array
        skeleton['compartment']   : length-N array (1=soma, 2=axon, 3=dendrite)
        skeleton['radius']        : length-N array
    """
    vertices = np.asarray(skeleton['vertices'])
    edges = np.asarray(skeleton['edges'], dtype=int)
    compartments = np.asarray(skeleton.get('compartment'))
    radii = np.asarray(skeleton.get('radius'))

    if len(vertices) != len(compartments) or len(vertices) != len(radii):
        raise ValueError("Length of 'vertices', 'compartment', and 'radius' must match.")

    # Create node IDs
    node_ids = np.arange(len(vertices))

    # Build parent relationships for SWC format
    parent = np.full(len(vertices), -1, dtype=int)
    for child, par in edges:
        parent[child] = par

    # Build SWC DataFrame (standard navis input)
    swc = pd.DataFrame({
        'node_id': node_ids,
        'compartment': compartments,   # 1=soma, 2=axon, 3=dendrite
        'x': vertices[:, 0],
        'y': vertices[:, 1],
        'z': vertices[:, 2],
        'radius': radii,
        'parent': parent
    })

    # Construct TreeNeuron directly from SWC table
    tn = navis.TreeNeuron(swc, units='microns')

    # Optionally attach metadata
    tn.id = skeleton.get('root_id', None)
    tn.name = f"Skeleton_{tn.id}" if tn.id else "Unnamed_Skeleton"

    return tn

@logged()
def split_branches(x : navis.TreeNeuron) -> list[navis.TreeNeuron]:
    """
    Split a TreeNeuron into its primary branches emanating from the soma.

    This function identifies all nodes directly connected to the soma/root node
    and extracts one subtree per soma-connected process. Each returned
    TreeNeuron contains only the nodes and parent–child relationships belonging
    to that branch, excluding the soma itself. The local root of each subtree is
    reassigned to the node that was originally attached to the soma.
    
    Note that each of these branches is assumed to belong to the same comaprtment. 
    So if the axon split from a dendrite and not from the soma, the information will 
    be lost in this operation. Split the axon and dendrite before proceeding in this case.

    Parameters
    ----------
    x : navis.TreeNeuron
        The input neuron to be split. Must be a single, connected tree
        with a defined root.

    Returns
    -------
    list of navis.TreeNeuron
        A list of TreeNeuron objects, each representing one primary branch
        (i.e., one subtree originating from a direct child of the soma).
        The soma/root node is excluded from all returned branches.

    Notes
    -----
    - The returned TreeNeurons preserve node geometry and units from the input.
    - This operation does not modify the original neuron.
    - The number of branches corresponds to the number of direct soma children.
    - Each branch is an independent TreeNeuron with its own local root.

    Examples
    --------
    >>> branches = split_branches(tn)
    >>> for b in branches:
    ...     print(b.name, b.n_nodes, f"{b.cable_length:.2f} µm")
    >>> navis.plot3d(branches, connectors=False)
    """
    if not isinstance(x, navis.TreeNeuron):
        raise TypeError('x must be a navis.TreeNeuron')
    
    tn = x.copy()
    root = tn.root 
    nodes = tn.nodes.copy()
    root_children = nodes[nodes['parent_id'] == root[0]]['node_id'].values
    branches = []
   
    branch_lengths = []
    for child in root_children:
        # We assume the same compartment for the entire branch 
        comp = nodes.loc[nodes['node_id'] == child, 'compartment'].values[0] 
        cnodes = {child} 
        frontier = {child}  

      
        while frontier:
            new_nodes = set(nodes.loc[nodes.parent_id.isin(frontier), "node_id"].values)
            frontier = new_nodes - cnodes
            cnodes |= new_nodes
        
        branch_df = nodes.loc[nodes.node_id.isin(cnodes)].copy()
        branch_df.loc[branch_df["node_id"] == child, "parent_id"] = -1
        branch_tn = navis.TreeNeuron(
            x=branch_df,
            units=tn.units,
            name=f"{tn.name or 'neuron'}_branch_{child}"
        )
        branch_tn.nodes['compartment'] = comp

        branches.append(branch_tn)
        branch_lengths.append(len(branch_df))

    return branches, root_children
   
@logged()
def reconstruct_neuron(org_neuron, branches):
    if not isinstance(org_neuron, navis.TreeNeuron):
        raise TypeError("org_neuron must be a navis.TreeNeuron")
    if not all(isinstance(b, navis.TreeNeuron) for b in branches):
        raise TypeError("All elements in 'branches' must be TreeNeuron objects")
    
    # --- extract soma/root row from original neuron
    root_row = org_neuron.nodes.loc[org_neuron.nodes["parent_id"] == -1].copy()
    root_id = root_row.iloc[0]["node_id"]

    root_row['node_id'] = 0
    next_id = 1
   
    node_tables = []

    # --- remap and reconnect branches
    for b in branches:
        mapping = {-1: -1}
        b_nodes = b.nodes.copy()

        # assign globally unique node IDs
        for node in b_nodes["node_id"].values:
            mapping[node] = next_id
            next_id += 1

        # substitute IDs
        b_nodes["node_id"] = b_nodes["node_id"].map(mapping)
        b_nodes["parent_id"] = (
            b_nodes["parent_id"].map(mapping)
            .fillna(b_nodes["parent_id"])
            .astype(int)
        )

        # reattach branch root to soma/root_id
        b_nodes.loc[b_nodes["parent_id"] == -1, "parent_id"] = 0

        node_tables.append(b_nodes)
    all_nodes = pd.concat([root_row] + node_tables, ignore_index=True)
    branch_tn = navis.TreeNeuron(
            x=all_nodes,
            units=org_neuron.units,
            name=f"reconstructed_{org_neuron.name or 'neuron'}"
        )
    return branch_tn

@logged()
def clean_resample(x:navis.TreeNeuron, resample_to):
    """
    This method is wrapper arround the navis.resample_skeleton method. 
    It will resample the skeleton except for connections towards the soma 
    (biologically implausible)
    """
    tn = x.copy()
    # We split the neuron first into its branches, then we resample each branch 
    branches, _ = split_branches(tn)
    logging.debug(f"--- Skeletons Interface --- clean_resample : Found {len(branches)} branch away from the soma ")
    logging.debug(f"--- Skeletons Interface --- clean_resample : Resampling each branch to {resample_to}")
    for i in range(len(branches)):
        old_comp =branches[i].nodes['compartment'].values[0]
        old_length = len(branches[i].nodes)
        branches[i] =navis.resample_skeleton(branches[i], resample_to=resample_to)
        branches[i].nodes['compartment'] = old_comp
        new_length = len(branches[i].nodes)
        logging.debug(f"--- Skeletons Interface --- clean_resample : Old length: {old_length} --> New Length {new_length}")
    return reconstruct_neuron(tn, branches)

@logged()
def extract_dend_axon(tn: navis.TreeNeuron,
                      include_soma_in_axon: bool = False,
                      include_soma_in_dend: bool = True) -> tuple[navis.TreeNeuron, navis.TreeNeuron]:
    """
    Split a neuron into axon and dendrite subtrees based on compartment labels.

    Parameters
    ----------
    tn : navis.TreeNeuron
        The neuron to split. Must have a 'compartment' column in `tn.nodes`.
        Convention: 2 = axon, 3 = dendrite, 1 = soma.
    include_soma_in_axon : bool, optional
        If True, include soma nodes (compartment == 1) in the axon TreeNeuron.
    include_soma_in_dend : bool, optional
        If True, include soma nodes (compartment == 1) in the dendrite TreeNeuron.

    Returns
    -------
    axon_tn, dend_tn : tuple of navis.TreeNeuron
        Two TreeNeurons corresponding to axonal and dendritic compartments.
        Each preserves spatial coordinates, radii, and other metadata.

    Notes
    -----
    - Nodes are filtered purely by their 'compartment' label.
    - Edges are rebuilt to include only intra-compartment connections.
    - If no nodes of a given compartment exist, returns None for that branch.
    - The soma can be optionally included in one or both splits.
    """
    if not isinstance(tn, navis.TreeNeuron):

        raise TypeError("tn must be a navis.TreeNeuron")
    if "compartment" not in tn.nodes.columns:
        raise ValueError("tn.nodes must contain a 'compartment' column")

    nodes = tn.nodes.copy()
    
    # identify node sets
    axon_nodes = set(nodes.loc[nodes["compartment"] == 2, "node_id"])
    dend_nodes = set(nodes.loc[nodes["compartment"] == 3, "node_id"])
    soma_nodes = set(nodes.loc[nodes["compartment"] == 1, "node_id"])
    
    if include_soma_in_axon:
        axon_nodes |= soma_nodes
    if include_soma_in_dend:
        dend_nodes |= soma_nodes
   
   
    
    # build helper to generate sub-neuron
    def build_subtree(node_ids: set[int], name_suffix: str):
        if not node_ids:
            return None

        sub_nodes = nodes[nodes["node_id"].isin(node_ids)].copy()
        
         # Reassign parent_id to -1 if parent not in this subtree
        mask = ~sub_nodes["parent_id"].isin(node_ids)
        sub_nodes.loc[mask, "parent_id"] = -1

        return navis.TreeNeuron(
            x=sub_nodes,
            units=tn.units,
            name=f"{tn.name or 'neuron'}_{name_suffix}"
        )

    axon_tn = build_subtree(axon_nodes, "axon")
    dend_tn = build_subtree(dend_nodes, "dendrite")

    return [axon_tn, dend_tn]



