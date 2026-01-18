import numpy as np 
import navis 
import pickle
import networkx as nx 
import pandas as pd 
from pathlib import Path 
from tqdm import tqdm 
from scipy.spatial import cKDTree
from morph_package.constants import OVERLAPS_FOLDER
from morph_package.microns_api.synapses import get_synapases
from morph_package.microns_api.skeletons import map_synapses
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any, Sequence

Pair = Tuple[int, int]
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
        return [c.get_post_soma_distance() for c in self.overlaps]
    
    
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
        return distances 
    
    
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
        
   

@dataclass
class OverlapRecord: 
    axon_id: int
    dend_id: int
    
    overlap_lengths: List[float]
    post_soma_distances: List[List[float]]
    
    total_overlap_length: float 
    n_overlap_groups: int 
    
    n_synapses_total: int 
    synapses : pd.DataFrame
    cache_path: Optional[str]
    
    def to_dict(self) -> dict:
        d = asdict(self)
        return d 
    

class OverlapDataset: 
    """
    Stores overlap summaries keyed by (axon_id, dend_id)
    
    Internal storage:
        self._record[(axon_id, dend_id)] = OverlapRecord(...)
    """
    
    def __init__(self):
        self._records: Dict[Pair, OverlapRecord] = {}
    
    
    def __len__(self) -> int:
        return len(self._records)
    
    def add(
        self,
        pairs: Union[Pair, Iterable[Pair]],
        *,
        overwrite: bool = True,
    ):
        # normalize pairs
        if isinstance(pairs, tuple) and len(pairs) == 2 and all(isinstance(x, int) for x in pairs):
            pairs_iter = [(int(pairs[0]), int(pairs[1]))]
        else:
            pairs_iter = [(int(a), int(d)) for (a, d) in pairs]  # type: ignore
        # avoid pointless work
        if not overwrite:
            pairs_iter = [p for p in pairs_iter if p not in self._records]
        added: List[Pair] = []
        for axon_id, dend_id in pairs_iter:
            key = (axon_id, dend_id)
            pkl_path = OVERLAPS_FOLDER / f"pre{axon_id}_post{dend_id}.pkl"
            collection = self._load_collection(pkl_path)
            rec = self._record_from_collection(
                axon_id=axon_id,
                dend_id=dend_id,
                collection=collection,
                cache_path=str(pkl_path),
            )
            self._records[key] = rec
            added.append(key)
        return added
            
    def add_parallel(
        self,
        pairs: Union[Pair, Iterable[Pair]],
        *,
        overwrite: bool = True,
        max_workers: int = 30,
        strict: bool = False,
    ) -> List[Pair]:
        """
        Parallel add using threads. Good for loading/unpickling many pickles.

        - strict=False: skip failures
        - strict=True: raise first exception encountered
        """
        # normalize pairs
        if isinstance(pairs, tuple) and len(pairs) == 2 and all(isinstance(x, int) for x in pairs):
            pairs_iter = [(int(pairs[0]), int(pairs[1]))]
        else:
            pairs_iter = [(int(a), int(d)) for (a, d) in pairs]  # type: ignore

        # avoid pointless work
        if not overwrite:
            pairs_iter = [p for p in pairs_iter if p not in self._records]
        print('Computing overlaps for {} pairs'.format(len(pairs_iter)))
        def worker(axon_id: int, dend_id: int):
            key = (axon_id, dend_id)
            pkl_path = OVERLAPS_FOLDER / f"pre{axon_id}_post{dend_id}.pkl"
            collection = self._load_collection(pkl_path)
            rec = self._record_from_collection(
                axon_id=axon_id,
                dend_id=dend_id,
                collection=collection,
                cache_path=str(pkl_path),
            )
            return key, rec

        added: List[Pair] = []
        # sensible default if user passes nonsense
        max_workers = int(max_workers) if max_workers is not None else 8
        if max_workers <= 0:
            max_workers = 8

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(worker, ax, de): (ax, de) for ax, de in pairs_iter}
            print(len(futures))
            for fut in tqdm(as_completed(futures), total=len(futures),desc="Loading overlaps"):
               
                try:
                    key, rec = fut.result()
                    # single-threaded mutation is safe here (we only mutate in this loop)
                    self._records[key] = rec
                    added.append(key)
                except Exception as e:
                    if strict:
                        raise
                    print(e)
                    continue

        return added

    
    @staticmethod
    def _load_collection(path: Path) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)
        
    def _record_from_collection(
        self,
        *,
        axon_id: int,
        dend_id: int,
        collection: Any,
        cache_path: Optional[str],
    ) -> OverlapRecord:
        """
        Extract only the pieces you want from OverlapCollection.
        Assumes OverlapCollection provides:
            - get_overlaps() -> list[float]  (per overlap group length)
            - post_soma_distances() -> list[list[float]]
            - synapses: DataFrame or []/None
            - synapses_in_overlaps() -> int
        """
        overlap_lengths = list(map(float, collection.get_overlaps()))
        post_soma_distances = [[float(x) for x in overlap] for overlap in collection.post_soma_distances()]

        total_overlap_length = float(np.sum(overlap_lengths)) if overlap_lengths else 0.0

        n_overlap_groups = int(len(overlap_lengths))

        syn_obj = getattr(collection, "synapses", None)

        # collection.synapses is either a DataFrame (your code) or [] when empty
        if isinstance(syn_obj, pd.DataFrame):
            n_synapses_total = int(len(syn_obj))
            synapses_df = syn_obj.copy(deep=False) 
        elif syn_obj is None:
            n_synapses_total = 0
            synapses_df = None
        else:
            # [] or other iterable----
            try:
                n_synapses_total = int(len(syn_obj))
            except Exception:
                n_synapses_total = 0
            synapses_df = None

      
        return OverlapRecord(
            axon_id=axon_id,
            dend_id=dend_id,
            overlap_lengths=overlap_lengths,
            post_soma_distances=post_soma_distances,
            total_overlap_length=total_overlap_length,
            n_overlap_groups=n_overlap_groups,
            n_synapses_total=n_synapses_total,
            synapses=synapses_df,
            cache_path=cache_path,
        )
    def save(self, *, overwrite: bool = True) -> Path:
        """
        Pickle the entire OverlapDataset to:
            OVERLAPS_FOLDER / "overlap_dataset.pkl"
        """
        path = OVERLAPS_FOLDER / "overlap_dataset.pkl"
        print(f"Saving to {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists")

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        return path
    
    @staticmethod
    def load() -> "OverlapDataset | None":
        """
        Load OVERLAPS_FOLDER / 'overlap_dataset.pkl' if it exists.
        Returns None if the file is not present.
        """
        path = OVERLAPS_FOLDER / "overlap_dataset.pkl"
        if not path.exists():
            return None

        with open(path, "rb") as f:
            return pickle.load(f)
        
    def overlap_length_matrix(
        self,
        axon_ids: Sequence[int],
        dend_ids: Sequence[int],
        *,
        min_soma_dist: Optional[float] = None,
        max_soma_dist: Optional[float] = None,
        fill_value: float = 0.0,
        dtype=np.float64,
    ) -> np.ndarray:
        """
        Returns a (n_pre, n_post) numpy array of overlap lengths.

        Entry (i, j) = sum of overlap_lengths for (axon_ids[i], dend_ids[j])
        after filtering overlap groups by post_soma_distances.

        Distance filter per overlap group d:
            - min_soma_dist: keep if d >= min_soma_dist
            - max_soma_dist: keep if d <= max_soma_dist
            - both: min <= d <= max
        """
        axon_ids = [int(a) for a in axon_ids]
        dend_ids = [int(d) for d in dend_ids]
        mat = np.full((len(axon_ids), len(dend_ids)), fill_value, dtype=dtype)
        for i, ax in enumerate(axon_ids):
            for j, de in enumerate(dend_ids):
                rec = self._records.get((ax, de))
                if rec is None:
                    continue

                L = np.asarray(rec.overlap_lengths, dtype=float)
                if L.size == 0:
                    mat[i, j] = 0.0
                    continue

                D = np.asarray(rec.post_soma_distances, dtype=float).min(axis=0)

                keep = np.ones_like(L, dtype=bool)
                if min_soma_dist is not None:
                    keep &= (D >= float(min_soma_dist))
                if max_soma_dist is not None:
                    keep &= (D <= float(max_soma_dist))

                mat[i, j] = L[keep].sum() if np.any(keep) else 0.0
        return mat
    
    def sum_over_posts_per_pre(
        self,
        axon_ids: Sequence[int],
        dend_ids: Sequence[int],
        *,
        min_soma_dist: Optional[float] = None,
        max_soma_dist: Optional[float] = None,
        fill_value: float = 0.0,
        dtype=np.float64,
    ) -> np.ndarray:
        """
        Returns a (n_pre,) vector where entry i is the sum of overlap lengths for
        axon_ids[i] across all dend_ids, with optional soma-distance filtering.
        """
        M = self.overlap_length_matrix(
            axon_ids,
            dend_ids,
            min_soma_dist=min_soma_dist,
            max_soma_dist=max_soma_dist,
            fill_value=fill_value,
            dtype=dtype,
        )
        return M.sum(axis=1)
    
    def sum_over_pres_per_post(
        self,
        axon_ids: Sequence[int],
        dend_ids: Sequence[int],
        *,
        min_soma_dist: Optional[float] = None,
        max_soma_dist: Optional[float] = None,
        fill_value: float = 0.0,
        dtype=np.float64,
    ) -> np.ndarray:
        """
        Returns a (n_post,) vector where entry j is the sum of overlap lengths for
        dend_ids[j] across all axon_ids, with optional soma-distance filtering.
        """
        M = self.overlap_length_matrix(
            axon_ids,
            dend_ids,
            min_soma_dist=min_soma_dist,
            max_soma_dist=max_soma_dist,
            fill_value=fill_value,
            dtype=dtype,
        )
        return M.sum(axis=0)
    
    
        

        
        
        
        
        
        