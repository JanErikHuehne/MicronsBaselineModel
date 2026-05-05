import numpy as np 
import navis 
import pickle
import copy
import networkx as nx 
import pandas as pd 
from pathlib import Path 
from tqdm import tqdm 
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any, Sequence

from morph_package.constants import OVERLAPS_FOLDER
from morph_package.microns_api.synapses import get_synapases
from morph_package.microns_api.skeletons import map_synapses, load_navis_skeletons, extract_dend_axon, clean_resample, load_single_skeleton_from_disk_safe


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
       
    def get_axon_ids(self):
        return [c.axon_nodes for c in self.overlaps]
    
    def get_dend_ids(self):
        return [c.dend_nodes for c in self.overlaps]
    
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
        try:
            soma_node = self.collection.dend_tree.root
            distances = []
            for d_node in self.dend_nodes:
                d_geo = navis.dist_between(self.collection.dend_tree, soma_node, d_node) 
                distances.append(d_geo)
        
            self.post_soma_distance = min(distances)
        except Exception as e:
            print('Error in method get_post_soma_distance:', e)
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
    Compute proximities between axonal segments of nv_pre_tree and dendritic segments of nv_post_tree.
    Returns an OverlapCollection object containing the proximities and associated synapses.
    Parameters
    ----------
    nv_pre_tree : navis Tree
        Presynaptic neuron morphology.
    nv_post_tree : navis Tree
        Postsynaptic neuron morphology.
    threshold_um : float    
        Distance threshold in microns for considering axonal and dendritic segments as proximal.
    Returns
    -------
    OverlapCollection
        An object containing the identified proximities and associated synapses.
    
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
    overlap_dend_nodes : List[List[float]]
    overlap_axon_nodes : List[List[float]]
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
        self.tree_metrics = {}
        self.sst_clusters = None
        self.bc_clusters = None
        self.distance = 5
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
        
    
    def save(self, *, path=None, overwrite: bool = True) -> Path:
        """
        Pickle the entire OverlapDataset to:
            OVERLAPS_FOLDER / "overlap_dataset.pkl"
        """
        if not path:
            path = OVERLAPS_FOLDER / "overlap_dataset.pkl"
        else:
            path =Path(path) / "overlap_dataset.pkl"
        print(f"Saving to {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists")

        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        return path
    
    
    @staticmethod
    def load(path) -> "OverlapDataset | None":
        """
        Load OVERLAPS_FOLDER / 'overlap_dataset.pkl' if it exists.
        Returns None if the file is not present.
        """
        if not path:
            path = OVERLAPS_FOLDER / "overlap_dataset.pkl"
        else:
            path = Path(path)
        if not path.exists():
            return None

        with open(path, "rb") as f:
            return pickle.load(f)
    
    
    @staticmethod
    def fuse(d1, d2, join='d1'):
        """
        Fuses two OverlapDatasets by joining their records on (axon_id, dend_id).
        - join='d1': if a record is in d1 and d2, use the one in d1
        - join='d2': if a record is in d1 and d2, use the one in d2
        """
        assert d1.distance == d2.distance, "Cannot fuse datasets with different distance thresholds"
        if join not in ('d1', 'd2'):
            raise ValueError("join must be 'd1' or 'd2'")
        new_dataset = cls()
        new_dataset.distance = d1.distance  # or d2.distance, they are the same
        keys = set(d1._records.keys()) | set(d2._records.keys())
        for key in keys:
            if key in d1._records and key in d2._records:
                rec = d1._records[key] if join == 'd1' else d2._records[key]
            elif key in d1._records:
                rec = d1._records[key]
            else:
                rec = d2._records[key]
            new_dataset._records[key] = rec
        
        return new_dataset
    
    
    def get_subset_dataset(self, neuron_ids: Sequence[int]):
        """Returns a new OverlapDataset with records only for the specified neuron IDs."""
        neuron_ids = set(int(n) for n in neuron_ids)
        self_copy = copy.deepcopy(self)
        # we need to rest records of the new dataset to only include those where either the axon_id or dend_id is in neuron_ids
        self_copy._records = {}
        i = 0
        for key, rec in tqdm(self._records.items()):
        
            if rec.axon_id in neuron_ids and rec.dend_id in neuron_ids:
                self_copy._records[key] = rec
                i += 1
        print(f"Filtered dataset to {i} records involving neurons {neuron_ids}")
        return self_copy
    
    
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

                lists = rec.post_soma_distances
                max_len = max(len(x) for x in lists)

                D = np.min(
                    np.vstack([
                        np.pad(x, (0, max_len - len(x)), constant_values=np.inf)
                        for x in lists
                    ]),
                    axis=1
                )
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
    
    
    def synapse_matrix(self, axon_ids, dend_ids):
        axon_ids = [int(a) for a in axon_ids]
        dend_ids = [int(d) for d in dend_ids]
        mat = np.full((len(axon_ids), len(dend_ids)), 0, dtype=int)
        for i, ax in enumerate(axon_ids):
            for j, de in enumerate(dend_ids):
                rec = self._records.get((ax, de))
                if rec is None:
                    continue
                else:
                    mat[i, j] = rec.n_synapses_total
        
        return mat 

    
    def jaccard_index_matrix(self, axon_ids, dend_ids, **kwargs):
        axon_ids = [int(a) for a in axon_ids]
        dend_ids = [int(d) for d in dend_ids]
        mat = self.overlap_length_matrix(axon_ids, dend_ids, **kwargs)
        
        # jaccard = o / (a_axon + a_dend - o)
        for i, ax in enumerate(axon_ids):
            axon_length, _ = self.get_axon_dend_length(ax)
            for j, de in enumerate(dend_ids):
                _, dend_length = self.get_axon_dend_length(de)
                o = mat[i, j]
                denom = axon_length + dend_length - o
                if denom > 0:
                    mat[i, j] = o / denom
                else:
                    mat[i, j] = 0.0
        return mat 
    
    
    def overlap_candidate_matrix(self, axon_ids, dend_ids):
        axon_ids = [int(a) for a in axon_ids]
        dend_ids = [int(d) for d in dend_ids]
        mat = np.full((len(axon_ids), len(dend_ids)), False, dtype=int)
        for i, ax in enumerate(axon_ids):
            for j, de in enumerate(dend_ids):
                rec = self._records.get((ax, de))
                if rec is None:
                    continue
                else:
                    mat[i, j] = rec.n_overlap_groups 
        return mat
    
    
    def get_axon_dend_length(self, pt_root_id):
        if not getattr(self, 'tree_metrics', None):
            self.tree_metrics = {}
        if pt_root_id in self.tree_metrics:
            return (self.tree_metrics[pt_root_id]['axon_length'], self.tree_metrics[pt_root_id]['dend_length'])
        else:
            n_neuron = list(load_navis_skeletons([pt_root_id]).values())[0]
            extracted_neuron = extract_dend_axon(n_neuron)
            axon_length = navis.morpho.cable_length(extracted_neuron[0])
            dend_length = navis.morpho.cable_length(extracted_neuron[1])
            self.tree_metrics[pt_root_id] = {'axon_length': axon_length, 'dend_length': dend_length}
    
    
    def filter_records(self, axon_soma_dist=5, force_recalc=False, n_workers=8):
        """
        Returns a new OverlapDataset with records re-filtered by axon-dendrite distance.

        Records that already satisfy the skip condition (axon_soma_dist < self.distance
        and total_overlap_length == 0.0) are carried over unchanged unless `force_recalc`
        is set. All remaining records are re-filtered in parallel using a ThreadPoolExecutor.

        Parameters
        ----------
        axon_soma_dist : float, optional
            Maximum surface-to-surface distance (microns) used to re-filter overlaps.
            Default is 5.
        force_recalc : bool, optional
            If True, bypasses the skip condition and recalculates every record regardless
            of its current state. Default is False.
        n_workers : int, optional
            Number of parallel threads to use for recalculation. Default is 8.

        Returns
        -------
        OverlapDataset
            Deep copy of self with qualifying records re-filtered.
        """

        self_copy = copy.deepcopy(self)

        # --- Step 1: Partition records into those that need recalculation and those that don't ---
        to_recalc = {}
        to_skip   = {}

        for key, rec in self._records.items():
            if not force_recalc and axon_soma_dist < self.distance and rec.total_overlap_length == 0.0:
                to_skip[key] = rec       # Condition not met — carry over unchanged
            else:
                to_recalc[key] = rec     # Needs re-filtering

        # Pre-warm: load everything into memory on the main thread before parallelising
        print("Pre-warming skeleton cache...")
        for pre,post in tqdm(to_recalc.keys()):
            load_single_skeleton_from_disk_safe(pre)
            load_single_skeleton_from_disk_safe(post)
            

        if force_recalc:
            print(f"Force recalc enabled — all {len(to_recalc)} records queued.")
        else:
            print(f"{len(to_recalc)} records queued for recalculation, {len(to_skip)} skipped.")

        # --- Step 2: Recalculate qualifying records in parallel ---
        def _recalc(item):
            """Worker: re-filters a single record and returns (key, updated_record)."""
            key, rec = item
            updated = self._filter_overlap_record(rec, axon_dend_dist=axon_soma_dist)
            return key, updated

        results = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_recalc, item): item[0] for item in to_recalc.items()}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering records"):
                try:
                    key, updated_record = future.result()
                    results[key] = updated_record
                except Exception as e:
                    key = futures[future]
                    print(f"Record {key} failed: {e}")
                    results[key] = self._records[key]   # Fall back to original on failure

        # --- Step 3: Merge skipped and recalculated records back into the copy ---
        self_copy._records.update(to_skip)
        self_copy._records.update(results)

        print(f"Filtered {len(to_recalc)} records with axon-dendrite distance <= {axon_soma_dist} um")
        return self_copy
    
    
    def _filter_overlap_record(self, overlap_record: OverlapRecord, axon_dend_dist=1) -> OverlapRecord:
        """
        Computes spatial overlap geometry for a pre/post neuron pair and returns
        an enriched copy of the input OverlapRecord.

        Loads the axon and dendrite skeletons for the neuron IDs stored in
        `overlap_record`, resamples them to 1 micron resolution, identifies
        overlapping node groups within `axon_dend_dist`, and annotates the copy
        with overlap node IDs, cable lengths, soma distances, and group count.
        The original record is never mutated.

        Parameters
        ----------
        overlap_record : OverlapRecord
            Record containing at minimum `axon_id` and `dend_id` identifying the
            pre- and post-synaptic neurons to analyse.
        axon_dend_dist : float, optional
            Maximum surface-to-surface distance (in microns) for two nodes to be
            considered overlapping. Default is 1.

        Returns
        -------
        OverlapRecord
            A deep copy of `overlap_record` with the following fields updated:

            - ``overlap_axon_nodes``    – list of axon node ID groups
            - ``overlap_dend_nodes``    – list of dendrite node ID sets per group
            - ``overlap_lengths``       – cable length of each axon overlap group
            - ``total_overlap_length``  – sum of all group lengths (microns)
            - ``post_soma_distances``   – soma distance for each dendrite group
            - ``n_overlap_groups``      – number of distinct overlap groups found

        Raises
        ------
        Exception
            If skeleton loading fails for either neuron ID.
        AssertionError
            If no axon compartment is found in the pre-synaptic neuron.

        Notes
        -----
        - Skeletons are resampled to 1 micron node spacing via `clean_resample`
        before any distance computation.
        - The original `overlap_record` is left intact; all modifications are
        applied to a deep copy.
        """

        # Load raw skeletons for both the pre- (axon) and post- (dendrite) neurons
        try:
            pre_nvs  = list(load_navis_skeletons([overlap_record.axon_id]).values())[0]
            post_nvs = list(load_navis_skeletons([overlap_record.dend_id]).values())[0]
        except Exception:
            raise Exception("Skeleton loading failed")

        # Isolate and resample the axon compartment of the pre-synaptic neuron
        extracted_neuron = extract_dend_axon(pre_nvs)
        assert extracted_neuron[0], 'Axon not found in this neuron, skipping'
        axon = clean_resample(extracted_neuron[0], "1 microns")

        # Isolate and resample the dendrite compartment of the post-synaptic neuron
        extracted_neuron = extract_dend_axon(post_nvs)
        dend = clean_resample(extracted_neuron[1], "1 microns")

        # Deep-copy the record so the original remains intact for comparison if needed
        overlap_copy = copy.deepcopy(overlap_record)

        # Identify spatially overlapping node groups between axon and dendrite
        axon_groups, dend_groups = self._calc_proximities(axon, dend, axon_dend_dist=axon_dend_dist)
        overlap_copy.overlap_axon_nodes = axon_groups
        overlap_copy.overlap_dend_nodes = dend_groups

        # Compute cable length for each axon overlap group and the cumulative total
        overlap_lengths, total_overlap_length = self._get_overlap_length(axon, axon_groups)
        overlap_copy.overlap_lengths       = overlap_lengths
        overlap_copy.total_overlap_length  = total_overlap_length

        # Compute the path distance from each dendrite overlap group to the post-synaptic soma
        overlap_copy.post_soma_distances = self._calculate_soma_distance(overlap_copy.overlap_dend_nodes, dend)

        # Record how many distinct overlap groups were found
        overlap_copy.n_overlap_groups = len(overlap_lengths)

        return overlap_copy
    
    
    def _calculate_soma_distance(self, overlap_groups, nv_tree):
        try:
            all_distances = []
            for node_ids in overlap_groups:
                if len(node_ids) == 0:
                    continue
                soma_node = nv_tree.root
                distances = []
                for d_node in node_ids:
                    d_geo = navis.dist_between(nv_tree, soma_node, d_node) 
                    distances.append(d_geo)
                all_distances.extend(distances)

        except Exception as e:
            print('Error in method get_post_soma_distance:', e)
        return all_distances 
    
    
    def _get_overlap_length(self, nv_tree, ids):
        """
        Computes the cable length of each overlap group within a neuron skeleton.

        For each group of node IDs, sums the Euclidean edge lengths of all edges
        whose both endpoints belong to that group. Returns per-group lengths and
        the total cumulative length across all groups.

        Parameters
        ----------
        nv_tree : navis.TreeNeuron
            The skeleton to measure. Must have a 'nodes' DataFrame with columns
            'node_id', 'x', 'y', 'z', and an 'edges' iterable of (parent, child) pairs.
        ids : list of list of int
            Each inner list is an overlap group containing node IDs from `nv_tree`
            (e.g. `finished_groups` returned by `calc_proximities`).

        Returns
        -------
        group_lengths : list of float
            Cable length of each overlap group, in the same spatial units as the
            skeleton coordinates. Parallel to `ids`.
        total_length : float
            Sum of all values in `group_lengths`.

        Notes
        -----
        - Only edges where **both** endpoints are in the group are counted, so
        isolated nodes (no in-group neighbours) contribute 0 length.
        - Edge length is Euclidean: ``||pos_parent - pos_child||``.

        Example
        -------
        >>> group_lengths, total_length = obj.get_overlap_length(nv_tree=axon, ids=finished_groups)
        >>> print(group_lengths)   # [123.4, 56.7, ...]
        >>> print(total_length)    # 180.1
        """

        # Build a node_id → {x, y, z} lookup for fast coordinate access during edge iteration
        node_coords = nv_tree.nodes.set_index('node_id')[['x', 'y', 'z']].astype(float)
        node_coords = node_coords.to_dict('index')

        group_lengths = []

        for group in ids:

            group_set = set(group)  # O(1) membership checks
            total_length = 0.0

            # Sum the Euclidean length of every edge contained within this group
            for parent, child in nv_tree.edges:
                if parent in group_set and child in group_set:
                    p_pos = np.array(list(node_coords[parent].values()))
                    c_pos = np.array(list(node_coords[child].values()))
                    total_length += np.linalg.norm(p_pos - c_pos)

            group_lengths.append(total_length)

        # Return per-group lengths and their cumulative sum
        return group_lengths, np.sum(group_lengths)

    
    def _calc_proximities(self, nv_pre_tree, nv_post_tree, axon_dend_dist=1):
        """
            Identifies spatial overlap groups between an axon and a dendrite skeleton.

            For each axon node, computes surface-to-surface distances to all dendrite
            nodes (accounting for both radii). Axon nodes within `axon_dend_dist` of
            any dendrite node are flagged as overlapping. Overlapping axon nodes that
            are also topologically connected in the axon graph are merged into a single
            group. Each group is returned alongside the union of dendrite node IDs it
            overlaps with.

            Parameters
            ----------
            nv_pre_tree : navis.TreeNeuron
                The pre-synaptic (axon) skeleton. Must have node attributes:
                'node_id', 'x', 'y', 'z', 'radius'.
            nv_post_tree : navis.TreeNeuron
                The post-synaptic (dendrite) skeleton. Must have node attributes:
                'node_id', 'x', 'y', 'z', 'radius'.
            axon_dend_dist : float, optional
                Maximum allowed surface-to-surface distance (in the tree's spatial
                units) for two nodes to be considered overlapping. Negative values
                indicate sphere overlap. Default is 1.

            Returns
            -------
            finished_groups : list of list of int
                Each inner list contains the axon node IDs belonging to one
                topologically connected overlap group.
            dendrite_groups : list of set of int
                Parallel to `finished_groups`. Each set contains the dendrite node
                IDs that fall within `axon_dend_dist` of any axon node in the
                corresponding group.

            Notes
            -----
            - Distance is computed as surface-to-surface: 
            ``dist = ||center_ax - center_dend|| - (radius_ax + radius_dend)``
            - Axon node grouping respects the axon's graph topology via
            ``nx.connected_components`` on the undirected subgraph of overlapping nodes.
            - `finished_groups[i]` and `dendrite_groups[i]` always correspond to the
            same overlap group.

            Example
            -------
            >>> finished_groups, dendrite_groups = obj._calc_proximities(
            ...     nv_pre_tree=axon, nv_post_tree=dendrite, axon_dend_dist=0.5
            ... )
            >>> print(finished_groups[0])   # axon node IDs in first group
            >>> print(dendrite_groups[0])   # dendrite node IDs overlapping with it
        """
        
     
        
        # Extract dendrite node radii, IDs, and positions from the post-synaptic tree
        dend_r = nv_post_tree.nodes['radius'].values.astype(float)
        dend_nodes = nv_post_tree.nodes['node_id'].values
        dend_pos = nv_post_tree.nodes[['x', 'y', 'z']].values.astype(float)
        
        
        
        overlap_list = [] # Axon nodes that are within axon_dend_dist of at least one dendrite node
        ax_dend_dict = {} # Maps each overlapping axon node_id -> set of nearby dendrite ndoe_ids 
        
        # --- Step 1: Find axon nodes that are spatially close to any dendrite node --- 
        for _,ax_node in nv_pre_tree._get_nodes().iterrows():
            
            ax_node_pos = ax_node[['x', 'y', 'z']].values[np.newaxis, ...].astype(float)
            ax_node_id = ax_node['node_id']
            ax_r = float(ax_node['radius'])
            
            
            # Compute center-to-center distances from this axon node to all dendrite nodes 
            center_dists = np.linalg.norm(dend_pos - ax_node_pos, axis=1).astype(float)
            
            # Convert to surface-to-surface distances by substracting both radii 
            surface_dists = center_dists - (ax_r + dend_r)
 
            min_d = surface_dists.min() # closests dendrite surface distance for this axon node 
            
            # Keep this axon node if any dendrite node is within axon_dend_dist
            close_mask = surface_dists <= axon_dend_dist
            if np.any(close_mask):
                overlap_list.append({'ax_node_id': ax_node_id, 'min_dist': min_d})
                # Store which dendrite nodes are close to this axon node for later grouping
                ax_dend_dict[ax_node_id] = set(int(d) for d in dend_nodes[close_mask])
        
        # --- Step 2: Group connected overlapping axon nodes using the axon's graph topology ---
        ax_graph = nv_pre_tree.graph
        
        # Extract just the axon node IDs that had overlaps for subgraphing
        groups = [l['ax_node_id'] for l in overlap_list]
        
        # Build a subgraph from only the overlapping axon nodes, the find connected components 
        # (i.e. spatially overlapping axon nodes that are also topologically connected)
        subgraph = ax_graph.subgraph(groups) 
        finished_groups = [list(int(cc) for cc in c) 
                           for c in nx.connected_components(subgraph.to_undirected())]
        
        # --- Step 3: For each axon group, collected the union of all nearby dendrite node IDs ---
        dendrite_groups = []
        for group in finished_groups:
            dend_set = set()
            for ax_node in group:
                dend_set.update(ax_dend_dict[ax_node]) # Union of all dendrite nodes across the whole axon group  
            dendrite_groups.append(list(dend_set))
        
        # Returns 
        #  finished_groups[i] --> list of axon node IDs in overlap group i 
        #  dendrite_groups[i] --> list of dendrite node IDs in overlap group i 
        return (finished_groups, dendrite_groups)


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
            axon_nodes = collection.get_axon_ids()
            dend_nodes = collection.get_dend_ids()
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
                overlap_axon_nodes=axon_nodes,
                overlap_dend_nodes=dend_nodes,
                overlap_lengths=overlap_lengths,
                post_soma_distances=post_soma_distances,
                total_overlap_length=total_overlap_length,
                n_overlap_groups=n_overlap_groups,
                n_synapses_total=n_synapses_total,
                synapses=synapses_df,
                cache_path=cache_path,
            )
  
        
    
def compute_rate(real: np.ndarray, overlap: np.ndarray) -> float:
    denom = overlap.sum()
    return float(real.sum() / denom) if denom > 0 else 0.0
        
        
        
def poisson_null_matrices(candidates: np.ndarray, rate: float, n: int = 100, rng=None):
    """
    candidates: exposure matrix (>=0), same shape as desired output
    rate: global conversion rate (synapses per candidate unit)
    n: number of null samples
    returns: (n, *candidates.shape) integer matrices
    """
    if rng is None:
        rng = np.random.default_rng()

    cand = np.asarray(candidates, dtype=float)
    if np.any(cand < 0):
        raise ValueError("candidates must be nonnegative")

    lam = rate * cand  # elementwise Poisson mean
    # broadcasting: lam has shape (*shape), output will be (n, *shape)
    return rng.poisson(lam=lam, size=(n,) + lam.shape).astype(np.int32)


