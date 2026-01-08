import logging
import pickle
from morph_package.microns_api.utils import initalize_overlaps_folder
from morph_package.constants import OVERLAPS_FOLDER
from morph_package.microns_api.skeletons import load_navis_skeletons, extract_dend_axon, clean_resample
from morph_package.proximities import compute_proxmities
from tqdm import tqdm 
logger = logging.getLogger(__name__)


@initalize_overlaps_folder
def workflow_overlap_generation(axon_pt_root_ids, dend_pt_root_ids, overwrite=False):
    """
    Calculate overlaps between all (pre axon) x (post dendrite) pairs.
    Uses cached overlap pickles unless overwrite=True.
    """
    axon_pt_root_ids = list(axon_pt_root_ids)
    dend_pt_root_ids = list(dend_pt_root_ids)

    n_pre = len(axon_pt_root_ids)
    n_post = len(dend_pt_root_ids)
    total_pairs = n_pre * n_post

    logger.info(
        "Starting overlap workflow | n_pre=%d n_post=%d total_pairs=%d overwrite=%s overlaps_dir=%s",
        n_pre, n_post, total_pairs, overwrite, str(OVERLAPS_FOLDER),
    )

    overlaps_list = []
    to_compute = []
    n_cached = 0

    for pre in axon_pt_root_ids:
        for post in dend_pt_root_ids:
            path = OVERLAPS_FOLDER / f"pre{pre}_post{post}.pkl"
            if (not overwrite) and path.exists():
                try:
                    with open(path, "rb") as f:
                        overlaps = pickle.load(f)
                    overlaps_list.append([pre, post, overlaps])
                    n_cached += 1
                    logger.debug("Cache hit | pre=%s post=%s file=%s", pre, post, str(path))
                except Exception:
                    # If cache is corrupt/unreadable, recompute
                    logger.exception("Cache read failed; will recompute | pre=%s post=%s file=%s", pre, post, str(path))
                    to_compute.append((pre, post))
            else:
                to_compute.append((pre, post))

    logger.info(
        "Pair partition | cached=%d to_compute=%d",
        n_cached, len(to_compute),
    )

    if not to_compute:
        logger.info("Done (all cached) | returned=%d", len(overlaps_list))
        return overlaps_list

    # Load only required skeletons
    pre_ids = sorted({pre for pre, _ in to_compute})
    post_ids = sorted({post for _, post in to_compute})

    logger.info("Loading navis skeletons | pre_ids=%d post_ids=%d", len(pre_ids), len(post_ids))
    try:
        pre_nvs = load_navis_skeletons(pre_ids)
        post_nvs = load_navis_skeletons(post_ids)
    except Exception:
        logger.exception("Skeleton loading failed")
        raise

    # Cache processed morphologies to avoid repeated split/resample
    pre_axon_cache = {}
    post_dend_cache = {}

    n_done = 0
    n_fail = 0

    for i, (pre, post) in tqdm(enumerate(to_compute, start=1)):
        path = OVERLAPS_FOLDER / f"pre{pre}_post{post}.pkl"
        logger.debug(
            "Compute start | idx=%d/%d pre=%s post=%s",
            i, len(to_compute), pre, post
        )

        try:
            if pre not in pre_axon_cache:
                pre_nv = pre_nvs[pre]
               
                extracted_neuron = extract_dend_axon(pre_nv)
                assert extracted_neuron[0], 'Axon not found in this neuron, skipping'
                pre_axon_cache[pre] = clean_resample(extracted_neuron[0], "1 microns")
                logger.debug("Prepared pre axon | pre=%s", pre)

            if post not in post_dend_cache:
                post_nv = post_nvs[post]
                extracted_neuron = extract_dend_axon(post_nv)
                assert extracted_neuron[1], 'Dend not found in this neuron, skipping'
                post_dend_cache[post] = clean_resample(extracted_neuron[1], "1 microns")
                logger.debug("Prepared post dendrite | post=%s", post)

            overlaps = compute_proxmities(
                pre_axon_cache[pre],
                post_dend_cache[post],
                axon_pt_root_id=pre,
                dend_pt_root_id=post,
            )

            # Save
            try:
                with open(path, "wb") as f:
                    pickle.dump(overlaps, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug("Saved overlaps | pre=%s post=%s file=%s", pre, post, str(path))
            except Exception:
                # Computation succeeded; saving failed
                logger.exception("Failed to save overlaps | pre=%s post=%s file=%s", pre, post, str(path))
                raise

            overlaps_list.append([pre, post, overlaps])
            n_done += 1

            if n_done % 50 == 0 or n_done == len(to_compute):
                logger.info(
                    "Progress | computed=%d/%d (%.1f%%) cached=%d total_returned=%d",
                    n_done, len(to_compute), 100.0 * n_done / len(to_compute), n_cached, len(overlaps_list),
                )

        except Exception:
            n_fail += 1
            logger.exception("Compute failed | pre=%s post=%s", pre, post)
            # choose behavior: continue or fail-fast
            continue

    logger.info(
        "Done | cached=%d computed=%d failed=%d returned=%d",
        n_cached, n_done, n_fail, len(overlaps_list),
    )
    return overlaps_list

