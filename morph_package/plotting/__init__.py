import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_skeleton_colormap(
    skel, 
    values=None, 
    cmap='viridis',
    linewidth=0.6, 
    label='scalar',
    alpha=1.0,
):
    
    verts = skel['vertices']
    edges = skel['edges']
    
    if values is None:
        values = skel['comparment']
    values = np.asanyarray(values)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    
    norm = plt.Normalize(vmin, vmax)
    cmap_func = plt.get_cmap(cmap)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    
    for (u, v) in edges:
        cval = 0.5 * (values[u] + values[v])  # average along edge
        color = cmap_func(norm(cval))
        p1, p2 = verts[u], verts[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, linewidth=linewidth, alpha=alpha)
        
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)")
    ax.set_title(f"Skeleton colored by {label}")
    
    sm = plt.cm.ScalarMappable(cmap=cmap_func, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=label)

    plt.show()
    

def plot_skeleton_synapses_2d(
    skel,
    synapses=None,
    syn_color='crimson',
    skel_color='black',
    linewidth=0.5,
    s=10,
    alpha=0.9,
    figsize=(8, 8),
):
    """
    Plot skeleton edges and synapse positions in 2D (x–y projection).

    Parameters
    ----------
    skel : dict or navis.TreeNeuron
        Contains 'vertices' (N×3 array) and 'edges' (M×2 array).
    synapses : DataFrame or dict, optional
        Should contain 'ctr_pt_position' column or key of shape (K×3).
    syn_color : str
        Color for synapse markers.
    skel_color : str
        Color for skeleton edges.
    linewidth : float
        Line width for skeleton.
    s : float
        Marker size for synapses.
    alpha : float
        Opacity for both skeleton and synapses.
    figsize : tuple
        Figure size.
    """
    if isinstance(skel, dict):
        verts = np.asarray(skel['vertices'])
        edges = np.asarray(skel['edges'])
    else:
        verts = np.vstack(skel.nodes[['x', 'y', 'z']].values)
        edges = np.asarray(skel.edges)
        node_ids = skel.nodes['node_id'].values
        node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
        edges = np.vectorize(node_id_to_index.get)(edges)
        print(len(verts))
        print(edges.max())

    fig, ax = plt.subplots(figsize=figsize)

    # --- plot skeleton edges ---
    for (u, v) in edges:
        p1, p2 = verts[u], verts[v]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=skel_color, linewidth=linewidth, alpha=alpha
        )

    # --- plot synapses ---
    if synapses is not None:
        if 'ctr_pt_position' in synapses:
            syn_xy = np.vstack(synapses['ctr_pt_position'].values)[:, :2]
        else:
            syn_xy = np.asarray(synapses)[:, :2]
        ax.scatter(
            syn_xy[:, 0], syn_xy[:, 1],
            c=syn_color, s=s, alpha=alpha, edgecolor='none'
        )

    ax.set_aspect('equal')
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')
    ax.set_title('Skeleton and synapse locations (x–y projection)')
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_proximities_2d(nt, proximity_groups, ax=None, nt_dend=None):
    """
    Plot a 2D (x–y) projection of a neuron's axon skeleton and highlight
    proximity clusters.

    Parameters
    ----------
    nt : navis.TreeNeuron
        The neuron whose axon skeleton is plotted (usually the presynaptic one).
    proximity_groups : list[list[int]]
        Output of `compute_proximities()` – each inner list contains node IDs
        belonging to one proximity cluster.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure is created.
    nt_dend : Optional plot the dend skeleton which is part of the overlay.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the plot.
    """
    nodes = nt.nodes
    edges = np.asarray(nt.edges)

    # Prepare plotting axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # --- Plot the entire axon skeleton in light gray
    xy = nodes[['x', 'y']].values
    node_index = dict(zip(nodes.node_id.values, range(len(nodes))))
    for e0, e1 in edges:
        if e0 in node_index and e1 in node_index:
            p0, p1 = xy[node_index[e0]], xy[node_index[e1]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='red',  alpha=0.3,lw=0.3, zorder=1)

    if nt_dend: 
        dnodes = nt_dend.nodes
        dedges = np.asarray(nt_dend.edges)
         # --- Plot the entire axon skeleton in light gray
        xy = dnodes[['x', 'y']].values
        node_index = dict(zip(dnodes.node_id.values, range(len(dnodes))))
        for e0, e1 in dedges:
            if e0 in node_index and e1 in node_index:
                p0, p1 = xy[node_index[e0]], xy[node_index[e1]]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='blue', lw=0.3,  alpha=0.3, zorder=1)

    # --- Assign distinct colors to each proximity cluster
   

    for group in proximity_groups:
        group_nodes = nodes[nodes.node_id.isin(group)]
        ax.scatter(group_nodes['x'], group_nodes['y'],
                   s=8, color='black', label=f'Cluster ({len(group)} nodes)', zorder=2)

    # --- Format plot
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title(f"Axon proximity clusters ({len(proximity_groups)} regions)")
    ax.set_aspect('equal', 'box')
    ax.tick_params(direction='out')
    plt.tight_layout()

    plt.show()
