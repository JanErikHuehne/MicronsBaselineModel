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
    skel : dict or similar
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
    verts = np.asarray(skel['vertices'])
    edges = np.asarray(skel['edges'])

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
