from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

class SkeletonPlotter:
    def __init__(
        self,
        projection="2d",
        figsize=(8, 8),
        ax=None,
        units="nm",
    ):
        self.projection = projection
        self.units = units

        if ax is None:
            if projection == "3d":
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        self.fig = fig
        self.ax = ax

    # ---------- geometry helpers ----------

    @staticmethod
    def _extract_vertices_edges(neuron):
        if isinstance(neuron, dict):
            verts = np.asarray(neuron["vertices"])
            edges = np.asarray(neuron["edges"])
            return verts, edges

        # navis.TreeNeuron
        nodes = neuron.nodes
        verts = nodes[["x", "y", "z"]].values
        node_index = dict(zip(nodes.node_id.values, range(len(nodes))))
        edges = np.vectorize(node_index.get)(np.asarray(neuron.edges))
        return verts, edges

    def _plot_edge(self, p0, p1, **kwargs):
        if self.projection == "3d":
            self.ax.plot(
                [p0[0], p1[0]],
                [p0[2], p1[2]],
                [p0[1], p1[1]],
                **kwargs,
            )
        else:
            self.ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                **kwargs,
            )



    def add_skeleton(
        self,
        neuron,
        values=None,
        compartments=True,
        cmap="viridis",
        color="black",
        linewidth=0.5,
        alpha=1.0,
        label=None,
        zorder=1,
        add_colorbar=False,
        soma_size=30,
    ):
        verts, edges = self._extract_vertices_edges(neuron)

        # ---- categorical compartments (default behavior if provided) ----
        if compartments :
            compartments = neuron.nodes["compartment"].values
            # default compartment styling
            COMP_COLORS = {
                1: "black",  # soma
                2: "red",    # axon
                3: "blue",   # dendrite
            }

            for u, v in edges:
                p0, p1 = verts[u], verts[v]

                cu, cv = compartments[u], compartments[v]

                # mixed edge → prioritize non-soma color
                if cu == cv:
                    c = COMP_COLORS.get(cu, "gray")
                else:
                    c = COMP_COLORS.get(cv if cv != 1 else cu, "gray")

                self._plot_edge(
                    p0, p1,
                    color=c,
                    lw=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )

            # draw soma nodes as large black circles
            soma_idx = np.where(compartments == 1)[0]
            if len(soma_idx) > 0:
                soma_pts = verts[soma_idx]
                if self.projection == "2d":
                    self.ax.scatter(
                        soma_pts[:, 0],
                        soma_pts[:, 1],
                        s=soma_size,
                        c="black",
                        alpha=alpha,
                        zorder=zorder + 1,
                    )
                else:
                    self.ax.scatter(
                        soma_pts[:, 0],
                        soma_pts[:, 2],
                        soma_pts[:, 1],
                        s=soma_size,
                        c="black",
                        alpha=alpha,
                        zorder=zorder + 1,
                    )

            return self

        # ---- continuous scalar coloring (original behavior) ----
        if values is not None:
            values = np.asarray(values)
            norm = Normalize(np.nanmin(values), np.nanmax(values))
            cmap_func = plt.get_cmap(cmap)

        for u, v in edges:
            p0, p1 = verts[u], verts[v]

            if values is None:
                c = color
            else:
                cval = 0.5 * (values[u] + values[v])
                c = cmap_func(norm(cval))

            self._plot_edge(
                p0,
                p1,
                color=c,
                lw=linewidth,
                alpha=alpha,
                zorder=zorder,
            )

        if values is not None and add_colorbar:
            sm = ScalarMappable(norm=norm, cmap=cmap_func)
            sm.set_array([])
            self.fig.colorbar(sm, ax=self.ax, label=label)

        return self

    
    def add_synapses(
        self,
        synapses,
        color="crimson",
        s=10,
        alpha=0.9,
        zorder=3,
    ):
        if synapses is None:
            return self

        if isinstance(synapses, dict) or "ctr_pt_position" in synapses:
            pts = np.vstack(synapses["ctr_pt_position"].values)
        else:
            pts = np.asarray(synapses)

        xy = pts[:, :2]

        self.ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=color,
            s=s,
            alpha=alpha,
            edgecolor="none",
            zorder=zorder,
        )
        return self

    def add_proximity_groups(
        self,
        neuron,
        groups,
        color="black",
        s=8,
        alpha=1.0,
        zorder=4,
    ):
        nodes = neuron.nodes

        for group in groups:
            g = nodes[nodes.node_id.isin(group)]
            if self.projection == "3d":
                self.ax.scatter(
                    g["x"],
                    g["z"],
                    g["y"],
                    s=s,
                    color=color,
                    alpha=alpha,
                    zorder=zorder,
                )
            else:
                self.ax.scatter(
                    g["x"],
                    g["z"],
                    s=s,
                    color=color,
                    alpha=alpha,
                    zorder=zorder,
                )
           
        return self


    def format(
        self,
        title=None,
        equal_aspect=True,
    ):
        self.ax.set_xlabel(f"x ({self.units})")
        self.ax.set_ylabel(f"y ({self.units})")
        
       
        if self.projection == "3d":
            self.ax.set_zlabel(f"z ({self.units})")
            self.ax.invert_zaxis()
        else:
            if equal_aspect:
                self.ax.set_aspect("equal", "box")
            self.ax.invert_yaxis()

        if title:
            self.ax.set_title(title)

        return self

    def show(self):
        plt.tight_layout()
        plt.show()
