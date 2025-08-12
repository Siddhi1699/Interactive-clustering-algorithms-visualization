"""
Bisecting K‑Means Educational App
Author: ChatGPT
Date: 2025‑07‑11
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import itertools
import time

# --- Helper functions --------------------------------------------------------


def generate_colors():
    """Infinite generator of distinct colors using matplotlib's tab20 palette, then cycle."""
    cmap = matplotlib.cm.get_cmap("tab20").colors
    while True:
        for c in cmap:
            yield c


def sse(points, centroid):
    """Sum of squared errors for a cluster."""
    if len(points) == 0:
        return 0.0
    return np.sum(((points - centroid) ** 2).sum(axis=1))


def two_means(points, max_iter=10):
    """Run a simple 2‑means and return labels, centroids, history.
    history is a list of (centroids, labels) for each iteration for animation."""
    n_samples = points.shape[0]
    # Initialize two random centroids from data
    indices = np.random.choice(n_samples, 2, replace=False)
    centroids = points[indices]
    history = []
    labels = None
    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(points[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Store state
        history.append((centroids.copy(), labels.copy()))
        # Update
        new_centroids = np.array([points[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                                  for j in range(2)])
        # Convergence
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids, history


# --- Main Application --------------------------------------------------------


class BisectingKMeansApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bisecting K‑Means Educational App")
        self.geometry("1000x700")

        # State
        self.X = None  # dataset (n_samples, 2)
        self.cluster_labels = None  # (n_samples,) current cluster assignments
        self.clusters = {}  # cluster_id -> indices list
        self.centroids = {}  # cluster_id -> centroid array
        self.colors = {}
        self.color_gen = generate_colors()
        self.target_k = tk.IntVar(value=4)

        # UI Setup
        self._build_controls()
        self._build_plot()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # UI builders -------------------------------------------------------------
    def _build_controls(self):
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Generate data
        gen_label = ttk.Label(ctrl_frame, text="Generate dataset")
        gen_label.pack(pady=(0, 5))

        self.n_samples_var = tk.IntVar(value=300)
        self.n_centers_var = tk.IntVar(value=3)
        self.noise_var = tk.DoubleVar(value=0.6)

        for text, var, rng in [
            ("# samples", self.n_samples_var, (50, 1000)),
            ("# true centers", self.n_centers_var, (1, 10)),
            ("cluster std", self.noise_var, (0.1, 3.0)),
        ]:
            row = ttk.Frame(ctrl_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=text, width=12).pack(side=tk.LEFT)
            entry = ttk.Entry(row, textvariable=var, width=6)
            entry.pack(side=tk.LEFT)

        ttk.Button(ctrl_frame, text="Generate", command=self.generate_data).pack(pady=5)

        # Bisect controls
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(ctrl_frame, text="Bisecting K‑Means").pack()
        rowk = ttk.Frame(ctrl_frame); rowk.pack(pady=2)
        ttk.Label(rowk, text="Target k").pack(side=tk.LEFT)
        ttk.Entry(rowk, textvariable=self.target_k, width=4).pack(side=tk.LEFT)

        ttk.Button(ctrl_frame, text="Bisect Step", command=self.bisect_step).pack(pady=5)

        ttk.Button(ctrl_frame, text="Reset", command=self.reset).pack(pady=5)

        # SSE listbox
        ttk.Separator(ctrl_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(ctrl_frame, text="Cluster SSE").pack()
        self.sse_list = tk.Listbox(ctrl_frame, width=25, height=10)
        self.sse_list.pack()

    def _build_plot(self):
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Dataset")
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Data --------------------------------------------------------------------
    def generate_data(self):
        n_samples = self.n_samples_var.get()
        n_centers = self.n_centers_var.get()
        cluster_std = self.noise_var.get()
        self.X, _ = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            cluster_std=cluster_std,
            random_state=None,
        )
        self.reset_clusters()
        self.draw()

    def reset_clusters(self):
        if self.X is None:
            return
        self.cluster_labels = np.zeros(len(self.X), dtype=int)
        self.clusters = {0: np.arange(len(self.X))}
        self.centroids = {0: self.X.mean(axis=0)}
        self.colors = {0: next(self.color_gen)}
        self.update_sse_list()

    def reset(self):
        self.reset_clusters()
        self.draw()

    # Bisect ------------------------------------------------------------------
    def bisect_step(self):
        if self.X is None:
            messagebox.showinfo("Info", "Generate data first.")
            return

        if len(self.clusters) >= self.target_k.get():
            messagebox.showinfo("Info", f"Already reached k={self.target_k.get()}")
            return

        # Choose cluster with max SSE
        sse_values = {cid: sse(self.X[idxs], self.centroids[cid]) for cid, idxs in self.clusters.items()}
        cid_to_split = max(sse_values, key=sse_values.get)

        # Run 2‑means on that cluster
        idxs = self.clusters[cid_to_split]
        points = self.X[idxs]
        labels2, centroids2, history = two_means(points)

        # Animate 2‑means iterations
        for cands, labs in history:
            self._draw_partial_split(cid_to_split, idxs, labs, cands)
            self.update()
            time.sleep(0.6)

        # Remove old cluster
        del self.clusters[cid_to_split]
        del self.centroids[cid_to_split]
        del self.colors[cid_to_split]

        # Add new clusters
        new_cids = [max(self.clusters.keys(), default=-1) + i + 1 for i in range(2)]
        for j, new_cid in enumerate(new_cids):
            new_idxs = idxs[labels2 == j]
            if len(new_idxs) == 0:
                continue
            self.clusters[new_cid] = new_idxs
            self.centroids[new_cid] = centroids2[j]
            self.colors[new_cid] = next(self.color_gen)
            self.cluster_labels[new_idxs] = new_cid

        self.update_sse_list()
        self.draw()

    # Drawing -----------------------------------------------------------------
    def _draw_partial_split(self, cid_to_split, idxs, labs, cands):
        """Visualize intermediate 2‑means iteration."""
        self.ax.clear()
        self.ax.set_title("Bisecting cluster {}".format(cid_to_split))
        # Draw remaining clusters
        for cid, pidx in self.clusters.items():
            if cid == cid_to_split:
                continue
            pts = self.X[pidx]
            self.ax.scatter(*pts.T, s=20, c=[self.colors[cid]], alpha=0.6)
            self.ax.scatter(*self.centroids[cid], marker="*", s=200,
                            c=[self.colors[cid]], edgecolors="k")
        # Draw split cluster with two temporary colors
        split_colors = [self.colors[cid_to_split], next(self.color_gen)]
        for j in range(2):
            pts = self.X[idxs[labs == j]]
            self.ax.scatter(*pts.T, s=20, c=[split_colors[j]], alpha=0.9)
            self.ax.scatter(*cands[j], marker="*", s=250,
                            c=[split_colors[j]], edgecolors="k", linewidths=1.5)

        self.canvas.draw()

    def draw(self):
        if self.X is None:
            return
        self.ax.clear()
        self.ax.set_title("Bisecting k‑means (k={})".format(len(self.clusters)))
        for cid, idxs in self.clusters.items():
            pts = self.X[idxs]
            self.ax.scatter(*pts.T, s=20, c=[self.colors[cid]], alpha=0.7)
            self.ax.scatter(*self.centroids[cid], marker="*", s=250,
                            c=[self.colors[cid]], edgecolors="k", linewidths=1.5,
                            label=f"Cluster {cid}")
        self.ax.legend(loc="best", fontsize="small")
        self.canvas.draw()

    # SSE listbox -------------------------------------------------------------
    def update_sse_list(self):
        self.sse_list.delete(0, tk.END)
        values = []
        for cid, idxs in self.clusters.items():
            err = sse(self.X[idxs], self.centroids[cid])
            values.append((cid, err))
        # Sort descending
        values.sort(key=lambda x: x[1], reverse=True)
        for cid, err in values:
            text = f"Cluster {cid}: SSE = {err:.2f}"
            self.sse_list.insert(tk.END, text)
        # Highlight first (largest) cluster
        if values:
            self.sse_list.itemconfig(0, bg="yellow")

    # ------------------------------------------------------------------------
    def on_close(self):
        self.destroy()


if __name__ == "__main__":
    app = BisectingKMeansApp()
    app.mainloop()
