"""
DBSCAN Trainer – an interactive Tkinter app to teach the DBSCAN clustering
algorithm step‑by‑step.

2025‑07‑11 — **Immediate border detection & clear colour scheme**
--------------------------------------------------------------
Stepping behaviour
==================
* **Core** (blue) — labelled immediately when point + ε‑neighbours ≥ minPts.
* **Border** (yellow) — labelled immediately if a point that’s not core
  already touches at least one core.
* **Undecided** (black) — otherwise. These may turn into border or noise at the
  very end once all cores are known.
* **Noise** (vermillion) — undecided points that still touch **no cores** after
  the final pass.
* **Unvisited** (grey) — not processed yet.

Border points keep their yellow colour even after clusters are coloured so
students can always spot them.

Run with:
    python dbscan_trainer.py
"""

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk, messagebox

try:
    from sklearn.datasets import make_blobs, make_moons, make_circles
except ImportError:
    make_blobs = make_moons = make_circles = None

import numpy as np

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Point:
    x: float
    y: float
    index: int
    label: str = "unvisited"      # core, border, noise, undecided, unvisited
    cluster_id: int | None = None
    oval_id: int | None = None
    halo_id: int | None = None

# =============================================================================
# Main application
# =============================================================================

class DBSCANTrainer(tk.Tk):
    WIDTH, HEIGHT = 750, 550

    TYPE_COLOURS = {
        "core": "#0072B2",       # blue
        "border": "#FFD700",     # yellow
        "noise": "#D55E00",      # vermillion
        "undecided": "#000000",  # black
        "unvisited": "#999999",  # grey
    }

    # Palette for clusters — avoids any TYPE_COLOURS hues
    CLUSTER_PALETTE = [
        "#009E73", "#CC79A7", "#56B4E9", "#F0E442", "#882255",
        "#44AA99", "#AA4499", "#117733", "#DDCC77", "#332288",
    ]

    def __init__(self):
        super().__init__()

        self.state('zoomed')  # Start maximized (Windows-friendly)

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        self.WIDTH = int(screen_width * 0.8)   # canvas area (80%)
        self.HEIGHT = screen_height            # full height
        self.SIDEBAR_WIDTH = screen_width - self.WIDTH

        # Tk variables
        self.eps = tk.DoubleVar(value=0.15)
        self.min_pts = tk.IntVar(value=4)
        self.dataset_choice = tk.StringVar(value="Blobs")
        self.quiz_mode = tk.BooleanVar(value=False)
        self.dataset_size = tk.IntVar(value=120)  # default size
        self.quiz_question_count = tk.IntVar(value=5)  # default number of questions

        # State containers
        self.points: list[Point] = []
        self.history: list[tuple[list[str], list[int | None]]] = []
        self.core_graph: dict[int, set[int]] = {}
        self.step_index = 0
        self._demo_circle_id: int | None = None
        self._show_eps_demo_flag = False
        self._hovered_point = None
        self._last_hovered_index = None

        self._second_pass_active = False
        self._second_pass_index = 0
        self._second_pass_list = []
        self._clustering_done = False


        self.bind("<Configure>", self._on_resize)
        self._build_ui()
        self.generate_dataset()

    def _on_resize(self, event):
        if event.widget == self:
            self.WIDTH = int(self.winfo_width() * 0.8)
            self.HEIGHT = self.winfo_height()
            self._draw_points()  # redraw points & clear canvas

            # Re-show ε-circle if it was enabled
            if self._show_eps_demo_flag:
                self._show_eps_demo()



    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        # Allow row and column to expand
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=8)  # canvas takes ~70%
        self.grid_columnconfigure(1, weight=2)  # sidebar ~30%

        # Canvas
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.after(100, self._draw_points)



        # Controls (sidebar)
        ctrl = ttk.Frame(self)
        ctrl.grid(row=0, column=1, sticky="ns", padx=6, pady=6)

        # Dataset selector
        ttk.Label(ctrl, text="Dataset:").pack(anchor="w")
        cmb = ttk.Combobox(ctrl, textvariable=self.dataset_choice, values=self._dataset_list(), state="readonly", width=12)
        cmb.pack(anchor="w", pady=(0, 6))
        cmb.bind("<<ComboboxSelected>>", lambda _: self.generate_dataset())


        # Dataset size selector
        ttk.Label(ctrl, text="Dataset size:").pack(anchor="w", pady=(4, 0))
        ds_row = ttk.Frame(ctrl); ds_row.pack(anchor="w", pady=(0, 6))

        # Spinbox to enter dataset size
        ds_spin = ttk.Spinbox(ds_row, from_=20, to=500, textvariable=self.dataset_size, width=5)
        ds_spin.pack(side="left")

        # The critical line: this button MUST point to generate_dataset
        ttk.Button(ds_row, text="Apply", command=self.generate_dataset).pack(side="left", padx=4)



        # ε slider
        ttk.Label(ctrl, text="ε (eps)").pack(anchor="w", pady=(4, 0))
        mp_row = ttk.Frame(ctrl); mp_row.pack(anchor="w")
        ttk.Scale(mp_row, from_=0.01, to=0.5, length=140, orient="horizontal", variable=self.eps,
                  command=self._on_eps_change).pack(side="left")
        self.eps_lbl = ttk.Label(mp_row, text=str(self.eps.get()))
        self.eps_lbl.pack(side="left", padx=4)

        # minPts slider
        ttk.Label(ctrl, text="minPts").pack(anchor="w", pady=(4, 0))
        mp_row = ttk.Frame(ctrl); mp_row.pack(anchor="w")
        ttk.Scale(mp_row, from_=2, to=10, length=100, orient="horizontal", variable=self.min_pts,
                  command=lambda v: (self.minpts_lbl.config(text=str(int(float(v)))), self.reset(), self._show_eps_demo())).pack(side="left")
        self.minpts_lbl = ttk.Label(mp_row, text=str(self.min_pts.get()))
        self.minpts_lbl.pack(side="left", padx=4)

        # Navigation buttons (no visible frame)
        nav = ttk.LabelFrame(ctrl, text="Navigation")
        nav.pack(anchor="w", pady=8, fill="x")

        btn_width = 16  # Increased to fully show 'Fast-forward'

        ttk.Button(nav, text="◀︎ Previous", command=self.step_prev, width=btn_width).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(nav, text="Next ▶︎", command=lambda: self.step_next(quiz_enabled=True), width=btn_width).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(nav, text="Fast-forward ⏩", command=lambda: self.fast_forward(), width=btn_width).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(nav, text="Auto-play ▶︎", command=lambda: self.auto_play(250), width=btn_width).grid(row=1, column=1, padx=2, pady=2)

        # Quiz mode + Reset
        extra_controls = ttk.Frame(ctrl)
        extra_controls.pack(anchor="w", pady=6, fill="x")

        ttk.Checkbutton(extra_controls, text="Quiz mode", variable=self.quiz_mode).pack(anchor="w", pady=(0, 4))
        # ttk.Button(extra_controls, text="Undo / Reset", command=self.reset).pack(anchor="w")
        # ttk.Label(extra_controls, text="Quiz questions:").pack(anchor="w")
        # ttk.Spinbox(extra_controls, from_=1, to=20, textvariable=self.quiz_question_count, width=5).pack(anchor="w", pady=(0, 6))
        # Add quiz question count entry
        quiz_row = ttk.Frame(extra_controls); quiz_row.pack(anchor="w", pady=(0, 4))
        ttk.Label(quiz_row, text="Number of questions:").pack(side="left")
        ttk.Spinbox(quiz_row, from_=1, to=self.dataset_size.get(), textvariable=self.quiz_question_count, width=4).pack(side="left", padx=4)

        ttk.Button(extra_controls, text="Reset",command=lambda: self.reset(hard=True)).pack(anchor="w")


        # Legend & explanation
        ttk.Label(ctrl, text="Legend:").pack(anchor="w")
        self.legend_frame = ttk.Frame(ctrl); self.legend_frame.pack(anchor="w")

        ttk.Label(ctrl, text="Step explanation:").pack(anchor="w", pady=(8, 0))
        self.explanation = tk.Text(ctrl, width=34, height=18, wrap="word", state="disabled", font=("Arial", 9))
        self.explanation.pack()

        self.canvas.bind("<Leave>", lambda e: setattr(self, "_last_hovered_index", None))



    # ------------------------------------------------ Dataset helpers
    def _dataset_list(self):
        lst = ["Blobs"]
        if make_moons:
            lst += ["Moons", "Circles", "Lines"]
        return lst

    def generate_dataset(self):
        kind = self.dataset_choice.get()
        n = self.dataset_size.get()  # ← use user-defined value
        rng = np.random.default_rng(3)
        
        if kind == "Blobs" or make_moons is None:
            X, _ = make_blobs(n_samples=n, centers=3, cluster_std=0.5, random_state=4)
        elif kind == "Moons":
            X, _ = make_moons(n_samples=n, noise=0.05, random_state=3)
        elif kind == "Circles":
            X, _ = make_circles(n_samples=n, factor=.5, noise=0.05)
        else:  # Lines
            X = np.vstack([
                np.column_stack((rng.uniform(-3, 3, n//2), rng.normal(-1, 0.1, n//2))),
                np.column_stack((rng.uniform(-3, 3, n//2), rng.normal(1, 0.1, n//2)))
            ])

        mins, maxs = X.min(0), X.max(0)
        Xn = (X - mins) / (maxs - mins)
        self.points = [Point(float(x), float(y), i) for i, (x, y) in enumerate(Xn)]
        for p in self.points[:5]:
            print(f"Point {p.index}: ({p.x:.3f}, {p.y:.3f})")

        self.reset(hard=True)
        self._show_eps_demo()

    # ------------------------------------------------ Utility methods
    def _to_canvas(self, x, y):
        pad = 40
        r = 5  # radius of drawn point
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()

        # Adjust range to fully contain the point within bounds
        x_canvas = pad + x * (c_width - 2 * pad - 2 * r) + r
        y_canvas = pad + y * (c_height - 2 * pad - 2 * r) + r

        return x_canvas, y_canvas

    def _on_eps_change(self, v):
        eps = float(v)
        self.eps.set(eps)
        self.eps_lbl.config(text=f"{eps:.3f}")

        if self._hovered_point:
            self._halo(self._hovered_point)

        # Always show demo circle when not clustering
        # if self.step == 0:  # clustering hasn't started
        self._show_eps_demo()




    def _dist(self, a: Point, b: Point):
        return math.hypot(a.x - b.x, a.y - b.y)



    def _epsilon_neigh(self, p: Point):
        eps = self.eps.get()
        tol = 1e-9  # tolerance to handle floating point precision
        return [q for q in self.points if q is not p and self._dist(p, q) <= eps + tol]

    # ------------------------------------------------ Drawing methods
    def _draw_points(self):
        self.canvas.delete("point")
        self.canvas.delete("halo")
        self.canvas.delete("edge")
        self.canvas.delete("coreedge") 
        r = 5
        for p in self.points:
            cx, cy = self._to_canvas(p.x, p.y)
            p.oval_id = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                                fill=self.TYPE_COLOURS[p.label], outline="black", width=1,
                                                tags=(f"pt{p.index}", "point"))
        self.canvas.tag_bind("point", "<Enter>", self._on_hover)
        self.canvas.tag_bind("point", "<Leave>", lambda _: self._hide_tip())


    def _on_hover(self, event):
        item = self.canvas.find_withtag("current")
        if not item:
            return

        idx = int(self.canvas.gettags(item[0])[0][2:])
        

        # Prevent repeated handling for same point
        if hasattr(self, "_last_hovered_index") and self._last_hovered_index == idx:
            return  # Skip processing if we're still on the same point

        self._last_hovered_index = idx  # Update the hovered tracker

        p = self.points[idx]
        self._hovered_point = p

        print(f"Hovering on Point {p.index}")
        print(f"ε value: {self.eps.get()}")
        print(f"Total points: {len(self.points)}")

        for q in self.points:
            if q is not p:
                d = self._dist(p, q)
                if d <= self.eps.get():
                    print(f" → Neighbor: Point {q.index}, dist = {d:.4f}")

        txt = f"Point {p.index}\nLabel: {p.label}\nε-neighbors: {len(self._epsilon_neigh(p))}"
        
        self._show_tip(event.x_root + 14, event.y_root + 12, txt)
        self._halo(p)

        for pt in self.points:
            self.canvas.itemconfig(pt.oval_id, outline="black", width=1)

        for n in self._epsilon_neigh(p):
            self.canvas.itemconfig(n.oval_id, outline="red", width=2)





    def _show_tip(self, x, y, text):
        self._hide_tip()
        tw = tk.Toplevel(self); tw.wm_overrideredirect(True); tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=text, bg="#ffffe0", relief="solid", borderwidth=1, font=("Arial", 9)).pack(ipadx=1)
        self.tooltip = tw
        for pt in self.points:
            self.canvas.itemconfig(pt.oval_id, outline="black", width=1)

    def _hide_tip(self):
        if hasattr(self, "tooltip"):
            self.tooltip.destroy(); del self.tooltip
        for pt in self.points:
            self.canvas.itemconfig(pt.oval_id, outline="black", width=1)
        self.canvas.delete("halo")  # Clear ε-sphere when mouse leaves
        self._hovered_point = None  # Reset the hovered state



    def _show_eps_demo(self):
        self.canvas.delete("eps_demo")
        self.canvas.update_idletasks()

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        pad = 40

        usable_w = canvas_w - 2 * pad
        usable_h = canvas_h - 2 * pad

        cx = canvas_w // 2
        cy = canvas_h // 2
        eps = self.eps.get()

        rx = eps * usable_w
        ry = eps * usable_h

        if rx > 1 and ry > 1:
            self.canvas.create_oval(
                cx - rx, cy - ry,
                cx + rx, cy + ry,
                outline="gray", dash=(4, 2),
                width=2, tags="eps_demo"
            )



    def _halo(self, p: Point):
        # self.canvas.delete("halo")
        self.canvas.update_idletasks()

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        pad = 40

        usable_w = canvas_w - 2 * pad
        usable_h = canvas_h - 2 * pad

        eps = self.eps.get()

        cx, cy = self._to_canvas(p.x, p.y)
        rx = eps * usable_w
        ry = eps * usable_h

        if rx > 1 and ry > 1:
            self.canvas.create_oval(
                cx - rx, cy - ry,
                cx + rx, cy + ry,
                outline="gray", dash=(2, 2), width=1.5, tags="halo"
            )
        
        self.canvas.tag_raise("halo")





    # ------------------------------------------------ Reset & snapshots
    def reset(self, hard=False):
        self.canvas.delete("eps_demo")
        self.canvas.delete("quiz_marker")
        self.step_index = 0
        self._show_eps_demo_flag = True
        self._hovered_point = None
        self._last_hovered_index = None
        self.quiz_mode.set(False)

        self._second_pass_active = False
        self._second_pass_index = 0
        self._second_pass_list = []
        self._clustering_done = False

        for p in self.points:
            p.label, p.cluster_id = "unvisited", None

        if hard:
            self.history.clear()
            self.core_graph.clear()

            # Reset quiz state
            self.quiz_prompt_count = 0
            self.quiz_question_count.set(5)


        self._draw_points()
        self._update_legend()
        self._explain("Dataset reset.")
        self._show_eps_demo()


    def _snapshot(self):
        self.history.append(([p.label for p in self.points], [p.cluster_id for p in self.points]))

    def _restore(self):
        if not self.history:
            return
        labels, cids = self.history.pop()
        for p, l, c in zip(self.points, labels, cids):
            p.label, p.cluster_id = l, c
        self._draw_points()
        self._update_legend()


    # ------------------------------------------------ Stepping control
    def step_prev(self):
        if not self.history:
            return
        self.step_index = max(0, self.step_index - 1)
        self._restore(); self._explain(f"Returned to step {self.step_index}.")

    def step_next(self, quiz_enabled=True):
        self.canvas.delete("eps_demo")
        if not self.points:
            return

        self._show_eps_demo_flag = False

        if not self._second_pass_active:
            if self.step_index < len(self.points):
                p = self.points[self.step_index]

                if quiz_enabled and self.quiz_mode.get():
                    if not hasattr(self, 'quiz_prompt_count'):
                        self.quiz_prompt_count = 0
                    if self.quiz_prompt_count < self.quiz_question_count.get():
                        self._ask_quiz_question(p)
                        self.quiz_prompt_count += 1
                        return

                self._process_step(p)
                return

            # All points processed; prepare second pass
            self._second_pass_active = True
            self._second_pass_list = [p for p in self.points if p.label == "undecided"]
            self._second_pass_index = 0
            self._explain("Starting second pass: handling undecided points.")
            return

        # Now inside second pass
        if self._second_pass_index < len(self._second_pass_list):
            p = self._second_pass_list[self._second_pass_index]
            self._snapshot()

            neighbors = self._epsilon_neigh(p)
            core_neighbors = [n for n in neighbors if n.label == "core"]

            if core_neighbors:
                p.label = "border"
                nearest_core = min(core_neighbors, key=lambda c: self._dist(p, c))
                p.cluster_id = nearest_core.cluster_id
                self.core_graph[nearest_core.index].add(p.index)
                self._explain(f"Point {p.index} has core neighbors → labelled BORDER")
            else:
                p.label = "noise"
                self._explain(f"Point {p.index} has no core neighbors → labelled NOISE")

            self._draw_points()
            self._halo(p)
            self._second_pass_index += 1
            self._update_legend()
            return

        # Final step: apply clustering
        if not self._clustering_done:
            self._finalise_clusters(skip_second_pass=True)
            self._clustering_done = True
            self._explain("Clustering applied after second pass.")



    def _process_step(self, p: Point):
        self._snapshot()
        self._label_point(p)
        self.step_index += 1
        self._draw_points()
        self._halo(p)
        self._update_legend()

    def _mark_quiz_point(self, p: Point):
        """Draws a temporary X over a point during quiz mode."""
        self.canvas.delete("quiz_x")
        r = 7
        cx, cy = self._to_canvas(p.x, p.y)
        self.canvas.create_line(cx - r, cy - r, cx + r, cy + r, fill="red", width=2, tags="quiz_x")
        self.canvas.create_line(cx - r, cy + r, cx + r, cy - r, fill="red", width=2, tags="quiz_x")
        self.canvas.tag_raise("quiz_x")


    def auto_play(self, speed=250):
        if not self.points:
            return

        if self.step_index < len(self.points):
            self.step_next(quiz_enabled=False)  # ⬅️ Important
            self.after(speed, lambda: self.auto_play(speed))
        else:
            self._finalise_clusters()

    def fast_forward(self):
        while self.step_index < len(self.points):
            self.step_next(quiz_enabled=False)  # ⬅️ Important
        self._finalise_clusters()



    def _ask_quiz_question(self, p: Point):
        self._halo(p)
        self._mark_quiz_point(p)  # draw X
        win = tk.Toplevel(self)
        win.title("Quiz")
        win.transient(self)
        win.grab_set()

        tk.Label(win, text=f"Point {p.index} is a ____ point?", font=("Arial", 11)).pack(padx=10, pady=(10, 5))

        choice = tk.StringVar(value="core")
        options = ["core", "border", "noise", "undecided"]

        for opt in options:
            ttk.Radiobutton(win, text=opt.capitalize(), variable=choice, value=opt).pack(anchor="w", padx=20)

        def submit():
            answer = choice.get()

            # Temporarily determine correct label without changing state
            self._snapshot()
            self._label_point(p)
            true_label = p.label
            self._restore()

            if answer == true_label:
                messagebox.showinfo("Result", "✅ Correct!")
            else:
                messagebox.showerror("Result", f"❌ Incorrect.\nCorrect answer: {true_label.capitalize()}")

            self.canvas.delete("quiz_x")  # remove X after quiz
            win.destroy()

            # Now actually proceed to label and step forward
            self._process_step(p)


        self.canvas.delete("quiz_marker")  # before drawing a new one
        ttk.Button(win, text="Submit", command=submit).pack(pady=10)


    # ------------------------------------------------ Labelling logic
    def _label_point(self, p: Point):
        neigh = self._epsilon_neigh(p); cnt = len(neigh); m = int(self.min_pts.get())
        if cnt + 1 >= m:
            p.label = "core"; self.core_graph[p.index] = set(q.index for q in neigh)
            self._explain(f"Point {p.index} labelled CORE")
        elif any(n.label == "core" for n in neigh):
            p.label = "border"
            self._explain(f"Point {p.index} touches a core → BORDER")
        else:
            p.label = "undecided"
            self._explain(f"Point {p.index} marked UNDECIDED (no core neighbour yet)")

    # ------------------------------------------------ Final clustering
    def _finalise_clusters(self, skip_second_pass=False):
        # Build clusters from connected core graph
        visited = set(); cid = 0
        for idx in [p.index for p in self.points if p.label == "core"]:
            if idx in visited:
                continue
            comp, stack = [], [idx]
            while stack:
                i = stack.pop()
                if i in visited:
                    continue
                visited.add(i); comp.append(i)
                stack.extend(self.core_graph.get(i, []))
            for i in comp:
                self.points[i].cluster_id = cid
            cid += 1
        

        # Decide undecided points
        if not skip_second_pass:
            self._explain(f"Starting second pass for undecided points")
            for p in self.points:
                if p.label != "undecided":
                    continue

                neighbors = self._epsilon_neigh(p)
                core_neighbors = [n for n in neighbors if n.label == "core"]

                if core_neighbors:
                    p.label = "border"
                    nearest_core = min(core_neighbors, key=lambda c: self._dist(p, c))
                    p.cluster_id = nearest_core.cluster_id
                    self.core_graph[nearest_core.index].add(p.index)
                else:
                    p.label = "noise"

            


        self._animate_core_edges(); self._apply_cluster_colours(); self._update_legend()
        self._explain("Final labels assigned. Clusters coloured.")
        # Hide eps demo circle when clustering is done
        # self._show_eps_demo_flag = False
        # self.canvas.delete("eps_demo")

    # ------------------------------------------------ Visual helpers
    def _animate_core_edges(self):
        self.canvas.delete("edge");self.canvas.delete("coreedge"); eps = self.eps.get(); drawn = set()
        def draw(i):
            if i >= len(self.points):
                return
            p1 = self.points[i]
            if p1.label == "core":
                for p2 in self.points:
                    if p2.index <= p1.index or p2.label != "core":
                        continue
                    if (p1.index, p2.index) in drawn:
                        continue
                    if self._dist(p1, p2) <= eps:
                        drawn.add((p1.index, p2.index))
                        x1, y1 = self._to_canvas(p1.x, p1.y)
                        x2, y2 = self._to_canvas(p2.x, p2.y)
                        self.canvas.create_line(x1, y1, x2, y2, fill="#BBBBBB", tags="edge")
            self.after(20, lambda: draw(i + 1))
        draw(0)

        for i, neighbors in self.core_graph.items():
            for j in neighbors:
                a, b = self.points[i], self.points[j]
                # Only draw if it's a border point connecting to a core
                if (a.label == "core" and b.label == "border") or (b.label == "core" and a.label == "border"):
                    self.canvas.create_line(self._to_canvas(a.x, a.y),
                                            self._to_canvas(b.x, b.y),
                                            fill="gray", dash=(2, 2), tags="coreedge")


    def _apply_cluster_colours(self):
        for p in self.points:
            if p.label == "core":
                colour = self.CLUSTER_PALETTE[p.cluster_id % len(self.CLUSTER_PALETTE)] if p.cluster_id is not None else "#CCCCCC"
                self.canvas.itemconfig(p.oval_id, fill=colour)
            elif p.label == "border":
                self.canvas.itemconfig(p.oval_id, fill=self.TYPE_COLOURS["border"])
            elif p.label == "noise":
                self.canvas.itemconfig(p.oval_id, fill=self.TYPE_COLOURS["noise"])
            elif p.label in ("undecided", "unvisited"):
                self.canvas.itemconfig(p.oval_id, fill=self.TYPE_COLOURS[p.label])

    # ------------------------------------------------ Legend & explanation
    def _update_legend(self):
        for w in self.legend_frame.winfo_children():
            w.destroy()
        for key in ("core", "border", "noise", "undecided", "unvisited"):
            row = ttk.Frame(self.legend_frame); row.pack(anchor="w")
            tk.Canvas(row, width=12, height=12, bg=self.TYPE_COLOURS[key], highlightthickness=1,
                      highlightbackground="black").pack(side="left")
            ttk.Label(row, text=key.capitalize()).pack(side="left", padx=4)
        cl_ids = sorted({p.cluster_id for p in self.points if p.cluster_id is not None})
        if cl_ids:
            ttk.Label(self.legend_frame, text="Clusters:").pack(anchor="w", pady=(4, 0))
            for cid in cl_ids:
                clr = self.CLUSTER_PALETTE[cid % len(self.CLUSTER_PALETTE)]
                row = ttk.Frame(self.legend_frame); row.pack(anchor="w")
                tk.Canvas(row, width=12, height=12, bg=clr, highlightthickness=1,
                          highlightbackground="black").pack(side="left")
                ttk.Label(row, text=f"C{cid}").pack(side="left", padx=4)

    def _explain(self, text):
        self.explanation.config(state="normal"); self.explanation.delete("1.0", tk.END)
        self.explanation.insert(tk.END, text); self.explanation.config(state="disabled")

# =============================================================================
if __name__ == "__main__":
    DBSCANTrainer().mainloop()

