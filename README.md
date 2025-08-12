# Interactive-clustering-algorithms-visualization
Interactive educational tools for visualizing clustering algorithms — includes step-by-step Tkinter apps for Bisecting K-Means and DBSCAN with dataset generation, real-time visualization, and quiz mode for learning.

This repository contains two interactive Python applications for learning and visualizing clustering algorithms step-by-step.  
Each tool is built with Tkinter for the interface and Matplotlib for real-time plotting, making them perfect for students, educators, and anyone wanting to understand clustering in depth.

---

Features Across Both Apps:
1. Interactive learning — Watch algorithms process each point.
2. Customizable datasets — Choose size, shape, and noise level.
3. Parameter control — Adjust ε, minPts, target clusters, and more.
4. Clear visualizations — Color-coded clusters and point types.
5. Educational focus — Step-by-step animations and explanations.

**Bisecting K-Means Educational App**
A visual tool to explore Bisecting K-Means:
- Generate synthetic datasets with adjustable:
  - Number of samples
  - Number of true centers
  - Cluster standard deviation
- Iteratively split the cluster with the highest SSE (Sum of Squared Errors).
- Animate each 2-means iteration when splitting.
- Live display of SSE values for all clusters.

Run:
python bisecting_kmeans_o3.py


**DBSCAN Trainer**
An interactive trainer for DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
- Multiple dataset types: Blobs, Moons, Circles, Lines.
- Step-by-step classification of points as:
  - Core (blue)
  - Border (yellow)
  - Noise (vermillion)
  - Undecided (black)
  - Unvisited (grey)
- Hover to see ε-neighborhood details.
- Visualize ε-circles and connections between points.
- Quiz Mode to test your knowledge.
- Second pass to handle undecided points before final clustering.

Run:
python db-scan_o3.py
