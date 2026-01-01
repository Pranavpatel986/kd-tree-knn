k-d Tree for k-Nearest Neighbors (k-NN) Optimization

🚀 Overview

This project implements a custom, optimized k-dimensional tree (k-d tree) data structure designed to accelerate k-Nearest Neighbor classification. By partitioning multidimensional space, this implementation reduces the search complexity from a linear brute-force scan  to a logarithmic search .

🛠️ Key Features
Recursive Tree Construction: Builds a balanced tree by choosing the split axis based on maximum variance to handle skewed data distributions efficiently.
Pruning Logic: The search algorithm uses backtracking to skip entire branches of the tree that cannot possibly contain a closer neighbor, significantly reducing latency.
Leaf Size Optimization: Includes a configurable `leaf_size` parameter to switch to brute-force distance calculation for small clusters, avoiding unnecessary recursion overhead.
Standardized Benchmarking: Includes a script to compare accuracy and query speed against `scikit-learn`'s brute-force and optimized k-d tree implementations.

📊 Performance Results

The implementation was evaluated using the Iris Dataset (4 features, 3 classes).

| Implementation | Accuracy | Latency (ms) |
| Custom k-d Tree | 100.00% | ~18ms |
| Brute-Force k-NN | 100.00% | ~50ms |
| Scikit-Learn (Native)| 100.00% | ~4ms |

Analysis: The custom k-d tree achieved a ~60% improvement in speed over brute-force while maintaining perfect classification accuracy.*

💻 Installation & Usage

Prerequisites:

* Python 3.x
* NumPy
* Scikit-Learn
* Matplotlib

How to run:

1. Clone the repository: `git clone https://github.com/yourusername/kd-tree-knn.git`
2. Run the main script: `python main.py`


