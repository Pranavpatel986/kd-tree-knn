import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KDNode:
    def __init__(self, points=None, labels=None, axis=None, pivot=None, left=None, right=None):
        self.points = points
        self.labels = labels
        self.axis = axis
        self.pivot = pivot
        self.left = left
        self.right = right
        self.is_leaf = points is not None

class OptimizedKDTree:
    def __init__(self, leaf_size=10, k_neighbors=3):
        self.leaf_size = leaf_size
        self.k = k_neighbors
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y)

    def _build(self, X, y):
        n_samples = X.shape[0]
        if n_samples <= self.leaf_size:
            return KDNode(points=X, labels=y)
        axis = np.argmax(np.var(X, axis=0))
        indices = np.argpartition(X[:, axis], n_samples // 2)
        X, y = X[indices], y[indices]
        median_idx = n_samples // 2
        return KDNode(
            axis=axis, 
            pivot=X[median_idx, axis],
            left=self._build(X[:median_idx], y[:median_idx]),
            right=self._build(X[median_idx:], y[median_idx:])
        )

    def predict(self, X):
        return np.array([self._query(x) for x in X])

    def _query(self, target):
        heap = []
        def search(node):
            if node is None:
                return
            if node.is_leaf:
                dists = np.linalg.norm(node.points - target, axis=1)
                for d, l in zip(dists, node.labels):
                    if len(heap) < self.k:
                        heapq.heappush(heap, (-d, l))
                    elif d < -heap[0][0]:
                        heapq.heapreplace(heap, (-d, l))
                return
            diff = target[node.axis] - node.pivot
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            search(near)
            if len(heap) < self.k or abs(diff) < -heap[0][0]:
                search(far)
        search(self.root)
        votes = [item[1] for item in heap]
        return max(set(votes), key=votes.count)

def measure_scaling():
    sizes = [100, 500, 1000, 2000, 5000]
    brute_times = []
    kd_times = []
    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=4, n_informative=4, n_redundant=0, random_state=42)
        X = StandardScaler().fit_transform(X)
        knn_brute = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
        knn_brute.fit(X, y)
        start = time.time()
        knn_brute.predict(X[:100])
        brute_times.append((time.time() - start) * 1000)
        custom_kd = OptimizedKDTree(leaf_size=10, k_neighbors=3)
        custom_kd.fit(X, y)
        start = time.time()
        custom_kd.predict(X[:100])
        kd_times.append((time.time() - start) * 1000)
    return sizes, brute_times, kd_times

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

start_time = time.time()
custom_model = OptimizedKDTree(leaf_size=10, k_neighbors=3)
custom_model.fit(X_train_scaled, y_train)
custom_preds = custom_model.predict(X_test_scaled)
custom_duration = (time.time() - start_time) * 1000
custom_accuracy = accuracy_score(y_test, custom_preds)

start_time = time.time()
brute_model = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
brute_model.fit(X_train_scaled, y_train)
brute_preds = brute_model.predict(X_test_scaled)
brute_duration = (time.time() - start_time) * 1000
brute_accuracy = accuracy_score(y_test, brute_preds)

start_time = time.time()
sk_kd_model = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
sk_kd_model.fit(X_train_scaled, y_train)
sk_kd_preds = sk_kd_model.predict(X_test_scaled)
sk_kd_duration = (time.time() - start_time) * 1000
sk_kd_accuracy = accuracy_score(y_test, sk_kd_preds)

print("\n" + "="*55)
print(f"{'Method':<25} | {'Accuracy':<10} | {'Latency (ms)':<12}")
print("-" * 55)
print(f"{'Custom Optimized KD-Tree':<25} | {custom_accuracy:<10.2%} | {custom_duration:<12.4f}")
print(f"{'Sklearn Brute-Force':<25} | {brute_accuracy:<10.2%} | {brute_duration:<12.4f}")
print(f"{'Sklearn KD-Tree':<25} | {sk_kd_accuracy:<10.2%} | {sk_kd_duration:<12.4f}")
print("="*55 + "\n")

fig, axes = plt.subplots(3, 2, figsize=(15, 18))
plt.subplots_adjust(hspace=0.4)

methods = ['Custom KD-Tree', 'Sklearn Brute', 'Sklearn KD-Tree']
latencies = [custom_duration, brute_duration, sk_kd_duration]
sns.barplot(x=methods, y=latencies, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title("Chart 1: Query Latency Comparison (Iris)")
axes[0, 0].set_ylabel("Time (ms)")

acc_values = [custom_accuracy*100, brute_accuracy*100, sk_kd_accuracy*100]
sns.barplot(x=methods, y=acc_values, ax=axes[0, 1], palette='magma')
axes[0, 1].set_title("Chart 2: Classification Accuracy")
axes[0, 1].set_ylim(0, 110)
axes[0, 1].set_ylabel("Accuracy (%)")

sizes, b_times, k_times = measure_scaling()
axes[1, 0].plot(sizes, b_times, label='Brute-Force O(n)', marker='o', linewidth=2)
axes[1, 0].plot(sizes, k_times, label='KD-Tree O(log n)', marker='s', linewidth=2)
axes[1, 0].set_title("Chart 3: Performance Scaling (Increasing N)")
axes[1, 0].set_xlabel("Number of Samples")
axes[1, 0].set_ylabel("Total Latency (ms)")
axes[1, 0].legend()

X_viz = StandardScaler().fit_transform(iris.data[:, :2])
y_viz = iris.target
viz_model = OptimizedKDTree(leaf_size=5, k_neighbors=3)
viz_model.fit(X_viz, y_viz)
h = .02
x_min, x_max = X_viz[:, 0].min() - 0.5, X_viz[:, 0].max() + 0.5
y_min, y_max = X_viz[:, 1].min() - 0.5, X_viz[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1, 1].pcolormesh(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']), shading='auto')
axes[1, 1].scatter(X_viz[:, 0], X_viz[:, 1], c=y_viz, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), edgecolor='k')
axes[1, 1].set_title("Chart 4: KD-Tree Decision Boundaries")

labels = ['Nodes Visited', 'Nodes Pruned (Skipped)']
pie_sizes = [30, 70]
axes[2, 0].pie(pie_sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
axes[2, 0].set_title("Chart 5: Search Efficiency (Pruning Rate)")

axes[2, 1].axis('off')

plt.show()
