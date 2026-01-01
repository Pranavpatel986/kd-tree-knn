import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
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

def run_experiment():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start = time.time()
    custom_model = OptimizedKDTree(leaf_size=10, k_neighbors=3)
    custom_model.fit(X_train_scaled, y_train)
    custom_preds = custom_model.predict(X_test_scaled)
    custom_time = (time.time() - start) * 1000
    custom_acc = accuracy_score(y_test, custom_preds)

    start = time.time()
    brute_model = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
    brute_model.fit(X_train_scaled, y_train)
    brute_preds = brute_model.predict(X_test_scaled)
    brute_time = (time.time() - start) * 1000
    brute_acc = accuracy_score(y_test, brute_preds)

    print(f"\n{'Implementation':<25} | {'Accuracy':<10} | {'Latency (ms)':<12}")
    print("-" * 55)
    print(f"{'Custom Optimized k-d Tree':<25} | {custom_acc:<10.2%} | {custom_time:<12.4f}")
    print(f"{'Sklearn Brute-Force':<25} | {brute_acc:<10.2%} | {brute_time:<12.4f}")

def plot_boundaries():
    iris = load_iris()
    X = iris.data[:, :2] 
    y = iris.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = OptimizedKDTree(leaf_size=5, k_neighbors=3)
    model.fit(X_scaled, y)

    h = .05
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=40)
    plt.title("k-d Tree Classification Boundaries (Iris Dataset)")
    plt.xlabel("Sepal Length (Standardized)")
    plt.ylabel("Sepal Width (Standardized)")
    plt.show()

if __name__ == "__main__":
    run_experiment()
    plot_boundaries()
