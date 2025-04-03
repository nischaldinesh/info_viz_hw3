from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

def get_blobs(n_samples=500, centers=3, cluster_std=1.0, random_state=42):
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)
    return StandardScaler().fit_transform(X)

def get_moons(n_samples=500, noise=0.05, random_state=42):
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return StandardScaler().fit_transform(X)

def get_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42):
    X, _ = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return StandardScaler().fit_transform(X)
