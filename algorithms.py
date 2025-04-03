def run_cblof(X, contamination=0.1):
    from pyod.models.cblof import CBLOF
    model = CBLOF(contamination=contamination, check_estimator=False)
    model.fit(X)
    return model


def run_lof(X, contamination=0.1):
    from pyod.models.lof import LOF
    model = LOF(contamination=contamination)
    model.fit(X)
    return model


def run_gmm(X, n_components=3):
    from sklearn.mixture import GaussianMixture
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(X)
    return model


def run_kde(X, bandwidth=0.3):
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(X)
    return kde


def run_isolation_forest(X, contamination=0.1):
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    return model


def run_lscp(X, contamination=0.1):
    from pyod.models.lscp import LSCP
    from pyod.models.lof import LOF
    from pyod.models.iforest import IForest
    detectors = [
        LOF(contamination=contamination),
        IForest(contamination=contamination, random_state=42)
    ]
    for detector in detectors:
        detector.fit(X)

    model = LSCP(detector_list=detectors, contamination=contamination)
    model.fit(X)
    return model


def run_pca(X, n_components=1):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def run_mcd(X, contamination=0.1):
    from sklearn.covariance import EllipticEnvelope
    model = EllipticEnvelope(contamination=contamination, random_state=42)
    model.fit(X)
    return model