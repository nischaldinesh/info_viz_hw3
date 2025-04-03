import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import get_blobs, get_moons, get_circles
from algorithms import run_cblof, run_lof, run_gmm, run_kde, run_isolation_forest, run_lscp, run_pca, run_mcd

app = Flask(__name__)

def generate_heatmap(model, X, title="Heatmap", use_decision_function=True):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    try:
        if use_decision_function:
            scores = model.decision_function(grid_points)
        else:
            scores = model.score_samples(grid_points)
    except AttributeError:
        raise ValueError("The model does not have a decision_function or score_samples method.")

    scores = scores.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(xx, yy, scores, levels=50, cmap='viridis')

    cbar = fig.colorbar(c, ax=ax, label="")

    cbar.set_ticks([])


    cbar.ax.text(1.2, 0.0, 'Likely Inlier', ha='left', va='center',
                 transform=cbar.ax.transAxes, fontsize=10, fontweight='bold')
    cbar.ax.text(1.2, 1.0, 'Likely Outlier', ha='left', va='center',
                 transform=cbar.ax.transAxes, fontsize=10, fontweight='bold')


    ax.scatter(X[:, 0], X[:, 1], c='red', s=10, edgecolor='k')


    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64



class GMMWrapper:
    def __init__(self, gmm_model):
        self.gmm_model = gmm_model

    def decision_function(self, X):
        return -self.gmm_model.score_samples(X)


class KDEWrapper:
    def __init__(self, kde_model):
        self.kde_model = kde_model

    def decision_function(self, X):
        return -self.kde_model.score_samples(X)


class PCAWrapper:
    def __init__(self, pca_model):
        self.pca_model = pca_model

    def decision_function(self, X):
        X_proj = self.pca_model.inverse_transform(self.pca_model.transform(X))
        return np.mean((X - X_proj) ** 2, axis=1)


def get_models(X, contamination=0.1):
    models = {}
    models["CBLOF"] = run_cblof(X, contamination)
    models["LOF"] = run_lof(X, contamination)
    gmm_model = run_gmm(X)
    models["GMM"] = GMMWrapper(gmm_model)
    kde_model = run_kde(X)
    models["KDE"] = KDEWrapper(kde_model)
    models["Isolation Forest"] = run_isolation_forest(X, contamination)
    models["LSCP"] = run_lscp(X, contamination)
    pca_model = run_pca(X)
    models["PCA"] = PCAWrapper(pca_model)
    models["MCD"] = run_mcd(X, contamination)
    return models

def get_models_grouped(X, contamination=0.1):
    models_grouped = {
        "Proximity-based": {
            "CBLOF": run_cblof(X, contamination),
            "LOF": run_lof(X, contamination)
        },
        "Probabilistic": {
            "GMM": GMMWrapper(run_gmm(X)),
            "KDE": KDEWrapper(run_kde(X))
        },
        "Ensemble": {
            "Isolation Forest": run_isolation_forest(X, contamination),
            "LSCP": run_lscp(X, contamination)
        },
        "Linear Models": {
            "PCA": PCAWrapper(run_pca(X)),
            "MCD": run_mcd(X, contamination)
        }
    }
    return models_grouped


datasets = {
    "Blobs": get_blobs(),
    "Moons": get_moons(),
    "Circles": get_circles()
}


@app.route('/')
def index():
    results = {}
    for ds_name, X in datasets.items():
        models_by_type = get_models_grouped(X)
        results[ds_name] = {}
        for category, models in models_by_type.items():
            results[ds_name][category] = {}
            for model_name, model in models.items():
                title = f"{model_name} Heatmap on {ds_name}"
                image_base64 = generate_heatmap(model, X, title=title, use_decision_function=True)
                results[ds_name][category][model_name] = image_base64
    return render_template('index.html', results=results)



if __name__ == '__main__':
    app.run(debug=True)
