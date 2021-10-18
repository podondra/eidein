import ipywidgets
from matplotlib import cm
from matplotlib import pyplot
import numpy
from sklearn import decomposition
from sklearn import manifold
import umap


class Eidein(ipywidgets.HBox):
    def __init__(self, ids, X, y, uncertainty, plot_fn, label_widget):
        super().__init__()

        self.ids, self.X, self.y = ids, X, y
        self.uncertainty = uncertainty
        self.plot_fn = plot_fn
        self.label_widget = label_widget
        self.picked_idx = None
        self.labelled = dict()

        pca_widget = ipywidgets.interactive(
            self.pca, {"manual": True, "manual_name": "Run PCA"},
            svd_solver=["auto", "full", "arpack", "randomized"])

        tsne_widget = ipywidgets.interactive(
            self.tsne, {"manual": True, "manual_name": "Run t-SNE"},
            perplexity=(5, 50, 1),
            early_exaggeration=ipywidgets.FloatText(value=12.0, step=0.1),
            learning_rate=ipywidgets.FloatText(value=200.0, step=0.1),
            n_iter=ipywidgets.IntText(1000),
            init=["random", "pca"],
            method=["barnes_hut", "exact"],
            n_jobs=ipywidgets.IntText(value=-1))

        umap_widget = ipywidgets.interactive(
            self.umap, {"manual": True, "manual_name": "Run UMAP"},
            n_neighbors=(2, 100, 1),
            metric=["euclidean", "manhattan", "chebyshev", "minkowski"],
            # TODO why I have here ipywidgets.fixed(None)?
            n_epochs=ipywidgets.fixed(None),
            learning_rate=ipywidgets.FloatText(value=1.0, min=0, step=0.1),
            init=["spectral", "random"],
            min_dist=ipywidgets.FloatText(value=0.1, min=0, step=0.1))

        # output of reducers (PCA, t-SNE, UMAP)
        self.reducer_out = ipywidgets.Output(layout={'border': '1px solid black'})

        proj_out = ipywidgets.Output()
        with proj_out:
            self.proj_fig, self.proj_ax = pyplot.subplots(constrained_layout=True)
            self.proj_cbar = self.proj_fig.colorbar(cm.ScalarMappable(), ax=self.proj_ax)
            self.proj_cbar.set_label("Uncertainty")

        spec_out = ipywidgets.Output()
        with spec_out:
            self.data_fig, self.data_ax = pyplot.subplots(constrained_layout=True)

        tabs = ipywidgets.Tab()
        tabs.children = [pca_widget, tsne_widget, umap_widget]
        for i, title in enumerate(['PCA', 't-SNE', 'UMAP']):
            tabs.set_title(i, title)

        self.proj_fig.canvas.mpl_connect("pick_event", self.onpick)
        self.label_widget.observe(self.plot_data, 'value')

        self.label_interactive = ipywidgets.interactive(
                self.add2labelled, {"manual": True, "manual_name": "Label"},
                label=self.label_widget)

        self.layout = ipywidgets.Layout(align_items="stretch", flex_flow="row wrap")
        self.children = [
                tabs,
                self.reducer_out,
                proj_out,
                spec_out,
                self.label_interactive]

    def reduce_dim(self, reducer):
        with self.reducer_out:
            self.reducer_out.clear_output()
            return reducer.fit_transform(self.X)

    def plot_projection(self, embedding):
        self.proj_ax.clear()
        collection = self.proj_ax.scatter(
                embedding[:, 0], embedding[:, 1], c=self.uncertainty, picker=True)
        self.proj_cbar.update_normal(collection)

    def plot_data(self, change=None):
        self.data_ax.clear()
        identifier = self.ids[self.picked_idx]
        x = self.X[self.picked_idx]
        y = self.y[self.picked_idx]
        label = self.label_widget.value
        self.plot_fn(self.data_ax, identifier, x, y, label)

    def onpick(self, event):
        # on pick update the picked index and plot the data with the index
        self.picked_idx = event.ind[0]
        # set the label to be the predicted label (i.e. y)
        self.label_widget.value = self.y[self.picked_idx]
        self.plot_data()

    def pca(self, whiten=False, svd_solver="auto"):
        reducer = decomposition.PCA(n_components=2, whiten=whiten, svd_solver=svd_solver)
        embedding = self.reduce_dim(reducer)
        self.plot_projection(embedding)

    def tsne(
            self, perplexity=30, early_exaggeration=12.0, learning_rate=200.0,
            n_iter=1000, init="random", method="barnes_hut", n_jobs=-1):
        reducer = manifold.TSNE(
            n_components=2, perplexity=perplexity,
            early_exaggeration=early_exaggeration, learning_rate=learning_rate,
            n_iter=n_iter, init=init, verbose=2, method=method, n_jobs=n_jobs)
        self.plot_projection(self.reduce_dim(reducer))

    def umap(
            self, n_neighbors=15, metric="euclidean", n_epochs=None, learning_rate=1.0,
            init="spectral", min_dist=0.1, low_memory=False):
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, metric=metric, n_epochs=n_epochs,
            learning_rate=learning_rate, init=init, min_dist=min_dist,
            low_memory=low_memory, verbose=True)
        self.plot_projection(self.reduce_dim(reducer))

    def add2labelled(self, label):
        self.labelled[self.ids[self.picked_idx]] = label
