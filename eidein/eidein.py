import ipywidgets
from matplotlib import pyplot
import numpy
from sklearn import decomposition
from sklearn import manifold
import umap


class Eidein(ipywidgets.HBox):
    def __init__(self, ids, X, y, plot_fn, widget_label):
        super().__init__()

        self.ids, self.X, self.y = ids, X, y
        self.plot_fn = plot_fn
        self.widget_label = widget_label

        pca_widget = ipywidgets.interactive(
            self.pca, {"manual": True},
            svd_solver=["auto", "full", "arpack", "randomized"])

        tsne_widget = ipywidgets.interactive(
            self.tsne, {"manual": True},
            perplexity=(5, 50, 1),
            early_exaggeration=ipywidgets.FloatText(value=12.0, step=0.1),
            learning_rate=ipywidgets.FloatText(value=200.0, step=0.1),
            n_iter=ipywidgets.IntText(1000),
            init=["random", "pca"],
            method=["barnes_hut", "exact"],
            n_jobs=ipywidgets.IntText(value=-1))

        umap_widget = ipywidgets.interactive(
            self.umap, {"manual": True},
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

        spec_out = ipywidgets.Output()
        with spec_out:
            self.data_fig, self.data_ax = pyplot.subplots(constrained_layout=True)

        tabs = ipywidgets.Tab()
        tabs.children = [pca_widget, tsne_widget, umap_widget]
        for i, title in enumerate(['PCA', 't-SNE', 'UMAP']):
            tabs.set_title(i, title)

        self.proj_fig.canvas.mpl_connect("pick_event", self.onpick)

        self.idx_picked = None
        self.widget_label.observe(self.plot_data, 'value')

        self.layout = ipywidgets.Layout(align_items="stretch", flex_flow="row wrap")
        self.children = [tabs, self.reducer_out, proj_out, spec_out, self.widget_label]

    def reduce_dim(self, reducer):
        with self.reducer_out:
            self.reducer_out.clear_output()
            return reducer.fit_transform(self.X)

    def plot_projection(self, embedding):
        self.proj_ax.clear()
        collection = self.proj_ax.scatter(embedding[:, 0], embedding[:, 1], c=self.y, picker=True)
        self.proj_fig.colorbar(collection, ax=self.proj_ax)

    def plot_data(self, change=None):
        self.data_ax.clear()
        identifier = self.ids[self.idx_picked]
        x = self.X[self.idx_picked]
        y = self.y[self.idx_picked]
        label = self.widget_label.value
        self.plot_fn(self.data_ax, identifier, x, y, label)

    def onpick(self, event):
        # on pick update the picked index and plot the data with the index
        self.idx_picked = event.ind[0]
        # set the label to be the predicted label (i.e. y)
        self.widget_label.value = self.y[self.idx_picked]
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
