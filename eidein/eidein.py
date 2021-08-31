import ipywidgets
from matplotlib import pyplot
import numpy
from sklearn import decomposition
from sklearn import manifold
import umap


class Eidein(ipywidgets.HBox):
    def __init__(self, ids, X, y):
        super().__init__()

        self.ids, self.X, self.y = ids, X, y

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
            n_epochs=ipywidgets.fixed(None),
            learning_rate=ipywidgets.FloatText(value=1.0, min=0, step=0.1),
            init=["spectral", "random"],
            min_dist=ipywidgets.FloatText(value=0.1, min=0, step=0.1))

        self.reducer_out = ipywidgets.Output(layout={'border': '1px solid black'})

        proj_out = ipywidgets.Output()
        with proj_out:
            self.proj_fig, self.proj_ax = pyplot.subplots(constrained_layout=True)

        spec_out = ipywidgets.Output()
        with spec_out:
            self.spec_fig, self.spec_ax = pyplot.subplots(constrained_layout=True)

        tabs = ipywidgets.Tab()
        tabs.children = [pca_widget, tsne_widget, umap_widget]
        for i, title in enumerate(['PCA', 't-SNE', 'UMAP']):
            tabs.set_title(i, title)

        self.proj_fig.canvas.mpl_connect("pick_event", self.onpick)

        self.layout = ipywidgets.Layout(align_items="stretch", flex_flow="row wrap")
        self.children = [tabs, self.reducer_out, proj_out, spec_out]

    def reduce_dim(self, reducer):
        with self.reducer_out:
            self.reducer_out.clear_output()
            return reducer.fit_transform(self.X)

    def plot_projection(self, embedding):
        self.proj_ax.clear()
        collection = self.proj_ax.scatter(embedding[:, 0], embedding[:, 1], c=self.y, picker=True)
        self.proj_fig.colorbar(collection, ax=self.proj_ax)

    def onpick(self, event):
        flux = self.X[event.ind[0]]
        # TODO provide custom plot function
        LOGLAMMIN, LOGLAMMAX = 3.5832, 3.9583
        N_FEATURES = 3752
        wave = numpy.power(10, numpy.linspace(LOGLAMMIN, LOGLAMMAX, N_FEATURES))
        self.spec_ax.clear()
        self.spec_ax.plot(wave, flux)
        self.spec_ax.grid(True)
        print(flux, wave)

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
