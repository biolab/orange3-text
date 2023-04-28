# pylint: disable=too-many-ancestors
from itertools import chain
from types import SimpleNamespace
from typing import Dict, Optional, Union, Iterable

import numpy as np
from AnyQt.QtCore import Qt, QRectF, QObject
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QWidget

import pyqtgraph as pg
from Orange.data import Domain, ContinuousVariable, DiscreteVariable, Table, \
    StringVariable
from Orange.data.util import array_equal, get_unique_names
from Orange.widgets import gui, report
from Orange.widgets.settings import Setting, ContextSetting, SettingProvider, \
    DomainContextHandler, Context
from Orange.widgets.utils.colorpalettes import LimitedDiscretePalette
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget
from Orange.widgets.widget import Msg, Input, Output
from orangecontrib.text import Corpus
from orangecontrib.text.annotate_documents import annotate_documents, \
    ClusterDocuments, ClusterType, ScoresType
from orangecontrib.text.widgets.utils.words import create_words_table


class _Clusters(SimpleNamespace):
    cluster_labels: Optional[np.ndarray] = None
    groups: Optional[Dict[int, ClusterType]] = None
    n_components: Optional[int] = None
    epsilon: Optional[float] = None
    scores: Optional[ScoresType] = None


def run(
        corpus: Optional[Corpus],
        attr_x: ContinuousVariable,
        attr_y: ContinuousVariable,
        clustering_method: int,
        n_components: Optional[int],
        epsilon: Optional[float],
        cluster_labels: Optional[np.ndarray],
        fdr_threshold: float,
        state: TaskState
) -> _Clusters:
    if not corpus:
        return _Clusters(cluster_labels=None, groups=None,
                         n_components=None, epsilon=None, scores=None)

    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    callback(0, "Calculating...")

    cluster_labels, clusters, n_components, epsilon, scores = \
        annotate_documents(
            corpus,
            corpus.transform(Domain([attr_x, attr_y])).X,
            clustering_method,
            n_components=n_components,
            epsilon=epsilon,
            cluster_labels=cluster_labels,
            fdr_threshold=fdr_threshold,
            progress_callback=callback
        )

    return _Clusters(cluster_labels=cluster_labels, groups=clusters,
                     n_components=n_components, epsilon=epsilon, scores=scores)


def index_to_cluster_label(index: Union[int, float]) -> str:
    return f"C{int(index) + 1}"


class CenteredTextItem(pg.TextItem):
    def __init__(self, view_box, x, y, words, tooltip):
        bg_color = QColor(Qt.white)
        bg_color.setAlpha(200)
        color = QColor(Qt.black)
        super().__init__(
            color=pg.mkColor(color),
            fill=pg.mkBrush(bg_color),
            html="<br>".join(words),
        )
        option = self.textItem.document().defaultTextOption()
        option.setAlignment(Qt.AlignCenter)
        self.textItem.document().setDefaultTextOption(option)
        self.textItem.setTextWidth(self.textItem.boundingRect().width())

        self._x = x
        self._y = y
        self._view_box = view_box
        self._view_box.sigStateChanged.connect(self.center)
        self.textItem.setToolTip(tooltip)
        self.setPos(x, y)
        self.center()

    def center(self):
        br = self.boundingRect()
        dx = br.width() / 2 * self._view_box.viewPixelSize()[0]
        dy = br.height() / 2 * self._view_box.viewPixelSize()[1]
        self.setPos(self._x - dx, self._y + dy)


class EventDelegate(QObject):
    def eventFilter(self, *_):
        return False


class OWAnnotatorGraph(OWScatterPlotBase):
    show_cluster_hull = Setting(True)
    n_cluster_labels = Setting(3)

    def __init__(self, scatter_widget, parent):
        super().__init__(scatter_widget, parent)
        self.cluster_hulls_items = []
        self.cluster_labels_items = []
        self._tooltip_delegate = EventDelegate()  # remove points tooltip

    def clear(self):
        super().clear()
        self.cluster_hulls_items.clear()
        self.cluster_labels_items.clear()

    def reset_view(self):
        x, y = [self.get_coordinates()[0]], [self.get_coordinates()[1]]
        if x[0] is None or y[0] is None:
            return

        hulls = self.master.get_cluster_hulls()
        if hulls is not None:
            x.extend([hull[:, 0] for hull, _ in hulls])
            y.extend([hull[:, 1] for hull, _ in hulls])

        x, y = np.hstack(x), np.hstack(y)
        min_x, max_x, min_y, max_y = np.min(x), np.max(x), np.min(y), np.max(y)
        rect = QRectF(min_x, min_y, max_x - min_x or 1, max_y - min_y or 1)
        self.view_box.setRange(rect, padding=0.025)

    def update_coordinates(self):
        super().update_coordinates()
        self.update_clusters()
        self.view_box.setAspectLocked(True, 1)
        self.reset_view()

    def update_clusters(self):
        self._update_cluster_hull()
        self._update_cluster_labels()

    def _update_cluster_hull(self):
        for item in self.cluster_hulls_items:
            self.plot_widget.removeItem(item)
        if not self.show_cluster_hull:
            return
        hulls = self.master.get_cluster_hulls()
        if hulls is None:
            return
        for hull, color in hulls:
            pen = pg.mkPen(color=QColor(*color), style=Qt.DashLine, width=3)
            item = pg.PlotCurveItem(x=np.hstack([hull[:, 0], hull[:1, 0]]),
                                    y=np.hstack([hull[:, 1], hull[:1, 1]]),
                                    pen=pen, antialias=True)
            self.plot_widget.addItem(item)
            self.cluster_hulls_items.append(item)

    def _update_cluster_labels(self):
        for item in self.cluster_labels_items:
            self.plot_widget.removeItem(item)
        if not self.n_cluster_labels:
            return
        labels = self.master.get_cluster_labels()
        if labels is None:
            return
        for label_per, (x, y), _ in labels:
            words = [label for label, _ in label_per[: self.n_cluster_labels]]
            ttip = "\n".join([f"{round(p * 100)}%  {label}"
                              for label, p in label_per])
            item = CenteredTextItem(self.view_box, x, y, words, ttip)
            if words:
                self.plot_widget.addItem(item)
                self.cluster_labels_items.append(item)


class AnnotatorContextHandler(DomainContextHandler):
    def match(self, context: Context, domain: Domain,
              attrs: Dict, metas: Dict) -> float:
        cluster_var = context.values.get("cluster_var", (None, -2))[0]
        if domain.has_discrete_attributes(True, True) and cluster_var is None:
            return self.NO_MATCH
        return super().match(context, domain, attrs, metas)


class OWAnnotator(OWDataProjectionWidget, ConcurrentWidgetMixin):
    name = "Annotated Corpus Map"
    description = "Annotates projection clusters."
    icon = "icons/Annotator.svg"
    priority = 1110
    keywords = "annotated corpus map, annotator"

    settingsHandler = AnnotatorContextHandler()
    GRAPH_CLASS = OWAnnotatorGraph
    graph = SettingProvider(OWAnnotatorGraph)

    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    p_threshold = Setting(0.05)
    clustering_type = Setting(ClusterDocuments.DBSCAN)
    use_epsilon = Setting(False)
    epsilon = Setting(0)
    use_n_components = Setting(False)
    n_components = Setting(0)
    cluster_var = ContextSetting(None)
    color_by_cluster = Setting(False)

    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        selected_data = Output("Selected Docs", Corpus, default=True)
        annotated_data = Output("Corpus", Corpus)
        scores = Output("Scores", Table)

    class Warning(OWDataProjectionWidget.Warning):
        same_axis_features = Msg("Selected features for Axis x "
                                 "and Axis y should differ.")

    class Error(OWDataProjectionWidget.Error):
        no_continuous_vars = Msg("Data has no continuous variables.")
        not_enough_inst = Msg("Not enough instances in data.")
        proj_error = Msg("An error occurred while annotating data.\n{}")

    def __init__(self):
        self.dbscan_box: QWidget = None
        self.gmm_box: QWidget = None
        self.var_box: QWidget = None
        OWDataProjectionWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.clusters: Optional[_Clusters] = None
        self.enable_controls()
        self._invalidate_clusters()

    # GUI
    def _add_controls(self):
        self.__add_annotation_controls()
        super()._add_controls()
        self.gui.add_control(self._effects_box, gui.hSlider, "Cluster labels:",
                             master=self.graph, value="n_cluster_labels",
                             minValue=0, maxValue=5, step=1, createLabel=False,
                             callback=self.graph.update_clusters)
        self._plot_box.children()[1].hide()  # Hide 'Show color regions'
        gui.checkBox(self._plot_box, self.graph, "show_cluster_hull",
                     "Show cluster hull", callback=self.graph.update_clusters)
        gui.checkBox(self._plot_box, self, "color_by_cluster",
                     "Color points by cluster",
                     callback=self.__on_color_by_cluster_changed)

    def __add_annotation_controls(self):
        combo_options = {"labelWidth": 50, "orientation": Qt.Horizontal,
                         "searchable": True, "sendSelectedValue": True,
                         "contentsLength": 14}
        box = gui.vBox(self.controlArea, box="Axes")
        order = DomainModel.METAS, DomainModel.ATTRIBUTES, DomainModel.CLASSES
        mod = DomainModel(order, valid_types=ContinuousVariable)
        gui.comboBox(box, self, "attr_x", label="Axis x:", model=mod,
                     callback=self.__on_axis_attr_changed, **combo_options)
        gui.comboBox(box, self, "attr_y", label="Axis y:", model=mod,
                     callback=self.__on_axis_attr_changed, **combo_options)

        box = gui.vBox(self.controlArea, box="Annotation")
        rbuttons = gui.radioButtons(box, self, "clustering_type",
                                    callback=self.__on_clustering_type_changed)

        gui.appendRadioButton(rbuttons, "DBSCAN")
        self.dbscan_box = ibox = gui.indentedBox(rbuttons, 20,
                                                 orientation=Qt.Horizontal)
        gui.checkBox(ibox, self, "use_epsilon", "epsilon:",  labelWidth=80,
                     callback=self.__on_epsilon_check_changed)
        gui.doubleSpin(ibox, self, "epsilon", 0.1, 10, 0.1,
                       callback=self.__on_epsilon_changed)

        gui.appendRadioButton(rbuttons, "Gaussian mixture models")
        self.gmm_box = ibox = gui.indentedBox(rbuttons, 20,
                                              orientation=Qt.Horizontal)
        gui.checkBox(ibox, self, "use_n_components", "clusters:", labelWidth=80,
                     callback=self.__on_n_components_check_changed)
        gui.spin(ibox, self, "n_components", 1, 100,
                 callback=self.__on_n_components_changed)

        gui.appendRadioButton(rbuttons, "From variable")
        self.var_box = ibox = gui.indentedBox(rbuttons, 20,
                                              orientation=Qt.Horizontal)
        gui.comboBox(ibox, self, "cluster_var",
                     model=DomainModel(order, valid_types=DiscreteVariable),
                     callback=self.__on_cluster_var_changed, **combo_options)

        gui.doubleSpin(box, self, "p_threshold", 0, 1, 0.01,
                       label="FDR threshold:", labelWidth=100,
                       callback=self.__on_fdr_threshold_changed)

    @property
    def effective_variables(self):
        attr_x, attr_y = self.attr_x, self.attr_y
        if attr_x and attr_y:
            if attr_x == attr_y:
                return [attr_x]
            return [attr_x, attr_y]
        return []

    @property
    def effective_data(self):
        return self.data.transform(Domain(self.effective_variables))

    def __on_axis_attr_changed(self):
        self._invalidate_clusters()
        self.setup_plot()
        self._run()

    def __on_clustering_type_changed(self):
        self._enable_clustering_controls()
        self._invalidate_clusters()
        self._run()

    def __on_epsilon_check_changed(self):
        self._enable_epsilon_spin()
        if not self.use_epsilon:
            self.__on_epsilon_changed()

    def __on_epsilon_changed(self):
        self._run()

    def __on_n_components_check_changed(self):
        self._enable_n_components_spin()
        if not self.use_n_components:
            self.__on_n_components_changed()

    def __on_n_components_changed(self):
        self._run()

    def __on_cluster_var_changed(self):
        self._invalidate_clusters()
        self._run()

    def __on_fdr_threshold_changed(self):
        self._run()

    def __on_color_by_cluster_changed(self):
        self.controls.attr_color.setEnabled(not self.color_by_cluster)
        self.graph.update_colors()

    @Inputs.corpus
    def set_data(self, data: Optional[Corpus]):
        attr_x, attr_y = self.attr_x, self.attr_y
        data_existed = self.data is not None
        effective_data = self.effective_data if data_existed else None
        super().set_data(data)
        if not (data_existed and self.data is not None and
                array_equal(effective_data.X, self.effective_data.X)) or \
                attr_x is not self.attr_x or attr_y is not self.attr_y:
            self.clear()
            self._invalidate_clusters()

    def handleNewSignals(self):
        invalidated = bool(self._invalidated)
        super().handleNewSignals()
        if invalidated:
            self._run()

    def _invalidate_clusters(self):
        self.clusters = _Clusters(cluster_labels=None, groups=None,
                                  n_components=None, epsilon=None, scores=None)

    def _run(self):
        self.Error.proj_error.clear()
        self.graph.update_clusters()  # Remove cluster hulls and labels
        can_annotate = (self.data and self.attr_x and self.attr_y
                        and self.attr_x is not self.attr_y)
        if not can_annotate:
            return

        if not self.valid_data.all():
            self.Error.proj_error("No valid embedding data.")
            return

        n_components, epsilon, labels = None, None, None
        if self.clustering_type == ClusterDocuments.DBSCAN:
            if self.use_epsilon:
                epsilon = self.epsilon
        elif self.clustering_type == ClusterDocuments.GAUSSIAN_MIXTURE:
            if self.use_n_components:
                n_components = self.n_components
        else:
            assert self.cluster_var
            column = self.data.get_column(self.cluster_var)
            labels = column.astype(float)
        self.start(run, self.data, self.attr_x, self.attr_y,
                   self.clustering_type, n_components, epsilon, labels,
                   self.p_threshold)

    def on_partial_result(self, clusters: _Clusters):
        pass

    def on_done(self, clusters: _Clusters):
        if clusters.epsilon is not None:
            self.epsilon = clusters.epsilon
        if clusters.n_components is not None:
            self.n_components = clusters.n_components
        self.clusters = clusters
        self.graph.update_clusters()
        self.graph.update_colors()
        self.graph.reset_view()
        self.commit.deferred()

    def on_exception(self, ex: Exception):
        self.Error.proj_error(ex)

    def check_data(self):
        self.Error.no_continuous_vars.clear()
        self.Error.not_enough_inst.clear()
        if self.data:
            if len(self.data) < 2:
                self.Error.not_enough_inst()
                self.data = None
            elif not self.data.domain.has_continuous_attributes(True, True):
                self.Error.no_continuous_vars()
                self.data = None

    def init_attr_values(self):
        super().init_attr_values()
        domain = self.data.domain if self.data and len(self.data) > 0 else None

        model = self.controls.attr_x.model()
        model.set_domain(domain)
        self.attr_x = model[0] if model else None
        self.attr_y = model[1] if len(model) >= 2 else self.attr_x

        model = self.controls.cluster_var.model()
        model.set_domain(domain)
        self.cluster_var = model[0] if model else None

    def enable_controls(self):
        super().enable_controls()
        self._enable_clustering_controls()
        self._enable_epsilon_spin()
        self._enable_n_components_spin()
        self.controls.attr_color.setEnabled(not self.color_by_cluster)

    def _enable_clustering_controls(self):
        enable = bool(len(self.controls.cluster_var.model()))
        index = len(ClusterDocuments.TYPES)
        self.controls.clustering_type.buttons[index].setEnabled(enable)
        for i, box in enumerate([self.dbscan_box, self.gmm_box, self.var_box]):
            box.setEnabled(i == self.clustering_type)

        if self.clustering_type not in ClusterDocuments.TYPES and not enable:
            self.clustering_type = ClusterDocuments.DBSCAN

    def _enable_epsilon_spin(self):
        self.controls.epsilon.setEnabled(self.use_epsilon)

    def _enable_n_components_spin(self):
        self.controls.n_components.setEnabled(self.use_n_components)

    def get_embedding(self):
        self.Warning.same_axis_features.clear()
        if not self.data:
            self.valid_data = None
            return None

        if self.attr_x is self.attr_y and self.attr_x is not None:
            self.Warning.same_axis_features()

        x_data = self.get_column(self.attr_x, filter_valid=False)
        y_data = self.get_column(self.attr_y, filter_valid=False)
        if x_data is None or y_data is None:
            return None

        self.valid_data = np.isfinite(x_data) & np.isfinite(y_data)
        return np.vstack((x_data, y_data)).T

    def get_color_data(self):
        if not self.color_by_cluster or self.clusters.cluster_labels is None:
            return super().get_color_data()

        all_data = self.clusters.cluster_labels
        if self.valid_data is not None:
            all_data = all_data[self.valid_data]
        return all_data

    def get_color_labels(self):
        if not self.color_by_cluster or not self.clusters.groups:
            return super().get_color_labels()

        if self.clustering_type not in ClusterDocuments.TYPES:  # From variable
            return self.get_column(self.cluster_var, return_labels=True)

        return [index_to_cluster_label(key) for key in self.clusters.groups]

    def is_continuous_color(self):
        if not self.color_by_cluster or not self.clusters.groups:
            return super().is_continuous_color()
        return False

    def get_palette(self):
        if not self.color_by_cluster or not self.clusters.groups:
            return super().get_palette()

        if self.clustering_type not in ClusterDocuments.TYPES:  # From variable
            return self.cluster_var.palette

        return LimitedDiscretePalette(len(self.clusters.groups))

    def get_cluster_hulls(self):
        if not self.clusters.groups:
            return None

        if self.clustering_type not in ClusterDocuments.TYPES:  # From variable
            colors = self.cluster_var.colors
        else:
            colors = LimitedDiscretePalette(len(self.clusters.groups)).palette
        return [(hull, colors[key])
                for key, (_, _, hull) in self.clusters.groups.items()]

    def get_cluster_labels(self):
        if not self.clusters.groups:
            return None

        if self.clustering_type not in ClusterDocuments.TYPES:  # From variable
            colors = self.cluster_var.colors
        else:
            colors = LimitedDiscretePalette(len(self.clusters.groups)).palette
        return [(ann, centroid, colors[key])
                for key, (ann, centroid, _) in self.clusters.groups.items()]

    @gui.deferred
    def commit(self):
        super().commit()
        self.send_scores()

    def send_scores(self):
        table = None

        if self.clusters.scores is not None:
            keywords, scores, p_values = self.clusters.scores
            table = create_words_table(keywords)
            table.name = "Scores"

            for i, key in enumerate(self.clusters.groups):
                label = index_to_cluster_label(key)

                var = ContinuousVariable(f"Score({label})")
                table = table.add_column(var, scores[i])

                var = ContinuousVariable(f"p_value({label})")
                table = table.add_column(var, p_values[i])

        self.Outputs.scores.send(table)

    def _get_projection_data(self):
        labels = self.clusters.cluster_labels
        if labels is None:
            return self.data

        corpus: Corpus = self.data

        if self.clustering_type in ClusterDocuments.TYPES:
            name = get_unique_names(self.data.domain, "Cluster")
            values = (index_to_cluster_label(i) for i in
                      sorted(set(labels[~np.isnan(labels)])))
            var = DiscreteVariable(name, values=values)
            corpus = corpus.add_column(var, labels, to_metas=True)

        return corpus

    def _get_send_report_caption(self):
        color_vr_name = "Clusters" if self.color_by_cluster else \
            self._get_caption_var_name(self.attr_color)
        return report.render_items_vert(
            (
                ("Color", color_vr_name),
                ("Label", self._get_caption_var_name(self.attr_label)),
                ("Shape", self._get_caption_var_name(self.attr_shape)),
                ("Size", self._get_caption_var_name(self.attr_size)),
                ("Jittering", self.graph.jitter_size != 0 and "{} %".format(
                    self.graph.jitter_size)),
            )
        )

    def clear(self):
        super().clear()
        self.cancel()
        self._invalidated = True

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.projection import PCA
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    from orangecontrib.text.preprocess import LowercaseTransformer, \
        RegexpTokenizer, StopwordsFilter, FrequencyFilter
    from orangecontrib.text.vectorization import BowVectorizer

    corpus_ = Corpus.from_file("book-excerpts")
    for pp in (LowercaseTransformer(), RegexpTokenizer(r"\w+"),
               StopwordsFilter("English"), FrequencyFilter(0.1)):
        corpus_ = pp(corpus_)

    transformed_corpus = BowVectorizer().transform(corpus_)

    pca = PCA(n_components=2)
    pca_model = pca(transformed_corpus)
    projection = pca_model(transformed_corpus)

    domain_ = Domain(
        transformed_corpus.domain.attributes,
        transformed_corpus.domain.class_vars,
        chain(transformed_corpus.domain.metas,
              projection.domain.attributes)
    )
    corpus_ = corpus_.transform(domain_)

    WidgetPreview(OWAnnotator).run(set_data=corpus_)
