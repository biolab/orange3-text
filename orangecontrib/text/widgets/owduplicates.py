from collections import Counter

import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import Qt, QLineF
from AnyQt.QtWidgets import QApplication, QListView

from Orange.clustering.hierarchical import dist_matrix_linkage
from Orange.data import Table, DiscreteVariable, Domain
from Orange.misc import DistMatrix
from Orange.widgets import gui, widget, settings
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.widget import Msg, OWWidget, Input, Output
from orangecontrib.text import Corpus


class OWDuplicates(widget.OWWidget):
    name = 'Duplicate Detection'
    description = 'Detect & remove duplicates from a corpus.'
    icon = 'icons/Duplicates.svg'
    priority = 700

    class Inputs:
        distances = Input("Distances", DistMatrix)

    class Outputs:
        corpus_without_duplicates = Output("Corpus Without Duplicates", Corpus)
        duplicates = Output("Duplicates Cluster", Corpus)
        corpus = Output("Corpus", Corpus)

    resizing_enabled = True

    class Error(OWWidget.Error):
        dist_matrix_invalid_shape = Msg('Duplicate detection only supports '
                                        'distances calculated between rows.')
        too_little_documents = Msg('More than one document is required.')

    LINKAGE = ['Single', 'Average', 'Complete', 'Weighted', 'Ward']
    linkage_method = settings.Setting(1)

    threshold = settings.Setting(.0)

    # Cluster variable domain role
    AttributeRole, ClassRole, MetaRole = 0, 1, 2
    CLUSTER_ROLES = ["Attributes", "Class", "Metas"]
    cluster_role = settings.Setting(2)

    def __init__(self):
        super().__init__()
        self.corpus = None              # corpus taken from distances
        self.linkage = None             # hierarchical clustering linkage as returned by Orange
        self.distances = None           # DistMatrix on input
        self.clustering_mask = None     # 1D array of clusters for self.corpus
        self.threshold_spin = None

        # Info
        self.n_documents = ''
        self.n_unique = ''
        self.n_duplicates = ''
        info_box = gui.widgetBox(self.controlArea, box='Info')
        gui.label(info_box, self, 'Documents: %(n_documents)s')
        gui.label(info_box, self, '  ◦ unique: %(n_unique)s')
        gui.label(info_box, self, '  ◦ duplicates: %(n_duplicates)s')

        # Threshold Histogram & Cluster View
        self.histogram = Histogram(self)
        self.table_view = gui.TableView(selectionMode=QListView.SingleSelection)
        self.table_model = PyTableModel()
        self.table_model.setHorizontalHeaderLabels(['Cluster', 'Size'])
        self.table_view.setModel(self.table_model)
        self.table_view.selectionModel().selectionChanged.connect(self.send_duplicates)

        # Add to main area
        height = 300
        main_area = gui.hBox(self.mainArea)
        self.histogram.setMinimumWidth(500)
        self.histogram.setMinimumHeight(height)
        self.table_view.setFixedWidth(140)
        main_area.layout().addWidget(self.histogram)
        main_area.layout().addWidget(self.table_view)

        # Controls
        gui.comboBox(self.controlArea, self, 'linkage_method', items=self.LINKAGE, box='Linkage',
                     callback=self.recalculate_linkage, orientation=Qt.Horizontal)
        self.threshold_spin = gui.doubleSpin(self.controlArea, self, 'threshold',
                                             0, float('inf'), 0.01, decimals=2,
                                             label='Distance threshold', box='Distances',
                                             callback=self.threshold_changed,
                                             keyboardTracking=False, controlWidth=60)
        self.histogram.region.sigRegionChangeFinished.connect(self.threshold_from_histogram_region)
        self.threshold_spin.setEnabled(False)
        gui.rubber(self.controlArea)

        # Output
        gui.comboBox(self.controlArea, self, "cluster_role", box='Output',
                     label='Append Cluster IDs to:', callback=self.send_corpus,
                     items=self.CLUSTER_ROLES)

    def reset(self):
        self.corpus = None
        self.linkage = None
        self.distances = None
        self.clustering_mask = None
        self.n_documents = ''
        self.n_unique = ''
        self.n_duplicates = ''
        self.threshold = 0
        self.threshold_spin.setEnabled(False)
        self.table_model.clear()
        self.histogram.setValues([])

    @Inputs.distances
    def set_distances(self, distances):
        self.Error.clear()
        self.distances = distances
        if distances is None:
            self.reset()
            return

        self.corpus = self.distances.row_items
        self.n_documents = len(self.corpus)
        if self.n_documents < 2:
            self.Error.too_little_documents()
            self.reset()
            return
        if distances.shape != (self.n_documents, self.n_documents):
            self.Error.dist_matrix_invalid_shape()
            self.reset()
            return
        self.threshold_spin.setEnabled(True)
        self.recalculate_linkage()

    def threshold_from_histogram_region(self):
        _, self.threshold = self.histogram.getRegion()
        self.threshold_changed()

    def threshold_changed(self):
        self.threshold = np.clip(self.threshold, *self.histogram.boundary())
        self.histogram.setRegion(0, self.threshold)
        self.detect_duplicates()

    def recalculate_linkage(self):
        if self.distances is not None:
            self.linkage = dist_matrix_linkage(self.distances,
                                               self.LINKAGE[self.linkage_method].lower())

            # Magnitude of the spinbox's step is data-dependent
            vals = sorted(self.linkage[:, 2])
            low, up = vals[0], vals[-1]
            step = (up - low) / 20

            self.threshold_spin.setSingleStep(step)
            self.threshold = np.clip(self.threshold, low, up)
            self.histogram.setValues([])    # without this range breaks when changing linkages
            self.histogram.setValues(vals)
            self.histogram.setRegion(0, self.threshold)

            self.detect_duplicates()

    def detect_duplicates(self):
        if self.distances is not None:
            self.cluster_linkage()
            self.send_corpus()
            self.send_corpus_without_duplicates()
            self.fill_cluster_view()

    def cluster_linkage(self):
        # cluster documents
        n = int(self.n_documents)
        clusters = {j: [j] for j in range(n)}
        for i, (c1, c2, dist, size) in enumerate(self.linkage):
            if dist > self.threshold:
                break
            clusters[n + i] = clusters[c1] + clusters[c2]
            del clusters[c1]
            del clusters[c2]

        self.n_unique = len(clusters)
        self.n_duplicates = n - self.n_unique

        # create mask
        self.clustering_mask = np.empty(n, dtype=int)
        for i, c in enumerate(clusters.values()):
            self.clustering_mask[c] = i

    def fill_cluster_view(self):
        self.table_model.clear()
        c = Counter(self.clustering_mask)
        for id_, count in c.items():
            self.table_model.append([Cluster(id_), count])
        self.table_view.sortByColumn(1, Qt.DescendingOrder)
        self.table_view.selectRow(0)

    def send_corpus(self):
        if self.clustering_mask is not None:
            cluster_var = DiscreteVariable(
                'Duplicates Cluster',
                values=[str(Cluster(v)) for v in set(self.clustering_mask.flatten())]
            )
            corpus, domain = self.corpus, self.corpus.domain
            attrs = domain.attributes
            class_ = domain.class_vars
            metas = domain.metas

            if self.cluster_role == self.AttributeRole:
                attrs = attrs + (cluster_var,)
            elif self.cluster_role == self.ClassRole:
                class_ = class_ + (cluster_var,)
            elif self.cluster_role == self.MetaRole:
                metas = metas + (cluster_var,)

            domain = Domain(attrs, class_, metas)
            corpus = corpus.from_table(domain, corpus)
            corpus.get_column_view(cluster_var)[0][:] = self.clustering_mask
            self.Outputs.corpus.send(corpus)
        else:
            self.Outputs.corpus.send(None)

    def send_corpus_without_duplicates(self):
        if self.clustering_mask is not None:
            # TODO make this more general, currently we just take the first document
            mask = [np.where(self.clustering_mask == i)[0][0]
                    for i in set(self.clustering_mask)]
            c = self.corpus[mask]
            c.name = '{} (Without Duplicates)'.format(self.corpus.name)
            self.Outputs.corpus_without_duplicates.send(c)
        else:
            self.Outputs.corpus_without_duplicates.send(None)

    def send_duplicates(self):
        c = None
        indices = self.table_view.selectionModel().selectedIndexes()
        if indices:
            cluster = self.table_view.model().data(indices[0], Qt.EditRole)
            mask = np.flatnonzero(self.clustering_mask == cluster.id)
            c = self.corpus[mask]
            c.name = '{} {}'.format(self.Outputs.duplicates.name, cluster)
        self.Outputs.duplicates.send(c)


    def send_report(self):
        self.report_items([
            ('Linkage', self.LINKAGE[self.linkage_method]),
            ('Distance threshold', '{:.2f}'.format(self.threshold)),
            ('Documents', self.n_documents),
            ('Unique', self.n_unique),
            ('Duplicates', self.n_duplicates),
        ])


class Cluster:
    def __init__(self, id):
        self.id = id

    def __str__(self):
        return 'C{}'.format(self.id)

    def __lt__(self, other):
        return self.id < other.id

# TODO move the code below to a common place.
# Currently this is a more or less a copy from OWNxFromDistances.py

# Switch to using white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')   # axis lines, numbers etc


pg_InfiniteLine = pg.InfiniteLine


class InfiniteLine(pg_InfiniteLine):
    def paint(self, p, *args):
        # From orange3-bioinformatics:OWFeatureSelection.py, thanks to @ales-erjavec
        brect = self.boundingRect()
        c = brect.center()
        line = QLineF(brect.left(), c.y(), brect.right(), c.y())
        t = p.transform()
        line = t.map(line)
        p.save()
        p.resetTransform()
        p.setPen(self.currentPen)
        p.drawLine(line)
        p.restore()

# Patched so that the Histogram's LinearRegionItem works on MacOS
pg.InfiniteLine = InfiniteLine
pg.graphicsItems.LinearRegionItem.InfiniteLine = InfiniteLine


class Histogram(pg.PlotWidget):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, setAspectLocked=True, **kwargs)
        self.curve = self.plot([0, 1], [0], pen=pg.mkPen('b', width=2), stepMode=True)
        self.region = pg.LinearRegionItem([0, 0], brush=pg.mkBrush('#02f1'), movable=True)
        self.region.sigRegionChanged.connect(self._update_region)
        # Selected region is only open-ended on the the upper side
        self.region.hoverEvent = self.region.mouseDragEvent = lambda *args: None
        self.region.lines[0].setVisible(False)
        self.addItem(self.region)
        self.fillCurve = self.plotItem.plot([0, 1], [0],
            fillLevel=0, pen=pg.mkPen('b', width=2), brush='#02f3', stepMode=True)
        self.plotItem.vb.setMouseEnabled(x=False, y=False)
        self.setValues([])

    def _update_region(self, region):
        rlow, rhigh = self.getRegion()
        low = max(0, np.searchsorted(self.xData, rlow, side='right') - 1)
        high = np.searchsorted(self.xData, rhigh, side='right')
        if high - low > 0:
            xData = self.xData[low:high + 1].copy()
            xData[0] = rlow  # set visible boundaries to match region lines
            xData[-1] = rhigh
            self.fillCurve.setData(xData, self.yData[low:high])

    def setBoundary(self, low, high):
        self.region.setBounds((low, high))

    def boundary(self):
        return self.xData[[0, -1]]

    def setRegion(self, low, high):
        low, high = np.clip([low, high], *self.boundary())
        self.region.setRegion((low, high))

    def getRegion(self):
        return self.region.getRegion()

    def setValues(self, values):
        self.fillCurve.setData([0, 1], [0])
        if not len(values):
            self.curve.setData([0, 1], [0])
            self.setBoundary(0, 0)
            self.autoRange()
            return
        nbins = min(len(values), 100)
        freq, edges = np.histogram(values, bins=nbins)
        self.curve.setData(edges, freq)
        self.setBoundary(edges[0], edges[-1])
        self.autoRange()

    @property
    def xData(self):
        return self.curve.xData

    @property
    def yData(self):
        return self.curve.yData


if __name__ == "__main__":
    from Orange.distance import Euclidean
    appl = QApplication([])
    data = Table('iris')
    dm = Euclidean(data)
    ow = OWDuplicates()
    ow.set_distances(dm)
    ow.show()
    appl.exec_()
