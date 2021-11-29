import time
from typing import Optional, List, Tuple, Any
import numpy as np
from networkx import Graph, minimum_spanning_tree
from networkx.convert_matrix import from_numpy_array
from networkx.relabel import relabel_nodes
from networkx.algorithms.centrality import voterank
from sentence_transformers import SentenceTransformer

from AnyQt.QtWidgets import QTreeWidget, QTreeWidgetItem
from AnyQt.QtCore import Qt

from Orange.data import Table, StringVariable, Domain
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from Orange.widgets.widget import OWWidget
from Orange.widgets import gui
from orangewidget.utils.signals import Input
from orangewidget.widget import Msg


def _generate_ontology(words: List[str]) -> Tuple[Graph, str]:
    """
    Ontology generator -- computes MST on the complete graph of words.
    """
    t = time.time()
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embs = np.array(model.encode(words))
    norms = np.linalg.norm(embs, axis=1).reshape(-1, 1)
    embs = embs / norms
    dists = 1 - np.dot(embs, embs.T)
    graph = from_numpy_array(dists, parallel_edges=False)
    mapping = {i: words[i] for i in range(len(words))}
    graph = relabel_nodes(graph, mapping)
    mst = minimum_spanning_tree(graph)
    root = voterank(graph, 1)[0]
    return mst, root


def to_tree(graph: Graph, root: str, prev: str = None):
    neighbors = [n for n in graph.neighbors(root) if n != prev]
    tree = {}
    for neighbor in neighbors:
        tree[neighbor] = to_tree(graph, neighbor, root)
    return tree


def _run(words: List[str], _):
    if len(words) > 1:
        words = [w.lower() for w in words]
        mst, root = _generate_ontology(words)
        return {root: to_tree(mst, root)}
    elif len(words) == 1:
        return {words[0]: {}}
    else:
        return {}


class OWOntology(OWWidget, ConcurrentWidgetMixin):
    name = "Ontology"
    description = ""
    keywords = []

    want_control_area = False

    class Inputs:
        words = Input("Words", Table)

    class Warning(OWWidget.Warning):
        no_string_vars = Msg("Input needs at least one Text variable.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.__words = None

        box = gui.vBox(self.mainArea, "Projects")

        self.tree = QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.setColumnWidth(0, 500)
        self.tree.setHeaderLabels(("Ontology",))
        box.layout().addWidget(self.tree)

    @Inputs.words
    def set_words(self, words: Optional[Table]):
        self._check_input_words(words)
        self.start_plot()

    def _check_input_words(self, words_table):
        self.Warning.no_string_vars.clear()
        if words_table:
            metas = words_table.domain.metas
            word_vars = (m for m in metas if isinstance(m, StringVariable))
            if not word_vars:
                self.Warning.no_string_vars()
                self.__words = None
            else:
                words_var = next(word_vars)
                self.__words = words_table.get_column_view(words_var)[0]

    def start_plot(self):
        if self.__words is not None:
            self.start(_run, self.__words)

    def on_done(self, tree):
        self.tree.clear()
        self.set_tree(tree, self.tree)

    def on_exception(self, ex: Exception):
        raise ex

    def on_partial_result(self, result: Any) -> None:
        pass

    def set_tree(self, data, parent):
        for key, value in data.items():
            node = QTreeWidgetItem(parent, [key])
            node.name = key
            if len(value) > 0:
                node.setExpanded(True)
                node.setFlags(node.flags() | Qt.ItemIsAutoTristate)
                # s = Qt.Checked if self.projects is None else Qt.Unchecked
                # Collapse indicators can not be hidden because of a QT bug
                # https://bugreports.qt.io/browse/QTBUG-59354
                # node.setChildIndicatorPolicy(node.DontShowIndicator)
                self.set_tree(value, node)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    words_vars = [StringVariable("S1")]
    words = [
        "Stvar",
        "Agent",
        "Organ",
        "Državni organ",
        "Drug državni organ",
        "Organ državne uprave",
        "Organ lokalne skupnosti",
        "Organ občine",
        "Organ upravljanja",
        "Registrski organ",
        "Nadzorni organ",
        "Upravni organ",
        "Ministrstvo",
        "Organ v sestavi ministrstva",
        "Upravna enota",
        "Bančni račun",
        "Transakcijski račun",
        "Delež",
        "Delež v družbi",
        "Lastniški delež",
        "Dovoljenje",
        "Dražba",
        "Izplačilo",
        "Plača",
        "Pravni akt",
        "Odločba",
        "Sklep",
    ]
    input_table = Table.from_list(Domain([], metas=words_vars), [[w] for w in words])

    WidgetPreview(OWOntology).run(input_table)
