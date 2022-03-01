from typing import Optional, List, Tuple, Any, Dict

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QTreeWidget, QTreeWidgetItem
from networkx import Graph, minimum_spanning_tree
from networkx.algorithms.centrality import voterank
from networkx.convert_matrix import from_numpy_array
from networkx.relabel import relabel_nodes
from sentence_transformers import SentenceTransformer

from Orange.data import Table, StringVariable, Domain
from Orange.widgets import gui
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import OWWidget, Msg, Input

WORDS_COLUMN_NAME = "Words"


def _generate_ontology(words: List[str]) -> Tuple[Graph, str]:
    """
    Ontology generator -- computes MST on the complete graph of words.
    """
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


def to_tree(graph: Graph, root: str, prev: str = None) -> Dict:
    neighbors = [n for n in graph.neighbors(root) if n != prev]
    tree = {}
    for neighbor in neighbors:
        tree[neighbor] = to_tree(graph, neighbor, root)
    return tree


def _run(words: List[str], state: TaskState) -> Dict:
    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    callback(0, "Calculating...")

    if len(words) > 1:
        words = [w.lower() for w in words]
        mst, root = _generate_ontology(words)
        return {root: to_tree(mst, root)}
    elif len(words) == 1:
        return {words[0].lower(): {}}
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
        no_words_column = Msg("Input is missing 'Words' column.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        box = gui.vBox(self.mainArea)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.setColumnWidth(0, 500)
        self.tree.setHeaderLabels(("Ontology",))
        box.layout().addWidget(self.tree)

    @Inputs.words
    def set_words(self, words: Optional[Table]):
        self.Warning.no_words_column.clear()
        if words:
            if WORDS_COLUMN_NAME in words.domain and words.domain[
                    WORDS_COLUMN_NAME].attributes.get("type") == "words":
                words = list(words.get_column_view(WORDS_COLUMN_NAME)[0])
                self.start(_run, words)
            else:
                self.Warning.no_words_column()

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

    lst = [
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
    words_var = StringVariable("Words")
    words_var.attributes = {"type": "words"}
    words_ = Table.from_list(Domain([], metas=[words_var]), [[w] for w in lst])
    words_.name = "Words"
    WidgetPreview(OWOntology).run(words_)
