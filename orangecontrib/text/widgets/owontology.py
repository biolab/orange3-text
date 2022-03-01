from typing import Optional, List, Tuple, Any, Dict, Union

import numpy as np
from AnyQt.QtCore import Qt, QModelIndex
from AnyQt.QtGui import QDropEvent, QStandardItemModel, QStandardItem
from AnyQt.QtWidgets import QWidget, QAction, QVBoxLayout, QTreeView
from networkx import Graph, minimum_spanning_tree
from networkx.algorithms.centrality import voterank
from networkx.convert_matrix import from_numpy_array
from networkx.relabel import relabel_nodes
from sentence_transformers import SentenceTransformer

from Orange.data import Table, StringVariable, Domain
from Orange.widgets import gui
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import OWWidget, Msg, Input
from orangewidget.utils.itemmodels import ModelActionsWidget

WORDS_COLUMN_NAME = "Words"
resources_path = os.path.join(os.path.dirname(__file__), "resources")


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


def _graph_to_tree(graph: Graph, root: str, prev: str = None) -> Dict:
    neighbors = [n for n in graph.neighbors(root) if n != prev]
    tree = {}
    for neighbor in neighbors:
        tree[neighbor] = _graph_to_tree(graph, neighbor, root)
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
        return {root: _graph_to_tree(mst, root)}
    elif len(words) == 1:
        return {words[0].lower(): {}}
    else:
        return {}


def _model_to_tree(item: QStandardItem) -> Dict:
    tree = {}
    for i in range(item.rowCount()):
        tree[item.child(i).text()] = _model_to_tree(item.child(i))
    return tree


def _tree_to_model(words: Dict, root: QStandardItem):
    if isinstance(words, dict):
        for word, words in words.items():
            item = QStandardItem(word)
            root.appendRow(item)
            if len(words):
                _tree_to_model(words, item)


class TreeView(QTreeView):
    Style = f"""
    QTreeView::branch {{
        background: palette(base);
    }}
    QTreeView::branch:has-siblings:!adjoins-item {{
        border-image: url({resources_path}/vline.png) 0;
    }}

    QTreeView::branch:has-siblings:adjoins-item {{
        border-image: url({resources_path}/branch-more.png) 0;
    }}

    QTreeView::branch:!has-children:!has-siblings:adjoins-item {{
        border-image: url({resources_path}/branch-end.png) 0;
    }}

    QTreeView::branch:has-children:!has-siblings:closed,
    QTreeView::branch:closed:has-children:has-siblings {{
            border-image: none;
            image: url({resources_path}/branch-closed.png);
    }}

    QTreeView::branch:open:has-children:!has-siblings,
    QTreeView::branch:open:has-children:has-siblings  {{
            border-image: none;
            image: url({resources_path}/branch-open.png);
    }}
    """
    drop_finished = Signal()

    def __init__(self):
        edit_triggers = QTreeView.DoubleClicked | QTreeView.EditKeyPressed
        super().__init__(
            editTriggers=int(edit_triggers),
            selectionMode=QTreeView.SingleSelection,
            dragEnabled=True,
            acceptDrops=True,
            defaultDropAction=Qt.MoveAction
        )
        self.setHeaderHidden(True)
        self.setDropIndicatorShown(True)
        self.setStyleSheet(self.Style)

    def startDrag(self, actions: Qt.DropActions):
        super().startDrag(actions)
        self.drop_finished.emit()

    def dropEvent(self, event: QDropEvent):
        super().dropEvent(event)
        self.expandAll()


class EditableTreeView(QWidget):
    dataChanged = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.__data_orig: Dict = {}

        self.__model = QStandardItemModel()
        self.__model.dataChanged.connect(self.__on_data_changed)
        self.__root = self.__model.invisibleRootItem()

        self.__tree = TreeView()
        self.__tree.setModel(self.__model)
        self.__tree.drop_finished.connect(self.__on_data_changed)

        actions_widget = ModelActionsWidget()
        actions_widget.layout().setSpacing(1)

        action = QAction("+", self, toolTip="Add a new word")
        action.triggered.connect(self.__on_add)
        actions_widget.addAction(action)

        action = QAction("\N{MINUS SIGN}", self, toolTip="Remove word")
        action.triggered.connect(self.__on_remove)
        actions_widget.addAction(action)

        action = QAction("\N{MINUS SIGN}R", self,
                         toolTip="Remove word recursively (incl. children)")
        action.triggered.connect(self.__on_remove_all)
        actions_widget.addAction(action)

        gui.rubber(actions_widget)

        action = QAction("Reset", self, toolTip="Reset to original data")
        action.triggered.connect(self.__on_reset)
        actions_widget.addAction(action)

        layout = QVBoxLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.__tree)
        layout.addWidget(actions_widget)
        self.setLayout(layout)

    def __on_data_changed(self):
        self.dataChanged.emit(self.get_data())

    def __on_add(self):
        parent: QStandardItem = self.__root
        selection: List = self.__tree.selectionModel().selectedIndexes()
        if selection:
            sel_index: QModelIndex = selection[0]
            parent: QStandardItem = self.__model.itemFromIndex(sel_index)

        item = QStandardItem("")
        parent.appendRow(item)
        index: QModelIndex = item.index()
        self.__model.setItemData(index, {Qt.EditRole: ""})
        self.__tree.setCurrentIndex(index)
        self.__tree.edit(index)

    def __on_remove_all(self):
        selection: List = self.__tree.selectionModel().selectedIndexes()
        if len(selection):
            index: QModelIndex = selection[0]
            self.__model.removeRow(index.row(), index.parent())

    def __on_remove(self):
        selection: List = self.__tree.selectionModel().selectedIndexes()
        if len(selection):
            index: QModelIndex = selection[0]

            # move children to item's parent
            item: QStandardItem = self.__model.itemFromIndex(index)
            for i in range(item.rowCount()):
                child: QStandardItem = item.takeChild(i)
                (item.parent() or self.__root).appendRow(child)

            self.__model.removeRow(index.row(), index.parent())

    def __on_reset(self):
        self.set_data(self.__data_orig)

    def get_data(self) -> Dict:
        return _model_to_tree(self.__root)

    def set_data(self, data: Dict):
        self.__data_orig = data
        self.clear()
        _tree_to_model(data, self.__root)
        self.__tree.expandAll()

    def clear(self):
        if self.__model.hasChildren():
            self.__model.removeRows(0, self.__model.rowCount())


class OWOntology(OWWidget, ConcurrentWidgetMixin):
    name = "Ontology"
    description = ""
    keywords = []

    want_main_area = False

    class Inputs:
        words = Input("Words", Table)

    class Warning(OWWidget.Warning):
        no_words_column = Msg("Input is missing 'Words' column.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        box = gui.hBox(self.mainArea)
        self.__ontology_view = EditableTreeView(self)
        box.layout().addWidget(self.__ontology_view)

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

    def on_done(self, data: Dict):
        self.__ontology_view.set_data(data)

    def on_exception(self, ex: Exception):
        raise ex

    def on_partial_result(self, _: Any):
        pass


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
