import os
import pickle
from contextlib import contextmanager
from typing import Optional, List, Tuple, Any, Dict, Callable

import numpy as np
from AnyQt.QtCore import Qt, QModelIndex, QItemSelection, Signal, \
    QItemSelectionModel
from AnyQt.QtGui import QDropEvent, QStandardItemModel, QStandardItem, \
    QPainter, QColor, QPalette
from AnyQt.QtWidgets import QWidget, QAction, QVBoxLayout, QTreeView, QMenu, \
    QToolButton, QGroupBox, QListView, QSizePolicy, QStyledItemDelegate, \
    QStyleOptionViewItem, QLineEdit, QFileDialog, QApplication
from networkx import Graph, minimum_spanning_tree
from networkx.algorithms.centrality import voterank
from networkx.convert_matrix import from_numpy_array
from networkx.relabel import relabel_nodes
from sentence_transformers import SentenceTransformer

from Orange.data import Table, StringVariable, Domain
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import ModelActionsWidget, PyListModel
from Orange.widgets.widget import OWWidget, Msg, Input

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


def _model_to_words(item: QStandardItem) -> List:
    words = [item.text()] if item.text() else []
    for i in range(item.rowCount()):
        words.extend(_model_to_words(item.child(i)))
    return words


def _tree_to_model(tree: Dict, root: QStandardItem):
    for word, words in tree.items():
        item = QStandardItem(word)
        root.appendRow(item)
        if len(words):
            _tree_to_model(words, item)


def _tree_to_html(tree: Dict) -> str:
    if not tree:
        return ""

    html = "<ul>"
    for k, v in tree.items():
        html += f"<li>{k}{_tree_to_html(v)}</li>"
    html += "</ul>"

    return html


@contextmanager
def disconnected(signal, slot, connection_type=Qt.AutoConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, connection_type)


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

    def __init__(self, data_changed_cb: Callable):
        self.__data_changed_cb = data_changed_cb

        edit_triggers = QTreeView.DoubleClicked | QTreeView.EditKeyPressed
        super().__init__(
            editTriggers=int(edit_triggers),
            selectionMode=QTreeView.ExtendedSelection,
            dragEnabled=True,
            acceptDrops=True,
            defaultDropAction=Qt.MoveAction
        )
        self.setHeaderHidden(True)
        self.setDropIndicatorShown(True)
        self.setStyleSheet(self.Style)

    def startDrag(self, actions: Qt.DropActions):
        with disconnected(self.model().dataChanged, self.__data_changed_cb):
            super().startDrag(actions)
        self.drop_finished.emit()

    def dropEvent(self, event: QDropEvent):
        super().dropEvent(event)
        self.expandAll()


class EditableTreeView(QWidget):
    dataChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.__data_orig: Dict = {}

        self.__model = QStandardItemModel()
        self.__model.dataChanged.connect(self.dataChanged)
        self.__root = self.__model.invisibleRootItem()

        self.__tree = TreeView(self.dataChanged)
        self.__tree.drop_finished.connect(self.dataChanged)
        self.__tree.setModel(self.__model)

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
        action.triggered.connect(self.__on_remove_recursive)
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

    def __on_add(self):
        parent: QStandardItem = self.__root
        selection: List = self.__tree.selectionModel().selectedIndexes()
        if selection:
            sel_index: QModelIndex = selection[0]
            parent: QStandardItem = self.__model.itemFromIndex(sel_index)

        item = QStandardItem("")
        parent.appendRow(item)
        index: QModelIndex = item.index()
        with disconnected(self.__model.dataChanged, self.dataChanged):
            self.__model.setItemData(index, {Qt.EditRole: ""})
        self.__tree.setCurrentIndex(index)
        self.__tree.edit(index)

    def __on_remove_recursive(self):
        sel_model: QItemSelectionModel = self.__tree.selectionModel()
        if len(sel_model.selectedIndexes()):
            while sel_model.selectedIndexes():
                index: QModelIndex = sel_model.selectedIndexes()[0]
                self.__model.removeRow(index.row(), index.parent())
            self.dataChanged.emit()

    def __on_remove(self):
        sel_model: QItemSelectionModel = self.__tree.selectionModel()
        if len(sel_model.selectedIndexes()):

            while sel_model.selectedIndexes():
                index: QModelIndex = sel_model.selectedIndexes()[0]

                # move children to item's parent
                item: QStandardItem = self.__model.itemFromIndex(index)
                children = [item.takeChild(i) for i in range(item.rowCount())]
                parent = item.parent() or self.__root

                self.__model.removeRow(index.row(), index.parent())

                for child in children[::-1]:
                    parent.insertRow(index.row(), child)

            self.__tree.expandAll()
            self.dataChanged.emit()

    def __on_reset(self):
        self.set_data(self.__data_orig)
        self.dataChanged.emit()

    def get_words(self) -> List:
        return _model_to_words(self.__root)

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


class Ontology:
    NotModified, Modified = range(2)

    def __init__(
            self,
            name: str,
            ontology: Dict,
            filename: Optional[str] = None
    ):
        self.name = name
        self.filename = filename
        self.word_tree = dict(ontology)  # library ontology
        self.cached_word_tree = dict(ontology)  # current ontology
        self.update_rule_flag = Ontology.NotModified

    @property
    def flags(self) -> int:
        # 0 - NotModified, 1 - Modified
        return int(self.word_tree != self.cached_word_tree or
                   self.update_rule_flag == Ontology.Modified)

    def as_dict(self) -> Dict:
        return {"name": self.name,
                "ontology": dict(self.word_tree),
                "filename": self.filename}

    @classmethod
    def from_dict(cls, state: Dict) -> "WordList":
        return Ontology(state["name"],
                        dict(state["ontology"]),
                        filename=state.get("filename"))

    @staticmethod
    def generate_name(taken_names: List[str]) -> str:
        default_name = "Untitled"
        indices = {0}
        for name in taken_names:
            if name.startswith(default_name):
                try:
                    indices.add(int(name[len(default_name):]))
                except ValueError:
                    pass
        index = min(set(range(max(indices) + 1 + 1)) - indices)
        return f"{default_name} {index}"


def read_from_file(parent: OWWidget) -> Optional[Ontology]:
    filename, _ = QFileDialog.getOpenFileName(
        parent, "Open Ontology",
        os.path.expanduser("~/"),
        "Ontology files (*.owl)"
    )
    if not filename:
        return None

    name = os.path.basename(filename)
    with open(filename, "rb") as f:
        data = pickle.load(f)

    assert isinstance(data, dict)
    return Ontology(name, data, filename=filename)


def save_ontology(parent: OWWidget, filename: str, data: Dict):
    filename, _ = QFileDialog.getSaveFileName(
        parent, "Save Ontology", filename,
        "Ontology files (*.owl)"
    )
    if filename:
        head, tail = os.path.splitext(filename)
        if not tail:
            filename = head + ".owl"

        assert isinstance(data, dict)
        with open(filename, "wb") as f:
            pickle.dump(data, f)


class LibraryItemDelegate(QStyledItemDelegate):
    @staticmethod
    def displayText(ontology: Ontology, _) -> str:
        return "*" + ontology.name if ontology.flags & Ontology.Modified \
            else ontology.name

    def paint(
            self,
            painter: QPainter,
            option: QStyleOptionViewItem,
            index: QModelIndex
    ):
        word_list = index.data(Qt.DisplayRole)
        if word_list.flags & Ontology.Modified:
            option = QStyleOptionViewItem(option)
            option.palette.setColor(QPalette.Text, QColor(Qt.red))
            option.palette.setColor(QPalette.Highlight, QColor(Qt.darkRed))
            option.palette.setColor(QPalette.HighlightedText, QColor(Qt.white))
        super().paint(painter, option, index)

    @staticmethod
    def createEditor(parent: QWidget, _, __) -> QLineEdit:
        return QLineEdit(parent)

    @staticmethod
    def setEditorData(editor: QLineEdit, index: QModelIndex):
        word_list = index.data(Qt.DisplayRole)
        editor.setText(word_list.name)

    @staticmethod
    def setModelData(editor: QLineEdit, model: PyListModel, index: QModelIndex):
        model[index.row()].name = str(editor.text())


class OWOntology(OWWidget, ConcurrentWidgetMixin):
    name = "Ontology"
    description = ""
    icon = "icons/Ontology.svg"
    priority = 1110
    keywords = []

    CACHED, LIBRARY = range(2)  # library list modification types

    settingsHandler = DomainContextHandler()
    ontology_library: List[Dict] = Setting([
        {"name": Ontology.generate_name([]), "ontology": {}},
    ])
    ontology_index: int = Setting(0)
    ontology: Dict = Setting({}, schema_only=True)

    class Inputs:
        words = Input("Words", Table)

    class Warning(OWWidget.Warning):
        no_words_column = Msg("Input is missing 'Words' column.")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        self.__model = PyListModel([], self, flags=flags)
        self.__library_view: QListView = None
        self.__ontology_view: EditableTreeView = None

        self._setup_gui()
        self._restore_state()
        self.settingsAboutToBePacked.connect(self._save_state)

    def _setup_gui(self):
        # control area
        library_box: QGroupBox = gui.vBox(self.controlArea, "Library")

        edit_triggers = QListView.DoubleClicked | QListView.EditKeyPressed
        self.__library_view = QListView(
            editTriggers=int(edit_triggers),
            minimumWidth=200,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding),
        )
        self.__library_view.setItemDelegate(LibraryItemDelegate(self))
        self.__library_view.setModel(self.__model)
        self.__library_view.selectionModel().selectionChanged.connect(
            self.__on_selection_changed
        )

        actions_widget = ModelActionsWidget()
        actions_widget.layout().setSpacing(1)

        tool_tip = "Add a new ontology to the library"
        action = QAction("+", self, toolTip=tool_tip)
        action.triggered.connect(self.__on_add)
        actions_widget.addAction(action)

        tool_tip = "Remove the ontology from the library"
        action = QAction("\N{MINUS SIGN}", self, toolTip=tool_tip)
        action.triggered.connect(self.__on_remove)
        actions_widget.addAction(action)

        tool_tip = "Save changes in the editor to the library"
        action = QAction("Update", self, toolTip=tool_tip)
        action.triggered.connect(self.__on_update)
        actions_widget.addAction(action)

        gui.rubber(actions_widget)

        action = QAction("More", self, toolTip="More actions")

        new_from_file = QAction("Import Ontology from File", self)
        new_from_file.triggered.connect(self.__on_import_file)

        new_from_url = QAction("Import Ontology from URL", self)
        new_from_url.triggered.connect(self.__on_import_url)

        save_to_file = QAction("Save Ontology to File", self)
        save_to_file.triggered.connect(self.__on_save)

        menu = QMenu(actions_widget)
        menu.addAction(new_from_file)
        # menu.addAction(new_from_url)
        menu.addAction(save_to_file)
        action.setMenu(menu)
        button = actions_widget.addAction(action)
        button.setPopupMode(QToolButton.InstantPopup)

        vlayout = QVBoxLayout()
        vlayout.setSpacing(1)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.addWidget(self.__library_view)
        vlayout.addWidget(actions_widget)

        library_box.layout().setSpacing(1)
        library_box.layout().addLayout(vlayout)

        self.__run_button = gui.button(
            self.controlArea, self, "Run", callback=self.__on_toggle_run
        )

        # main area
        ontology_box: QGroupBox = gui.vBox(self.mainArea, box=True)

        self.__ontology_view = EditableTreeView(self)
        self.__ontology_view.dataChanged.connect(
            self.__on_ontology_data_changed
        )

        ontology_box.layout().setSpacing(1)
        ontology_box.layout().addWidget(self.__ontology_view)

    def __on_selection_changed(self, selection: QItemSelection, *_):
        if selection.indexes():
            self.ontology_index = row = selection.indexes()[0].row()
            data = self.__model[row].cached_word_tree
            self.__ontology_view.set_data(data)

    def __on_add(self):
        name = Ontology.generate_name([l.name for l in self.__model])
        data = self.__ontology_view.get_data()
        self.__model.append(Ontology(name, data))
        self.__set_selected_row(len(self.__model) - 1)

    def __on_remove(self):
        index = self.__get_selected_row()
        if index is not None:
            del self.__model[index]
            self.__set_selected_row(max(index - 1, 0))

    def __on_update(self):
        self.__set_current_modified(self.LIBRARY)

    def __on_import_file(self):
        ontology = read_from_file(self)
        if ontology is not None:
            self.__model.append(ontology)
            self.__set_selected_row(len(self.__model) - 1)
        QApplication.setActiveWindow(self)

    def __on_import_url(self):
        pass

    def __on_save(self):
        index = self.__get_selected_row()
        if index is not None:
            filename = self.__model[index].filename
        else:
            filename = os.path.expanduser("~/")
        save_ontology(self, filename, self.__ontology_view.get_data())
        QApplication.setActiveWindow(self)

    def __on_toggle_run(self):
        if self.task is not None:
            self.cancel()
            self.__run_button.setText("Run")
        else:
            self._run()

    def __on_ontology_data_changed(self):
        self.__set_current_modified(self.CACHED)

    @Inputs.words
    def set_words(self, words: Optional[Table]):
        self.Warning.no_words_column.clear()
        if words:
            if WORDS_COLUMN_NAME in words.domain and words.domain[
                    WORDS_COLUMN_NAME].attributes.get("type") == "words":
                col = words.get_column_view(WORDS_COLUMN_NAME)[0]
                words = dict(zip(col, np.full(len(col), {})))
                self.__ontology_view.set_data(words)
            else:
                self.Warning.no_words_column()

    def _run(self):
        self.__run_button.setText("Stop")
        self.start(_run, self.__ontology_view.get_words())

    def on_done(self, data: Dict):
        self.__run_button.setText("Run")
        self.__ontology_view.set_data(data)
        self.__set_current_modified(self.CACHED)

    def on_exception(self, ex: Exception):
        raise ex

    def on_partial_result(self, _: Any):
        pass

    def onDeleteWidget(self):
        self.shutdown()
        super().onDeleteWidget()

    def __set_selected_row(self, row: int):
        self.__library_view.selectionModel().select(
            self.__model.index(row, 0), QItemSelectionModel.ClearAndSelect
        )

    def __get_selected_row(self) -> Optional[int]:
        rows = self.__library_view.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def __set_current_modified(self, mod_type: int):
        index = self.__get_selected_row()
        if index is not None:
            if mod_type == self.LIBRARY:
                ontology = self.__ontology_view.get_data()
                self.__model[index].word_tree = ontology
                self.__model[index].cached_word_tree = ontology
                self.__model[index].update_rule_flag = Ontology.NotModified
            elif mod_type == self.CACHED:
                ontology = self.__ontology_view.get_data()
                self.__model[index].cached_word_tree = ontology
            else:
                raise NotImplementedError
            self.__model.emitDataChanged(index)
            self.__library_view.repaint()

    def _restore_state(self):
        source = [Ontology.from_dict(s) for s in self.ontology_library]
        self.__model.wrap(source)
        self.__set_selected_row(self.ontology_index)
        if self.ontology:
            self.__ontology_view.set_data(self.ontology)
            self.__set_current_modified(self.CACHED)

    def _save_state(self):
        self.ontology_library = [s.as_dict() for s in self.__model]
        self.ontology = self.__ontology_view.get_data()

    def send_report(self):
        model = self.__model
        library = model[self.ontology_index].name if model else "/"
        self.report_items("Settings", [("Library", library)])

        ontology = self.__ontology_view.get_data()
        style = """
        <style>
            ul {
                padding-top: 0px;
                padding-right: 0px;
                padding-bottom: 0px;
                padding-left: 20px;
            }
        </style>
        """
        self.report_raw("Ontology", style + _tree_to_html(ontology))


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
