import json
import os
import pickle
import tempfile
from contextlib import contextmanager
from typing import Optional, List, Tuple, Any, Dict, Callable, Set, Union
from urllib.parse import urlparse
import requests

from AnyQt.QtCore import Qt, QModelIndex, QItemSelection, Signal, \
    QItemSelectionModel
from AnyQt.QtGui import QDropEvent, QStandardItemModel, QStandardItem, \
    QPainter, QColor, QPalette, QDragEnterEvent, QDragLeaveEvent
from AnyQt.QtWidgets import QWidget, QAction, QVBoxLayout, QTreeView, QMenu, \
    QToolButton, QGroupBox, QListView, QSizePolicy, QStyledItemDelegate, \
    QStyleOptionViewItem, QLineEdit, QFileDialog, QApplication, QDialog, \
    QDialogButtonBox, QHBoxLayout, QLabel
from owlready2 import Thing, World, OwlReadyOntologyParsingError

from orangewidget.utils.listview import ListViewSearch

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler, Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import ModelActionsWidget, PyListModel
from Orange.widgets.widget import OWWidget, Msg, Input, Output

from orangecontrib.text.ontology import OntologyHandler
from orangecontrib.text.widgets.utils.words import create_words_table, \
    WORDS_COLUMN_NAME

OntoType = Tuple[Dict[str, "OntoType"], bool]
resources_path = os.path.join(os.path.dirname(__file__), "resources")


def _run(handler: Callable, args: Tuple, state: TaskState) -> Dict:
    def callback(i: float, status=""):
        state.set_progress_value(i * 100)
        if status:
            state.set_status(status)
        if state.is_interruption_requested():
            raise Exception

    callback(0, "Calculating...")
    return handler(*args, callback=callback)


def _model_to_tree(
        item: QStandardItem,
        selection: List,
        with_selection: bool
) -> Union[Dict, OntoType]:
    tree = {}
    for i in range(item.rowCount()):
        tree[item.child(i).text()] = \
            _model_to_tree(item.child(i), selection, with_selection)
    return (tree, item.index() in selection) if with_selection else tree


def _model_to_words(item: QStandardItem) -> List:
    words = [item.text()] if item.text() else []
    for i in range(item.rowCount()):
        words.extend(_model_to_words(item.child(i)))
    return words


def _tree_to_model(
        tree: Dict,
        root: QStandardItem,
        sel_model: QItemSelectionModel
) -> None:
    # tuple of subtree and selection flag
    if isinstance(tree, tuple):
        tree, _ = tree

    # read from .json
    if isinstance(tree, list):
        tree = {t: {} for t in tree}

    for word, words in tree.items():
        item = QStandardItem(word)
        root.appendRow(item)
        if isinstance(words, tuple):
            _, selected = words
            if selected:
                sel_model.select(item.index(), QItemSelectionModel.Select)
        if len(words):
            _tree_to_model(words, item, sel_model)


def _tree_to_html(tree: Dict) -> str:
    if not tree:
        return ""

    html = "<ul>"
    for k, v in tree.items():
        html += f"<li>{k}{_tree_to_html(v)}</li>"
    html += "</ul>"

    return html


def _onto_to_tree(thing: Thing, world: World) -> Dict:
    tree = {}
    for cl in list(thing.subclasses(world=world)):
        tree[cl.name] = _onto_to_tree(cl, world)
    return tree


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
            editTriggers=edit_triggers,
            selectionMode=QTreeView.ExtendedSelection,
            dragEnabled=True,
            acceptDrops=True,
            defaultDropAction=Qt.MoveAction
        )
        self.setHeaderHidden(True)
        self.setDropIndicatorShown(True)
        self.setStyleSheet(self.Style)

        self.__disconnected = False

    def startDrag(self, actions: Qt.DropAction):
        with disconnected(self.model().dataChanged, self.__data_changed_cb):
            super().startDrag(actions)
        self.drop_finished.emit()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.source() != self:
            self.__disconnected = True
            self.model().dataChanged.disconnect(self.__data_changed_cb)
        super().dragEnterEvent(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        super().dragLeaveEvent(event)
        if self.__disconnected:
            self.__disconnected = False
            self.model().dataChanged.connect(self.__data_changed_cb)

    def dropEvent(self, event: QDropEvent):
        super().dropEvent(event)
        self.expandAll()
        if self.__disconnected:
            self.__disconnected = False
            self.model().dataChanged.connect(self.__data_changed_cb)
            self.drop_finished.emit()


class EditableTreeView(QWidget):
    dataChanged = Signal()
    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.__stack: List = []
        self.__stack_index: int = -1

        def push_on_data_changed(_, __, roles):
            if Qt.EditRole in roles:
                self._push_data()

        self.__model = QStandardItemModel()
        self.__model.dataChanged.connect(self.dataChanged)
        self.__model.dataChanged.connect(push_on_data_changed)
        self.__root: QStandardItem = self.__model.invisibleRootItem()

        self.__tree = TreeView(self.dataChanged)
        self.__tree.drop_finished.connect(self.dataChanged)
        self.__tree.drop_finished.connect(self._push_data)
        self.__tree.setModel(self.__model)
        self.__tree.selectionModel().selectionChanged.connect(
            self.selectionChanged)

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

        self.__undo_action = action = QAction("Undo", self, toolTip="Undo")
        action.triggered.connect(self.__on_undo)
        actions_widget.addAction(action)

        self.__redo_action = action = QAction("Redo", self, toolTip="Redo")
        action.triggered.connect(self.__on_redo)
        actions_widget.addAction(action)

        self._enable_undo_redo()

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
            self._push_data()
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
            self._push_data()
            self.dataChanged.emit()

    def __on_undo(self):
        self.__stack_index -= 1
        self._set_from_stack()

    def __on_redo(self):
        self.__stack_index += 1
        self._set_from_stack()

    def get_words(self) -> List:
        return _model_to_words(self.__root)

    def get_selected_words(self) -> Set:
        return set(self.__model.itemFromIndex(index).text() for index in
                   self.__tree.selectionModel().selectedIndexes())

    def get_selected_words_with_children(self) -> Set:
        words = set()
        for index in self.__tree.selectionModel().selectedIndexes():
            item: QStandardItem = self.__model.itemFromIndex(index)
            words.update(_model_to_words(item))
        return words

    def get_data(self, with_selection=False) -> Union[Dict, OntoType]:
        selection = self.__tree.selectionModel().selectedIndexes()
        return _model_to_tree(self.__root, selection, with_selection)

    def set_data(self, data: Dict, keep_history: bool = False):
        if not keep_history:
            self.__stack = []
            self.__stack_index = -1
        self._set_data(data)
        self._push_data()

    def _set_data(self, data: Dict):
        self.clear()
        _tree_to_model(data, self.__root, self.__tree.selectionModel())
        self.__tree.expandAll()

    def clear(self):
        if self.__model.hasChildren():
            self.__model.removeRows(0, self.__model.rowCount())

    def _enable_undo_redo(self):
        index = self.__stack_index
        self.__undo_action.setEnabled(index >= 1)
        self.__redo_action.setEnabled(index < len(self.__stack) - 1)

    def _push_data(self):
        self.__stack_index += 1
        self.__stack = self.__stack[:self.__stack_index]
        self.__stack.append(self.get_data())
        self._enable_undo_redo()

    def _set_from_stack(self):
        assert self.__stack_index < len(self.__stack)
        assert self.__stack_index >= 0
        self._set_data(self.__stack[self.__stack_index])
        self._enable_undo_redo()
        self.dataChanged.emit()


class Ontology:
    NotModified, Modified = range(2)

    def __init__(
            self,
            name: str,
            ontology: Dict,
            filename: Optional[str] = None,
            error_msg: Optional[str] = None
    ):
        self.name = name
        self.filename = filename
        self.error_msg = error_msg
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
                "filename": self.filename,
                "error_msg": self.error_msg}

    @classmethod
    def from_dict(cls, state: Dict) -> "Ontology":
        return Ontology(state["name"],
                        dict(state["ontology"]),
                        filename=state.get("filename"),
                        error_msg=state.get("error_msg"))

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


class UrlDialog(QDialog):
    def __init__(self, parent=None, name="Open"):
        super().__init__(parent)

        self.__url_edit = QLineEdit()
        self.__url_edit.setMinimumWidth(300)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Cancel | QDialogButtonBox.Open,
            Qt.Horizontal, self
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout_ = QHBoxLayout()
        layout_.addWidget(QLabel("URL:"))
        layout_.addWidget(self.__url_edit)
        layout.addLayout(layout_)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.setWindowTitle(name)

    @property
    def url(self) -> str:
        return self.__url_edit.text().strip()

    @staticmethod
    def getOpenUrlName(parent: OWWidget, name: str) -> Optional[str]:
        dlg = UrlDialog(parent, name)
        dlg.exec()
        return dlg.result() and dlg.url


LoadFileFormats = [
    "JSON file (*.json)",
    "Pickled Python object file (*.pkl)",
    "OWL file (*.owl)",
]

SaveFileFormats = [
    "JSON file (*.json)",
    "Pickled Python object file (*.pkl)",
]


def read_from_url(parent: OWWidget) -> Optional[Ontology]:
    url = UrlDialog.getOpenUrlName(parent, "Open Ontology")
    if not url:
        return None

    name = os.path.basename(urlparse(url).path)
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.TemporaryDirectory() as dir_name:
            filename = os.path.join(dir_name, name)
            with open(filename, "wb") as f:
                f.write(response.content)
            return _read_from_source(filename)
    except Exception as ex:
        return Ontology(name, {}, filename=url, error_msg=ex.args[0])


def read_from_file(parent: OWWidget) -> Optional[Ontology]:
    filename, _ = QFileDialog.getOpenFileName(
        parent, "Open Ontology",
        os.path.expanduser("~/"),
        ";;".join(LoadFileFormats)
    )
    if not filename:
        return None

    return _read_from_source(filename)


def _read_from_source(filename: str) -> Ontology:
    name = os.path.basename(filename)
    _, ext = os.path.splitext(name)

    if ext == ".json":
        with open(filename) as json_file:
            data = json.load(json_file)

    elif ext == ".pkl":
        with open(filename, "rb") as f:
            data = pickle.load(f)

    elif ext == ".owl":
        world = World()

        try:
            ontology = world.get_ontology(filename)
            onto = ontology.load()
        except OwlReadyOntologyParsingError as ex:
            return Ontology(name, {}, filename=filename, error_msg=ex.args[0])

        with onto:
            data = _onto_to_tree(Thing, world)

    else:
        raise NotImplementedError(f"No readers for file {name}")

    assert isinstance(data, dict)
    return Ontology(name, data, filename=filename)


def save_ontology(parent: OWWidget, filename: str, data: Dict):
    filename, _ = QFileDialog.getSaveFileName(
        parent, "Save Ontology", filename,
        ";;".join(SaveFileFormats)
    )
    if filename:
        assert isinstance(data, dict)

        head, tail = os.path.splitext(filename)

        if tail == ".json":
            with open(filename, "w") as f:
                f.write(json.dumps(data))

        elif tail == ".pkl":
            with open(filename, "wb") as f:
                pickle.dump(data, f)

        else:
            raise NotImplementedError()


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
    RUN_BUTTON, INC_BUTTON = "Generate", "Include"

    settingsHandler = DomainContextHandler()
    ontology_library: List[Dict] = Setting([
        {"name": Ontology.generate_name([]), "ontology": {}},
    ])
    ontology_index: int = Setting(0)
    ontology: OntoType = Setting((), schema_only=True)
    include_children = Setting(True)
    auto_commit = Setting(True)

    class Inputs:
        words = Input("Words", Table)

    class Outputs:
        words = Output("Words", Table, dynamic=False)

    class Warning(OWWidget.Warning):
        no_words_column = Msg("Input is missing 'Words' column.")

    class Error(OWWidget.Error):
        load_error = Msg("{}")

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.__onto_handler = OntologyHandler()

        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        self.__model = PyListModel([], self, flags=flags)
        self.__input_model = QStandardItemModel()
        self.__library_view: QListView = None
        self.__input_view: ListViewSearch = None
        self.__ontology_view: EditableTreeView = None
        self.ontology_info = ""

        self._setup_gui()
        self._restore_state()
        self.settingsAboutToBePacked.connect(self._save_state)

    def _setup_gui(self):
        # control area
        library_box: QGroupBox = gui.vBox(self.controlArea, "Library")
        library_box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)

        edit_triggers = QListView.DoubleClicked | QListView.EditKeyPressed
        self.__library_view = QListView(
            editTriggers=edit_triggers,
            minimumWidth=200,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding),
        )
        self.__library_view.setFixedHeight(100)
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
        menu.addAction(new_from_url)
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

        input_box: QGroupBox = gui.vBox(self.controlArea, "Input")
        self.__input_view = ListViewSearch(
            sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding),
            selectionMode=QListView.ExtendedSelection,
            dragEnabled=True,
        )
        self.__input_view.setModel(self.__input_model)
        self.__input_view.selectionModel().selectionChanged.connect(
            self._enable_include_button
        )

        self.__inc_button = gui.button(
            None, self, self.INC_BUTTON, enabled=False,
            toolTip="Include selected words into the ontology",
            autoDefault=False, callback=self.__on_toggle_include
        )

        input_box.layout().setSpacing(1)
        input_box.layout().addWidget(self.__input_view)
        input_box.layout().addWidget(self.__inc_button)

        self.__run_button = gui.button(
            self.controlArea, self, self.RUN_BUTTON,
            callback=self.__on_toggle_run
        )
        gui.checkBox(
            self.controlArea, self, "include_children", "Include subtree",
            box="Output", callback=self.commit.deferred
        )
        box = gui.vBox(self.controlArea, "Ontology info")
        gui.label(box, self, "%(ontology_info)s")

        gui.auto_send(self.buttonsArea, self, "auto_commit")

        # main area
        ontology_box: QGroupBox = gui.vBox(self.mainArea, box=True)

        self.__ontology_view = EditableTreeView(self)
        self.__ontology_view.dataChanged.connect(
            self.__on_ontology_data_changed
        )
        self.__ontology_view.selectionChanged.connect(self.commit.deferred)

        ontology_box.layout().setSpacing(1)
        ontology_box.layout().addWidget(self.__ontology_view)

        self._enable_include_button()

    def __on_selection_changed(self, selection: QItemSelection, *_):
        self.Error.load_error.clear()
        if selection.indexes():
            self.ontology_index = row = selection.indexes()[0].row()
            data = self.__model[row].cached_word_tree
            self.__ontology_view.set_data(data)
            self.__update_score()
            error_msg = self.__model[row].error_msg
            if error_msg:
                self.Error.load_error(error_msg)

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
        self._import_ontology(ontology)

    def __on_import_url(self):
        ontology = read_from_url(self)
        self._import_ontology(ontology)

    def __on_save(self):
        index = self.__get_selected_row()
        if index is not None:
            filename = self.__model[index].filename
            if filename:
                filename, _ = os.path.splitext(filename)
        else:
            filename = os.path.expanduser("~/")
        save_ontology(self, filename, self.__ontology_view.get_data())
        QApplication.setActiveWindow(self)

    def __on_toggle_include(self):
        if self.task is not None:
            self._cancel_tasks()
        else:
            self._run_insert()

    def __on_toggle_run(self):
        if self.task is not None:
            self._cancel_tasks()
        else:
            self._run()

    def __on_ontology_data_changed(self):
        self.__set_current_modified(self.CACHED)
        self.__update_score()
        self._enable_include_button()
        self.commit.deferred()

    @Inputs.words
    def set_words(self, words: Optional[Table]):
        self.Warning.no_words_column.clear()
        self.__input_model.clear()
        if words:
            if WORDS_COLUMN_NAME in words.domain and words.domain[
                    WORDS_COLUMN_NAME].attributes.get("type") == "words":
                for word in words.get_column_view(WORDS_COLUMN_NAME)[0]:
                    self.__input_model.appendRow(QStandardItem(word))
            else:
                self.Warning.no_words_column()

    @gui.deferred
    def commit(self):
        if self.include_children:
            words = self.__ontology_view.get_selected_words_with_children()
        else:
            words = self.__ontology_view.get_selected_words()
        words_table = self._create_output_table(sorted(words))
        self.Outputs.words.send(words_table)

    @staticmethod
    def _create_output_table(words: List[str]) -> Optional[Table]:
        if not words:
            return None
        return create_words_table(words)

    def _cancel_tasks(self):
        self.cancel()
        self.__inc_button.setText(self.INC_BUTTON)
        self.__run_button.setText(self.RUN_BUTTON)

    def _run(self):
        self.__run_button.setText("Stop")
        words = self.__ontology_view.get_words()
        handler = self.__onto_handler.generate
        self.start(_run, handler, (words,))

    def _run_insert(self):
        self.__inc_button.setText("Stop")
        tree = self.__ontology_view.get_data()
        words = self.__get_selected_input_words()
        handler = self.__onto_handler.insert
        self.start(_run, handler, (tree, words))

    def on_done(self, data: Dict):
        self.__inc_button.setText(self.INC_BUTTON)
        self.__run_button.setText(self.RUN_BUTTON)
        self.__ontology_view.set_data(data, keep_history=True)
        self.__set_current_modified(self.CACHED)
        self.__update_score()

    def __update_score(self):
        tree = self.__ontology_view.get_data()
        score = round(self.__onto_handler.score(tree), 2) \
            if len(tree) == 1 and list(tree.values())[0] else "/"
        self.ontology_info = f"Score: {score}"

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

    def __get_selected_input_words(self) -> List[str]:
        return [self.__input_view.model().data(index) for index in
                self.__input_view.selectedIndexes()]

    def _import_ontology(self, ontology: Ontology):
        if ontology is not None:
            self.__model.append(ontology)
            self.__set_selected_row(len(self.__model) - 1)
        QApplication.setActiveWindow(self)

    def _restore_state(self):
        source = [Ontology.from_dict(s) for s in self.ontology_library]
        self.__model.wrap(source)
        self.__set_selected_row(self.ontology_index)
        if self.ontology:
            self.__ontology_view.set_data(self.ontology)
            self.__set_current_modified(self.CACHED)
            self.__update_score()
            self.commit.now()

    def _save_state(self):
        self.ontology_library = [s.as_dict() for s in self.__model]
        self.ontology = self.__ontology_view.get_data(with_selection=True)

    def _enable_include_button(self):
        tree = self.__ontology_view.get_data()
        words = self.__get_selected_input_words()
        enabled = len(tree) == 1 and len(words) > 0
        self.__inc_button.setEnabled(enabled)

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

    ls = [
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
    words_ = create_words_table(ls)
    WidgetPreview(OWOntology).run(words_)
