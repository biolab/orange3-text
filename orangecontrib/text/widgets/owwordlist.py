# pylint: disable=missing-docstring,no-name-in-module
from contextlib import contextmanager
from typing import Optional, List, Dict, Set
import os

import numpy as np

from AnyQt.QtCore import Qt, QModelIndex, QItemSelectionModel, \
    QItemSelection, QItemSelectionRange, Signal
from AnyQt.QtGui import QKeySequence, QPalette, QColor, QPainter
from AnyQt.QtWidgets import QListView, QSizePolicy, QGridLayout, QLineEdit, \
    QRadioButton, QGroupBox, QToolButton, QMenu, QAction, QFileDialog, \
    QStyledItemDelegate, QStyleOptionViewItem, QWidget

from orangewidget.utils.listview import ListViewSearch

from Orange.data import Table, StringVariable, Domain
from Orange.widgets import gui
from Orange.widgets.settings import Setting, DomainContextHandler, \
    ContextSetting
from Orange.widgets.utils.annotated_data import create_annotated_table
from Orange.widgets.utils.itemmodels import DomainModel, ModelActionsWidget, \
    PyListModel
from Orange.widgets.widget import Input, Output, OWWidget, Msg


@contextmanager
def disconnected(signal, slot, connection_type=Qt.AutoConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, connection_type)


class WordList:
    NotModified, Modified = range(2)

    def __init__(self, name: str, words: List, filename: Optional[str] = None):
        self.name = name
        self.filename = filename
        # library words
        self.words = list(words)
        # current words
        self.cached_words = list(words)
        # update rule changes current words
        self.update_rule_flag = WordList.NotModified

    @property
    def flags(self) -> int:
        # 0 - NotModified, 1 - Modified
        return int(self.words != self.cached_words or
                   self.update_rule_flag == WordList.Modified)

    def as_dict(self) -> Dict:
        return {"name": self.name,
                "words": list(self.words),
                "filename": self.filename}

    @classmethod
    def from_dict(cls, state: Dict) -> "WordList":
        return WordList(state["name"],
                        state["words"].copy(),
                        filename=state.get("filename"))

    @staticmethod
    def generate_word_list_name(taken_names: List[str]) -> str:
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


class UpdateRules:
    INTERSECT, UNION, INPUT, LIBRARY = range(4)
    ITEMS = ["Intersection", "Union", "Only input", "Ignore input"]

    @staticmethod
    def update(model: PyListModel, lib_words: List, in_words: List, rule: int):
        if rule == UpdateRules.INTERSECT:
            intersect = set(lib_words) & set(in_words)
            model.wrap(list({k: None for k in lib_words if k in intersect}))
        elif rule == UpdateRules.UNION:
            model.wrap(list({k: None for k in lib_words + in_words}))
        elif rule == UpdateRules.INPUT:
            model.wrap(list(in_words))
        elif rule == UpdateRules.LIBRARY:
            model.wrap(list(lib_words))
        else:
            raise NotImplementedError


class WordListItemDelegate(QStyledItemDelegate):
    def displayText(self, word_list: WordList, _) -> str:
        return "*" + word_list.name if word_list.flags & WordList.Modified \
            else word_list.name

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex):
        word_list = index.data(Qt.DisplayRole)
        if word_list.flags & WordList.Modified:
            option = QStyleOptionViewItem(option)
            option.palette.setColor(QPalette.Text, QColor(Qt.red))
            option.palette.setColor(QPalette.Highlight, QColor(Qt.darkRed))
            option.palette.setColor(QPalette.HighlightedText, QColor(Qt.white))
        super().paint(painter, option, index)

    def createEditor(self, parent: QWidget, _, __) -> QLineEdit:
        return QLineEdit(parent)

    def setEditorData(self, editor: QLineEdit, index: QModelIndex):
        word_list = index.data(Qt.DisplayRole)
        editor.setText(word_list.name)

    def setModelData(self, editor: QLineEdit,
                     model: PyListModel, index: QModelIndex):
        model[index.row()].name = str(editor.text())


class ListView(ListViewSearch):
    drop_finished = Signal()

    def __init__(self):
        super().__init__(
            editTriggers=QListView.DoubleClicked | QListView.EditKeyPressed,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding),
            minimumWidth=200,
            selectionMode=QListView.ExtendedSelection,
            dragEnabled=True,
            acceptDrops=True,
            defaultDropAction=Qt.MoveAction,
        )
        self.setDropIndicatorShown(True)

    def startDrag(self, actions):
        super().startDrag(actions)
        self.drop_finished.emit()


class OWWordList(OWWidget):
    name = "Word List"
    description = "Create a list of words."
    icon = "icons/WordList.svg"
    priority = 1000

    class Inputs:
        words = Input("Words", Table)

    class Outputs:
        selected_words = Output("Selected Words", Table)
        words = Output("Words", Table)

    class Warning(OWWidget.Warning):
        no_string_vars = Msg("Input needs at least one Text variable.")

    NONE, CACHED, LIBRARY = range(3)  # library list modification types

    want_main_area = False
    resizing_enabled = True

    settingsHandler = DomainContextHandler()
    word_list_library: List[Dict] = Setting([
        {"name": WordList.generate_word_list_name([]), "words": []},
    ])
    word_list_index: int = Setting(0)
    words_var: Optional[StringVariable] = ContextSetting(None)
    update_rule_index: int = Setting(UpdateRules.INTERSECT)
    words: List[str] = Setting(None, schema_only=True)
    selected_words: Set[str] = Setting(set(), schema_only=True)

    def __init__(self):
        super().__init__(self)
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        self.library_model = PyListModel([], self, flags=flags)
        self.words_model = PyListModel([], self, flags=flags, enable_dnd=True)

        self.library_view: QListView = None
        self.words_view: ListView = None

        self.__input_words_model = DomainModel(valid_types=(StringVariable,))
        self.__input_words: Optional[Table] = None

        self.__library_box: QGroupBox = gui.vBox(None, "Library")
        self.__input_box: QGroupBox = gui.vBox(None, "Input")
        self.__words_box: QGroupBox = gui.vBox(None, box=True)
        self.__update_rule_rb: QRadioButton = None

        self.__add_word_action: QAction = None
        self.__remove_word_action: QAction = None

        self._setup_gui()
        self._restore_state()
        self.settingsAboutToBePacked.connect(self._save_state)

    def _setup_gui(self):
        layout = QGridLayout()
        gui.widgetBox(self.controlArea, orientation=layout)

        self._setup_library_box()
        self._setup_input_box()
        self._setup_words_box()

        layout.addWidget(self.__library_box, 0, 0)
        layout.addWidget(self.__input_box, 1, 0)
        layout.addWidget(self.__words_box, 0, 1, 0, 1)

    def _setup_library_box(self):
        self.library_view = QListView(
            editTriggers=QListView.DoubleClicked | QListView.EditKeyPressed,
            minimumWidth=200,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding),
        )
        self.library_view.setItemDelegate(WordListItemDelegate(self))
        self.library_view.setModel(self.library_model)
        self.library_view.selectionModel().selectionChanged.connect(
            self.__on_library_selection_changed
        )

        self.__library_box.layout().setSpacing(1)
        self.__library_box.layout().addWidget(self.library_view)

        actions_widget = ModelActionsWidget()
        actions_widget.layout().setSpacing(1)

        action = QAction("+", self)
        action.setToolTip("Add a new word list to the library")
        action.triggered.connect(self.__on_add_word_list)
        actions_widget.addAction(action)

        action = QAction("\N{MINUS SIGN}", self)
        action.setToolTip("Remove word list from library")
        action.triggered.connect(self.__on_remove_word_list)
        actions_widget.addAction(action)

        action = QAction("Update", self)
        action.setToolTip("Save changes in the editor to library")
        action.setShortcut(QKeySequence(QKeySequence.Save))
        action.triggered.connect(self.__on_update_word_list)
        actions_widget.addAction(action)

        gui.rubber(actions_widget.layout())

        action = QAction("More", self, toolTip="More actions")

        new_from_file = QAction("Import Words from File", self)
        new_from_file.triggered.connect(self.__on_import_word_list)

        save_to_file = QAction("Save Words to File", self)
        save_to_file.setShortcut(QKeySequence(QKeySequence.SaveAs))
        save_to_file.triggered.connect(self.__on_save_word_list)

        menu = QMenu(actions_widget)
        menu.addAction(new_from_file)
        menu.addAction(save_to_file)
        action.setMenu(menu)
        button = actions_widget.addAction(action)
        button.setPopupMode(QToolButton.InstantPopup)
        self.__library_box.layout().addWidget(actions_widget)

    def __on_library_selection_changed(self, selected: QItemSelection, *_):
        index = [i.row() for i in selected.indexes()]
        if index:
            current = index[0]
            word_list: WordList = self.library_model[current]
            self.word_list_index = current
            self.selected_words = set()
            self.words_model.wrap(list(word_list.cached_words))
            self._apply_update_rule()

    def __on_add_word_list(self):
        taken = [l.name for l in self.library_model]
        name = WordList.generate_word_list_name(taken)
        word_list = WordList(name, self.words_model[:])
        self.library_model.append(word_list)
        self._set_selected_word_list(len(self.library_model) - 1)

    def __on_remove_word_list(self):
        index = self._get_selected_word_list_index()
        if index is not None:
            del self.library_model[index]
            self._set_selected_word_list(max(index - 1, 0))
            self._apply_update_rule()

    def __on_update_word_list(self):
        self._set_word_list_modified(mod_type=self.LIBRARY)

    def __on_import_word_list(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Word List",
            os.path.expanduser("~/"),
            "Text files (*.txt)\nAll files(*.*)"
        )
        if filename:
            name = os.path.basename(filename)
            with open(filename, encoding="utf-8") as f:
                words = [line.strip() for line in f.readlines()]
            self.library_model.append(WordList(name, words, filename=filename))
            self._set_selected_word_list(len(self.library_model) - 1)
            self._apply_update_rule()

    def __on_save_word_list(self):
        index = self._get_selected_word_list_index()
        if index is not None:
            word_list = self.library_model[index]
            filename = word_list.filename
        else:
            filename = os.path.expanduser("~/")

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Word List", filename,
            "Text files (*.txt)\nAll files(*.*)"
        )
        if filename:
            head, tail = os.path.splitext(filename)
            if not tail:
                filename = head + ".txt"

            with open(filename, "w", encoding="utf-8") as f:
                for word in self.words_model:
                    f.write(f"{word}\n")

    def _setup_input_box(self):
        gui.comboBox(
            self.__input_box, self, "words_var", label="Word variable:",
            orientation=Qt.Vertical, model=self.__input_words_model,
            callback=self._apply_update_rule
        )
        gui.radioButtons(
            self.__input_box, self, "update_rule_index", UpdateRules.ITEMS,
            label="Update: ", orientation=Qt.Vertical,
            callback=self.__on_update_rule_changed
        )
        self.__input_box.setEnabled(False)

    def __on_update_rule_changed(self):
        self._enable_words_actions()
        self._apply_update_rule()

    def _setup_words_box(self):
        self.words_view = ListView()
        self.words_view.drop_finished.connect(self.__on_words_data_changed)
        self.words_view.setModel(self.words_model)
        self.words_view.selectionModel().selectionChanged.connect(
            self.__on_words_selection_changed)

        self.words_model.dataChanged.connect(self.__on_words_data_changed)

        self.__words_box.layout().setSpacing(1)
        self.__words_box.layout().addWidget(self.words_view)

        actions_widget = ModelActionsWidget()
        actions_widget.layout().setSpacing(1)

        action = QAction("+", self.words_view, toolTip="Add a new word")
        action.triggered.connect(self.__on_add_word)
        actions_widget.addAction(action)
        self.__add_word_action = action

        action = QAction("\N{MINUS SIGN}", self, toolTip="Remove word")
        action.triggered.connect(self.__on_remove_word)
        actions_widget.addAction(action)
        self.__remove_word_action = action

        gui.rubber(actions_widget)

        action = QAction("Sort", self)
        action.setToolTip("Sort words alphabetically")
        action.triggered.connect(self.__on_apply_sorting)
        actions_widget.addAction(action)

        self.__words_box.layout().addWidget(actions_widget)

    def __on_words_data_changed(self):
        self._set_word_list_modified(mod_type=self.CACHED)
        self.commit()

    def __on_words_selection_changed(self):
        self.commit()

    def __on_add_word(self):
        row = self.words_model.rowCount()
        if not self.words_model.insertRow(self.words_model.rowCount()):
            return
        with disconnected(self.words_view.selectionModel().selectionChanged,
                          self.__on_words_selection_changed):
            self._set_selected_words([0])
            index = self.words_model.index(row, 0)
            self.words_view.setCurrentIndex(index)
            self.words_model.setItemData(index, {Qt.EditRole: ""})
        self.words_view.edit(index)

    def __on_remove_word(self):
        rows = self.words_view.selectionModel().selectedRows(0)
        if not rows:
            return

        indices = sorted([row.row() for row in rows], reverse=True)
        with disconnected(self.words_view.selectionModel().selectionChanged,
                          self.__on_words_selection_changed):
            for index in indices:
                self.words_model.removeRow(index)
            if self.words_model:
                self._set_selected_words([max(0, indices[-1] - 1)])
        self.__on_words_data_changed()

    def __on_apply_sorting(self):
        if not self.words_model:
            return
        words = self.words_model[:]
        mask = np.zeros(len(words), dtype=bool)
        selection = self._get_selected_words_indices()
        if selection:
            mask[selection] = True

        indices = np.argsort(words)
        self.words_model.wrap([words[i] for i in indices])
        self._set_word_list_modified(mod_type=self.CACHED)
        if selection:
            self._set_selected_words(list(np.flatnonzero(mask[indices])))
        else:
            self.commit()

    @Inputs.words
    def set_words(self, words: Optional[Table]):
        self.closeContext()
        self.__input_words = words
        self._check_input_words()
        self._init_controls()
        self.openContext(self.__input_words)
        self._apply_update_rule()

    def _check_input_words(self):
        self.Warning.no_string_vars.clear()
        if self.__input_words:
            metas = self.__input_words.domain.metas
            if not any(isinstance(m, StringVariable) for m in metas):
                self.Warning.no_string_vars()
                self.__input_words = None

    def _init_controls(self):
        words = self.__input_words
        domain = words.domain if words is not None else None
        self.__input_words_model.set_domain(domain)
        if len(self.__input_words_model) > 0:
            self.words_var = self.__input_words_model[0]
        self.__input_box.setEnabled(bool(self.__input_words_model))
        self._enable_words_actions()

    def _enable_words_actions(self):
        if bool(self.__input_words_model) \
                and self.update_rule_index != UpdateRules.LIBRARY:
            self.words_view.setEditTriggers(QListView.NoEditTriggers)
            self.__add_word_action.setEnabled(False)
            self.__remove_word_action.setEnabled(False)
        else:
            self.words_view.setEditTriggers(
                QListView.DoubleClicked | QListView.EditKeyPressed
            )
            self.__add_word_action.setEnabled(True)
            self.__remove_word_action.setEnabled(True)

    def _apply_update_rule(self):
        lib_index = self._get_selected_word_list_index()
        lib_words, in_words, update_rule = [], [], UpdateRules.LIBRARY
        if lib_index is not None:
            lib_words = self.library_model[lib_index].cached_words
        else:
            lib_words = self.words_model[:]
        if self.__input_words is not None:
            in_words = self.__input_words.get_column_view(self.words_var)[0]
            in_words = list(in_words)
            update_rule = self.update_rule_index

        UpdateRules.update(self.words_model, lib_words, in_words, update_rule)
        if lib_index is not None:
            cached = self.library_model[lib_index].cached_words
            modified = WordList.NotModified if cached == self.words_model[:] \
                else WordList.Modified
            self.library_model[lib_index].update_rule_flag = modified
            self._set_word_list_modified(mod_type=self.NONE)
            self.library_view.repaint()

        # Apply selection. selection_changed invokes commit().
        # If there is no selection, call commit explicitly.
        if any(w in self.selected_words for w in self.words_model):
            self.set_selected_words()
            self.words_view.repaint()
        else:
            self.commit()

    def commit(self):
        selection = self._get_selected_words_indices()
        self.selected_words = set(np.array(self.words_model)[selection])

        words, selected_words = None, None
        if self.words_model:
            words_var = StringVariable("Words")
            words_var.attributes = {"type": "words"}
            domain = Domain([], metas=[words_var])
            _words = Table.from_list(domain, [[w] for w in self.words_model])
            _words.name = "Words"
            if selection:
                selected_words = _words[selection]
            words = create_annotated_table(_words, selection)
        self.Outputs.words.send(words)
        self.Outputs.selected_words.send(selected_words)

    def _set_word_list_modified(self, mod_type):
        index = self._get_selected_word_list_index()
        if index is not None:
            if mod_type == self.LIBRARY:
                self.library_model[index].words = self.words_model[:]
                self.library_model[index].cached_words = self.words_model[:]
                self.library_model[index].update_rule_flag \
                    = WordList.NotModified
            elif mod_type == self.CACHED:
                self.library_model[index].cached_words = self.words_model[:]
            elif mod_type == self.NONE:
                pass
            else:
                raise NotImplementedError
            self.library_model.emitDataChanged(index)
            self.library_view.repaint()

    def _set_selected_word_list(self, index: int):
        sel_model: QItemSelectionModel = self.library_view.selectionModel()
        sel_model.select(self.library_model.index(index, 0),
                         QItemSelectionModel.ClearAndSelect)

    def _get_selected_word_list_index(self) -> Optional[int]:
        rows = self.library_view.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _set_selected_words(self, indices: List[int]):
        selection = QItemSelection()
        sel_model: QItemSelectionModel = self.words_view.selectionModel()
        for i in indices:
            selection.append(QItemSelectionRange(self.words_model.index(i, 0)))
        sel_model.select(selection, QItemSelectionModel.ClearAndSelect)

    def _get_selected_words_indices(self) -> List[int]:
        rows = self.words_view.selectionModel().selectedRows()
        return [row.row() for row in rows]

    def set_selected_words(self):
        if self.selected_words:
            indices = [i for i, w in enumerate(self.words_model)
                       if w in self.selected_words]
            self._set_selected_words(indices)

    def _restore_state(self):
        source = [WordList.from_dict(s) for s in self.word_list_library]
        self.library_model.wrap(source)
        # __on_library_selection_changed() (invoked by _set_selected_word_list)
        # clears self.selected_words
        selected_words = self.selected_words
        self._set_selected_word_list(self.word_list_index)

        if self.words is not None:
            self.words_model.wrap(list(self.words))
            self._set_word_list_modified(mod_type=self.CACHED)
            if selected_words:
                self.selected_words = selected_words
                self.set_selected_words()
            elif len(self.word_list_library) > self.word_list_index and \
                self.word_list_library[self.word_list_index] != self.words:
                self.commit()

    def _save_state(self):
        self.word_list_library = [s.as_dict() for s in self.library_model]
        self.words = self.words_model[:]

    def send_report(self):
        library = self.library_model[self.word_list_index].name \
            if self.library_model else "/"
        settings = [("Library", library)]
        if self.__input_words:
            self.report_data("Input Words", self.__input_words)
            settings.append(("Word variable", self.words_var))
            rule = UpdateRules.ITEMS[self.update_rule_index]
            settings.append(("Update", rule))
        self.report_items("Settings", settings)
        self.report_paragraph("Words", ", ".join(self.words_model[:]))


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    words_vars = [StringVariable("S1"), StringVariable("S2")]
    lst = [["foo", "A"], ["bar", "B"], ["foobar", "C"]]
    input_table = Table.from_list(Domain([], metas=words_vars), lst)
    # WidgetPreview(OWWordList).run(set_words=input_table)
    WidgetPreview(OWWordList).run()
