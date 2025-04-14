from typing import List, Tuple

import numpy as np
from AnyQt.QtCore import QSize, Qt, Signal
from AnyQt.QtWidgets import (
    QGroupBox,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)
from Orange.data import Domain, StringVariable
from Orange.widgets import gui
from Orange.widgets.widget import Output, OWWidget
from orangewidget.settings import Setting

from orangecontrib.text import Corpus
from orangecontrib.text.language import (
    DEFAULT_LANGUAGE, LanguageModel, LANG2ISO, migrate_language_name
)


class EditorsVerticalScrollArea(gui.VerticalScrollArea):
    def __init__(self):
        super().__init__(parent=None)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


class CustomQPlainTextEdit(QPlainTextEdit):
    editingFinished = Signal()

    def focusOutEvent(self, _):
        # TextEdit does not have editingFinished
        self.editingFinished.emit()


class DocumentEditor(QGroupBox):
    text_changed = Signal(str, str)
    remove_clicked = Signal()

    def __init__(self, title, text, parent=None):
        super().__init__(parent)

        self.setLayout(QVBoxLayout())
        self.title_le = QLineEdit()
        self.title_le.setPlaceholderText("Document title")
        self.title_le.setText(title)
        self.title_le.editingFinished.connect(self._on_text_changed)
        self.text_area = CustomQPlainTextEdit()
        self.text_area.setPlaceholderText("Document text")
        self.text_area.setPlainText(text)
        self.text_area.editingFinished.connect(self._on_text_changed)

        remove_button = QPushButton("x")
        remove_button.setFixedWidth(35)
        remove_button.setFocusPolicy(Qt.NoFocus)
        remove_button.clicked.connect(self._on_remove_clicked)
        box = gui.hBox(self)
        box.layout().addWidget(self.title_le)
        box.layout().addWidget(remove_button)
        self.layout().addWidget(self.text_area)

    def _on_text_changed(self):
        self.text_changed.emit(self.title_le.text(), self.text_area.toPlainText())

    def _on_remove_clicked(self):
        self.remove_clicked.emit()


class OWCreateCorpus(OWWidget):
    name = "Create Corpus"
    description = "Write/paste documents to create a corpus"
    icon = "icons/CreateCorpus.svg"
    priority = 120
    keywords = "create corpus, write"

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False

    settings_version = 2
    language: str = Setting(DEFAULT_LANGUAGE)
    texts: List[Tuple[str, str]] = Setting([("", "")] * 3)
    auto_commit: bool = Setting(True)

    def __init__(self):
        super().__init__()
        self.editors = []

        gui.comboBox(
            self.controlArea,
            self,
            "language",
            model=LanguageModel(include_none=True),
            box="Language",
            orientation=Qt.Horizontal,
            callback=self.commit.deferred,
            sendSelectedValue=True,
            searchable=True,
        )

        scroll_area = EditorsVerticalScrollArea()
        self.editor_vbox = gui.vBox(self.controlArea, spacing=0)
        self.editor_vbox.layout().setSpacing(10)
        scroll_area.setWidget(self.editor_vbox)
        self.controlArea.layout().addWidget(scroll_area)

        for t in self.texts:
            self._add_document_editor(*t)

        add_btn = gui.button(
            self.buttonsArea, self, "Add document", self._add_new_editor
        )
        add_btn.setFocusPolicy(Qt.NoFocus)
        gui.auto_apply(self.buttonsArea, self, "auto_commit")
        self.commit.now()

    def _add_document_editor(self, title, text):
        """Function that handles adding new editor with texts provided"""
        editor = DocumentEditor(title, text)
        editor.text_changed.connect(self._text_changed)
        editor.remove_clicked.connect(self._remove_document_editor)
        self.editors.append(editor)
        self.editor_vbox.layout().addWidget(editor)
        self.editor_vbox.updateGeometry()

    def _remove_document_editor(self):
        """Remove the editor on the click of x button on the editor"""
        if len(self.editors) > 1:
            editor = self.sender()
            i = self.editors.index(editor)
            del self.texts[i]
            self.editor_vbox.layout().removeWidget(editor)
            self.editors.remove(editor)
            editor.deleteLater()
            self.commit.deferred()

    def _add_new_editor(self):
        """Add editor on the click of Add document button"""
        self.texts.append(("", ""))
        self._add_document_editor(*self.texts[-1])
        self.commit.deferred()

    def _text_changed(self, title, text):
        """Called when any text change, corrects texts in settings"""
        editor = self.sender()
        self.texts[self.editors.index(editor)] = (title, text)
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        """Create a new corpus and output it"""
        filtered_texts = [
        (title.strip(), text.strip())
        for title, text in self.texts
        if title.strip() or text.strip()
        ]

        if not filtered_texts:
            self.Outputs.corpus.send(None)
            return

        metas = np.array(filtered_texts, dtype=object)
        doc_var = StringVariable("Document")
        title_var = StringVariable("Title")
        domain = Domain([], metas=[title_var, doc_var])
        corpus = Corpus.from_numpy(
            domain,
            np.empty((len(filtered_texts), 0)),
            metas=metas,
            text_features=[doc_var],
            language=self.language,
        )
        corpus.set_title_variable(title_var)
        self.Outputs.corpus.send(corpus)

    def sizeHint(self) -> QSize:
        return QSize(600, 650)

    @classmethod
    def migrate_settings(cls, settings, version):
        if version is None or version < 2:
            if "language" in settings:
                language = migrate_language_name(settings["language"])
                settings["language"] = LANG2ISO[language]


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWCreateCorpus).run()
