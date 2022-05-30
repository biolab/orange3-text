from functools import partial
from typing import List, Tuple

import numpy as np
from AnyQt.QtCore import Signal
from Orange.data import Domain, StringVariable
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Output
from AnyQt.QtWidgets import QGroupBox
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QVBoxLayout, QLineEdit, QPlainTextEdit, QSizePolicy, \
    QSpacerItem
from orangewidget.settings import Setting

from orangecontrib.text import Corpus


class EditorsVerticalScrollArea(gui.VerticalScrollArea):
    def __init__(self):
        super().__init__(parent=None)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self):
        sh = super().sizeHint()
        sh.setHeight(350)
        return sh


class CustomQPlainTextEdit(QPlainTextEdit):
    editingFinished = Signal()

    def focusOutEvent(self, _):
        self.editingFinished.emit()


class DocumentEditor(QGroupBox):
    text_changed = Signal(str, str)

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
        self.layout().addWidget(self.title_le)
        self.layout().addWidget(self.text_area)

    def _on_text_changed(self):
        self.text_changed.emit(self.title_le.text(), self.text_area.toPlainText())

    # todo: add x button


class OWCreateCorpus(OWWidget):
    name = "Create Corpus"
    description = "Write/paste documents to create a corpus"
    icon = "icons/TextFile.svg"
    priority = 100  # todo

    class Outputs:
        corpus = Output('Corpus', Corpus)

    want_main_area = False

    texts: List[Tuple[str, str]] = Setting([("", ""), ("", ""), ("", "")])
    auto_commit: bool = Setting(True)

    def __init__(self):
        super().__init__()

        self.editors = []

        scroll_area = EditorsVerticalScrollArea()
        self.editor_vbox = gui.vBox(self.controlArea, spacing=0)
        scroll_area.setWidget(self.editor_vbox)
        self.controlArea.layout().addWidget(scroll_area)

        for t in self.texts:
            self.add_document_editor(*t)

        gui.button(self.buttonsArea, self, "Add document", self.add_new_editor)
        gui.auto_send(self.buttonsArea, self, "auto_commit")
        self.commit.now()

    def add_document_editor(self, title, text):
        editor = DocumentEditor(title, text)
        editor.text_changed.connect(partial(self._text_changed, len(self.editors)))
        self.editors.append(editor)
        if len(self.editors) > 1:
            # add spacer before each item that boxes do not stick together
            # (except before the first one)
            self.editor_vbox.layout().addSpacerItem(QSpacerItem(1, 10))
        self.editor_vbox.layout().addWidget(editor)
        self.editor_vbox.updateGeometry()

    def remove_document_editor(self):
        if len(self.texts) > 1:
            del self.texts[-1]
            self.editor_vbox.layout().remove(self.editors[-1])
            del self.editors[-1]

    def add_new_editor(self):
        self.texts = [("", "")]
        self.add_document_editor(*self.texts[-1])

    def _text_changed(self, i, title, text):
        self.texts[i] = (title, text)
        self.commit.deferred()

    @gui.deferred
    def commit(self):
        doc_var = StringVariable("Document")
        title_var = StringVariable("Title")
        domain = Domain([], metas=[title_var, doc_var])
        corpus = Corpus.from_numpy(
            domain, np.empty((len(self.texts), 0)), metas=np.array(self.texts),
            text_features=[doc_var]
        )
        corpus.set_title_variable(title_var)
        self.Outputs.corpus.send(corpus)

    def sizeHint(self):
        return QSize(600, 400)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWCreateCorpus).run()
