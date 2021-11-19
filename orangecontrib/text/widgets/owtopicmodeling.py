import functools
from typing import Any

import numpy as np

from AnyQt import QtGui, QtCore
from AnyQt.QtCore import pyqtSignal, QSize
from AnyQt.QtWidgets import (QVBoxLayout, QButtonGroup, QRadioButton,
                             QGroupBox, QTreeWidgetItem, QTreeWidget,
                             QStyleOptionViewItem, QStyledItemDelegate, QStyle)
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from orangewidget.utils.itemdelegates import text_color_for_state

from gensim.models import CoherenceModel

from Orange.widgets import settings
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.data import Table
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import Topic, Topics, LdaWrapper, HdpWrapper, \
    LsiWrapper
from orangecontrib.text.topics.topics import GensimWrapper


class TopicWidget(gui.OWComponent, QGroupBox):
    Model = NotImplemented
    valueChanged = pyqtSignal(object)

    parameters = ()
    spin_format = '{description}:'

    def __init__(self, master, **kwargs):
        QGroupBox.__init__(self, **kwargs)
        gui.OWComponent.__init__(self, master)
        self.model = self.create_model()
        QVBoxLayout(self)
        for parameter, description, minv, maxv, step, _type in self.parameters:
            spin = gui.spin(self, self, parameter, minv=minv, maxv=maxv, step=step,
                            label=self.spin_format.format(description=description, parameter=parameter),
                            labelWidth=220, spinType=_type)
            spin.clearFocus()
            spin.editingFinished.connect(self.on_change)

    def on_change(self):
        self.model = self.create_model()
        self.valueChanged.emit(self)

    def create_model(self):
        return self.Model(**{par[0]: getattr(self, par[0]) for par in self.parameters})

    def report_model(self):
        return self.model.name, ((par[1], getattr(self, par[0])) for par in self.parameters)


class LdaWidget(TopicWidget):
    Model = LdaWrapper

    parameters = (
        ('num_topics', 'Number of topics', 1, 500, 1, int),
    )
    num_topics = settings.Setting(10)


class LsiWidget(TopicWidget):
    Model = LsiWrapper

    parameters = (
        ('num_topics', 'Number of topics', 1, 500, 1, int),
    )
    num_topics = settings.Setting(10)


class HdpWidget(TopicWidget):
    Model = HdpWrapper

    spin_format = '{description}:'
    parameters = (
        ('gamma', 'First level concentration (γ)', .1, 10, .5, float),
        ('alpha', 'Second level concentration (α)', 1, 10, 1, int),
        ('eta', 'The topic Dirichlet (α)', 0.001, .5, .01, float),
        ('T', 'Top level truncation level (Τ)', 10, 150, 1, int),
        ('K', 'Second level truncation level (Κ)', 1, 50, 1, int),
        ('kappa', 'Learning rate (κ)', .1, 10., .1, float),
        ('tau', 'Slow down parameter (τ)', 16., 256., 1., float),
    )
    gamma = settings.Setting(1)
    alpha = settings.Setting(1)
    eta = settings.Setting(.01)
    T = settings.Setting(150)
    K = settings.Setting(15)
    kappa = settings.Setting(1)
    tau = settings.Setting(64)


def require(attribute):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, attribute, None) is not None:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


def _run(corpus: Corpus, model: GensimWrapper, state: TaskState):
    def callback(i: float):
        state.set_progress_value(i * 100)
        if state.is_interruption_requested():
            raise Exception

    return model.fit_transform(corpus.copy(), on_progress=callback)


class OWTopicModeling(OWWidget, ConcurrentWidgetMixin):
    name = "Topic Modelling"
    description = "Uncover the hidden thematic structure in a corpus."
    icon = "icons/TopicModeling.svg"
    priority = 400
    keywords = ["LDA"]

    settingsHandler = DomainContextHandler()

    # Input/output
    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Table, default=True)
        selected_topic = Output("Selected Topic", Topic)
        all_topics = Output("All Topics", Topics)

    want_main_area = True

    methods = [
        (LsiWidget, 'lsi'),
        (LdaWidget, 'lda'),
        (HdpWidget, 'hdp'),
    ]

    # Settings
    autocommit = settings.Setting(True)
    method_index = settings.Setting(0)

    lsi = settings.SettingProvider(LsiWidget)
    hdp = settings.SettingProvider(HdpWidget)
    lda = settings.SettingProvider(LdaWidget)

    selection = settings.Setting(None, schema_only=True)

    control_area_width = 300

    class Warning(OWWidget.Warning):
        less_topics_found = Msg('Less topics found than requested.')

    class Error(OWWidget.Error):
        unexpected_error = Msg("{}")

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        self.corpus = None
        self.learning_thread = None
        self.__pending_selection = self.selection
        self.perplexity = "n/a"
        self.coherence = "n/a"

        # Commit button
        gui.auto_commit(self.buttonsArea, self, 'autocommit', 'Commit', box=False)

        button_group = QButtonGroup(self, exclusive=True)
        button_group.buttonClicked[int].connect(self.change_method)

        self.widgets = []
        method_layout = QVBoxLayout()
        self.controlArea.layout().addLayout(method_layout)
        for i, (method, attr_name) in enumerate(self.methods):
            widget = method(self, title='Options')
            widget.setFixedWidth(self.control_area_width)
            widget.valueChanged.connect(self.commit)
            self.widgets.append(widget)
            setattr(self, attr_name, widget)

            rb = QRadioButton(text=widget.Model.name)
            button_group.addButton(rb, i)
            method_layout.addWidget(rb)
            method_layout.addWidget(widget)

        button_group.button(self.method_index).setChecked(True)
        self.toggle_widgets()
        method_layout.addStretch()

        box = gui.vBox(self.controlArea, "Topic evaluation")
        gui.label(box, self, "Log perplexity: %(perplexity)s")
        gui.label(box, self, "Topic coherence: %(coherence)s")
        self.controlArea.layout().insertWidget(1, box)

        # Topics description
        self.topic_desc = TopicViewer()
        self.topic_desc.topicSelected.connect(self.send_topic_by_id)
        self.mainArea.layout().addWidget(self.topic_desc)
        self.topic_desc.setFocus()

    @Inputs.corpus
    def set_data(self, data=None):
        self.Warning.less_topics_found.clear()
        self.corpus = data
        self.apply()

    def commit(self):
        if self.corpus is not None:
            self.apply()

    @property
    def model(self):
        return self.widgets[self.method_index].model

    def change_method(self, new_index):
        if self.method_index != new_index:
            self.method_index = new_index
            self.toggle_widgets()
            self.commit()

    def toggle_widgets(self):
        for i, widget in enumerate(self.widgets):
            widget.setVisible(i == self.method_index)

    def apply(self):
        self.cancel()
        self.topic_desc.clear()
        if self.corpus is not None:
            self.Warning.less_topics_found.clear()
            self.start(_run, self.corpus, self.model)
        else:
            self.topic_desc.clear()
            self.Outputs.corpus.send(None)
            self.Outputs.selected_topic.send(None)
            self.Outputs.all_topics.send(None)

    def on_done(self, corpus):
        self.Outputs.corpus.send(corpus)
        pos_tags = self.corpus.pos_tags is not None
        self.topic_desc.show_model(self.model, pos_tags=pos_tags)
        if self.__pending_selection:
            self.topic_desc.select(self.__pending_selection)
            self.__pending_selection = None

        if self.model.actual_topics != self.model.num_topics:
            self.Warning.less_topics_found()

        if self.model.name == "Latent Dirichlet Allocation":
            bound = self.model.model.log_perplexity(corpus.ngrams_corpus)
            self.perplexity = "{:.5f}".format(np.exp2(-bound))
        cm = CoherenceModel(
            model=self.model.model, texts=corpus.tokens, corpus=corpus, coherence="c_v"
        )
        coherence = cm.get_coherence()
        self.coherence = "{:.5f}".format(coherence)

        self.Outputs.all_topics.send(self.model.get_all_topics_table())

    def on_exception(self, ex: Exception):
        self.Error.unexpected_error(str(ex))

    def on_partial_result(self, result: Any) -> None:
        pass

    def send_report(self):
        self.report_items(*self.widgets[self.method_index].report_model())
        if self.corpus is not None:
            self.report_items('Topics', self.topic_desc.report())

    def send_topic_by_id(self, topic_id=None):
        self.selection = topic_id
        if self.model.model and topic_id is not None:
            self.Outputs.selected_topic.send(
                self.model.get_topics_table_by_id(topic_id))


class TopicViewerTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, topic_id, words, weights, parent,
                 color_by_weights=False, pos_tags=False):
        super().__init__(parent)
        self.topic_id = topic_id
        self.words = words
        self.weights = weights
        self.color_by_weights = color_by_weights
        self.pos_tags = pos_tags

        self.setText(0, '{:d}'.format(topic_id + 1))
        self.setText(1, ', '.join(self._color(word.rsplit("_", 1)[0], weight) if
                                  self.pos_tags else self._color(word, weight)
                                  for word, weight in zip(words, weights)))

    def _color(self, word, weight):
        if self.color_by_weights:
            red = '#ff6600'
            green = '#00cc00'
            color = green if weight > 0 else red
            return '<span style="color: {}">{}</span>'.format(color, word)
        else:
            return word

    def report(self):
        return self.text(0), self.text(1)


class TopicViewer(QTreeWidget):
    """ Just keeps stuff organized. Holds topic visualization widget and related functions.

    """

    columns = ['Topic', 'Topic keywords']
    topicSelected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setColumnCount(len(self.columns))
        self.setHeaderLabels(self.columns)
        self.resize_columns()
        self.itemSelectionChanged.connect(self.selected_topic_changed)
        self.setItemDelegate(HTMLDelegate())    # enable colors
        self.selected_id = None

    def resize_columns(self):
        for i in range(self.columnCount()):
            self.resizeColumnToContents(i)

    def show_model(self, topic_model, pos_tags=False):
        self.clear()
        if topic_model.model:
            for i in range(topic_model.num_topics):
                words, weights = topic_model.get_top_words_by_id(i)
                if words:
                    it = TopicViewerTreeWidgetItem(
                        i, words, weights, self,
                        color_by_weights=topic_model.has_negative_weights,
                        pos_tags=pos_tags)
                    self.addTopLevelItem(it)

            self.resize_columns()
            self.setCurrentItem(self.topLevelItem(0))

    def selected_topic_changed(self):
        selected = self.selectedItems()
        if selected:
            self.select(selected[0].topic_id)
            self.topicSelected.emit(self.selected_id)
        else:
            self.topicSelected.emit(None)

    def report(self):
        root = self.invisibleRootItem()
        child_count = root.childCount()
        return [root.child(i).report()
                for i in range(child_count)]

    def sizeHint(self):
        return QSize(700, 300)

    def select(self, index):
        self.selected_id = index
        self.setCurrentItem(self.topLevelItem(index))


class HTMLDelegate(QStyledItemDelegate):
    """ This delegate enables coloring of words in QTreeWidgetItem. 
    Adopted from https://stackoverflow.com/a/5443112/892987 """
    def paint(self, painter, option, index):
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        style = QApplication.style() if options.widget is None else options.widget.style()

        doc = QtGui.QTextDocument()
        doc.setHtml(options.text)

        options.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()
        ctx.palette.setColor(QtGui.QPalette.Text,
                             text_color_for_state(option.palette, option.state))

        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)
        painter.save()
        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        doc = QtGui.QTextDocument()
        doc.setHtml(options.text)
        doc.setTextWidth(options.rect.width())
        return QtCore.QSize(int(doc.idealWidth()), int(doc.size().height()))


if __name__ == '__main__':
    from AnyQt.QtWidgets import QApplication

    app = QApplication([])
    widget = OWTopicModeling()
    # widget.set_data(Corpus.from_file('book-excerpts'))
    widget.set_data(Corpus.from_file('deerwester'))
    widget.show()
    app.exec()
