import functools

from AnyQt import QtGui, QtCore
from AnyQt.QtCore import pyqtSignal, QSize
from AnyQt.QtWidgets import (QVBoxLayout, QButtonGroup, QRadioButton,
                             QGroupBox, QTreeWidgetItem, QTreeWidget,
                             QStyleOptionViewItem, QStyledItemDelegate, QStyle)

from Orange.widgets import settings
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table
from Orange.widgets.data.contexthandlers import DomainContextHandler
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import Topic, LdaWrapper, HdpWrapper, LsiWrapper
from orangecontrib.text.widgets.utils.concurrent import asynchronous


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


class OWTopicModeling(OWWidget):
    name = "Topic Modelling"
    description = "Uncover the hidden thematic structure in a corpus."
    icon = "icons/TopicModeling.svg"
    priority = 400

    settingsHandler = DomainContextHandler()

    # Input/output
    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Table)
        selected_topic = Output("Selected Topic", Topic)
        all_topics = Output("All Topics", Table)

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

    control_area_width = 300

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.learning_thread = None

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

        # Topics description
        self.topic_desc = TopicViewer()
        self.topic_desc.topicSelected.connect(self.send_topic_by_id)
        self.mainArea.layout().addWidget(self.topic_desc)
        self.topic_desc.setFocus()

    @Inputs.corpus
    def set_data(self, data=None):
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
        self.learning_task.stop()
        if self.corpus is not None:
            self.learning_task()
        else:
            self.on_result(None)

    @asynchronous
    def learning_task(self):
        return self.model.fit_transform(self.corpus.copy(), chunk_number=100, on_progress=self.on_progress)

    @learning_task.on_start
    def on_start(self):
        self.progressBarInit(None)
        self.topic_desc.clear()

    @learning_task.on_result
    def on_result(self, corpus):
        self.progressBarFinished(None)
        self.Outputs.corpus.send(corpus)
        if corpus is None:
            self.topic_desc.clear()
            self.Outputs.selected_topic.send(None)
            self.Outputs.all_topics.send(None)
        else:
            self.topic_desc.show_model(self.model)
            self.Outputs.all_topics.send(self.model.get_all_topics_table())

    @learning_task.callback
    def on_progress(self, p):
        self.progressBarSet(100 * p, processEvents=None)

    def send_report(self):
        self.report_items(*self.widgets[self.method_index].report_model())
        if self.corpus is not None:
            self.report_items('Topics', self.topic_desc.report())

    def send_topic_by_id(self, topic_id=None):
        if self.model.model and topic_id is not None:
            self.Outputs.selected_topic.send(self.model.get_topics_table_by_id(topic_id))


class TopicViewerTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, topic_id, words, weights, parent,
                 color_by_weights=False):
        super().__init__(parent)
        self.topic_id = topic_id
        self.words = words
        self.weights = weights
        self.color_by_weights = color_by_weights

        self.setText(0, '{:d}'.format(topic_id + 1))
        self.setText(1, ', '.join(self._color(word, weight)
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

    def resize_columns(self):
        for i in range(self.columnCount()):
            self.resizeColumnToContents(i)

    def show_model(self, topic_model):
        self.clear()
        if topic_model.model:
            for i in range(topic_model.num_topics):
                words, weights = topic_model.get_top_words_by_id(i)
                if words:
                    it = TopicViewerTreeWidgetItem(
                        i, words, weights, self,
                        color_by_weights=topic_model.has_negative_weights)
                    self.addTopLevelItem(it)

            self.resize_columns()
            self.setCurrentItem(self.topLevelItem(0))

    def selected_topic_changed(self):
        selected = self.selectedItems()
        if selected:
            topic_id = selected[0].topic_id
            self.setCurrentItem(self.topLevelItem(topic_id))
            self.topicSelected.emit(topic_id)
        else:
            self.topicSelected.emit(None)

    def report(self):
        root = self.invisibleRootItem()
        child_count = root.childCount()
        return [root.child(i).report()
                for i in range(child_count)]

    def sizeHint(self):
        return QSize(700, 300)


class HTMLDelegate(QStyledItemDelegate):
    """ This delegate enables coloring of words in QTreeWidgetItem. 
    Adopted from https://stackoverflow.com/a/5443112/892987 """
    def paint(self, painter, option, index):
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options,index)

        style = QApplication.style() if options.widget is None else options.widget.style()

        doc = QtGui.QTextDocument()
        doc.setHtml(options.text)

        options.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        if options.state & QStyle.State_Selected:
            ctx.palette.setColor(QtGui.QPalette.Text,
                                 options.palette.color(QtGui.QPalette.Active,
                                                       QtGui.QPalette.HighlightedText))

        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)
        painter.save()
        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options,index)

        doc = QtGui.QTextDocument()
        doc.setHtml(options.text)
        doc.setTextWidth(options.rect.width())
        return QtCore.QSize(doc.idealWidth(), doc.size().height())


if __name__ == '__main__':
    from AnyQt.QtWidgets import QApplication

    app = QApplication([])
    widget = OWTopicModeling()
    # widget.set_data(Corpus.from_file('book-excerpts'))
    widget.set_data(Corpus.from_file('deerwester'))
    widget.show()
    app.exec()
    widget.saveSettings()
