from PyQt4 import QtGui
from PyQt4.QtGui import QVBoxLayout, QButtonGroup, QRadioButton
from PyQt4 import QtCore
from PyQt4.QtCore import Qt, QMetaObject, Q_ARG
from Orange.widgets.widget import OWWidget
from Orange.widgets import settings
from Orange.widgets import gui
from Orange.data import Table
from Orange.widgets.data.contexthandlers import DomainContextHandler
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import Topics, LdaWrapper, HdpWrapper, LsiWrapper


class TopicWidget(gui.OWComponent, QtGui.QGroupBox):
    Model = NotImplemented
    valueChanged = QtCore.pyqtSignal(object)

    parameters = ()
    spin_format = '{description}:'

    def __init__(self, master, **kwargs):
        QtGui.QGroupBox.__init__(self, **kwargs)
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

    spin_format = '{description}: ({parameter})'
    parameters = (
        ('gamma', 'First level concentration', .1, 10, .5, float),
        ('alpha', 'Second level concentration', 1, 10, 1, int),
        ('eta', 'The topic Dirichlet', 0.001, .5, .01, float),
        ('T', 'Top level truncation level', 10, 150, 1, int),
        ('K', 'Second level truncation level', 1, 50, 1, int),
        ('kappa', 'Learning rate', .1, 10., .1, float),
        ('tau', 'Slow down parameter', 16., 256., 1., float),
    )
    gamma = settings.Setting(1)
    alpha = settings.Setting(1)
    eta = settings.Setting(.01)
    T = settings.Setting(150)
    K = settings.Setting(15)
    kappa = settings.Setting(1)
    tau = settings.Setting(64)


class Output:
    CORPUS = "Corpus"
    TOPIC = "Topic"


class OWTopicModeling(OWWidget):
    name = "Topic Modelling"
    description = "Uncover the hidden thematic structure in a corpus."
    icon = "icons/TopicModeling.svg"
    priority = 50

    settingsHandler = DomainContextHandler()

    # Input/output
    inputs = [("Corpus", Corpus, "set_data")]
    outputs = [(Output.CORPUS, Table),
               (Output.TOPIC, Topics)]
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
        self.apply_mutex = QtCore.QMutex()
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

    def update_topics(self):
        self.topic_desc.show_model(self.model)

    @QtCore.pyqtSlot(int)
    def progress(self, p):
        self.progressBarSet(p, processEvents=None)

    def apply(self):
        self.stop_learning()
        if self.corpus:
            self.start_learning()
        else:
            self.send(Output.CORPUS, None)
            self.send(Output.TOPIC, None)

    def start_learning(self):
        self.topic_desc.clear()
        if self.corpus:
            self.progressBarInit()

            def progress_callback(i):
                QMetaObject.invokeMethod(self, "progress", Qt.QueuedConnection, Q_ARG(int, i))

            self.learning_thread = LearningThread(self.model, self.corpus.copy(),
                                                  result_callback=self.send_corpus,
                                                  progress_callback=progress_callback)
            self.learning_thread.finished.connect(self.learning_finished)
            self.learning_thread.start()

    @QtCore.pyqtSlot()
    def stop_learning(self):

        if self.learning_thread and self.learning_thread.isRunning():
            self.learning_thread.terminate()
            self.learning_thread.wait()
            self.progressBarFinished()

    def send_corpus(self, corpus):
        self.send(Output.CORPUS, corpus)

    def learning_finished(self):
        self.update_topics()
        self.progressBarFinished()

    def send_report(self):
        self.report_items(*self.widgets[self.method_index].report_model())
        if self.corpus:
            self.report_items('Topics', [(i, self.model.get_top_words_by_id(i))
                                         for i in range(len(self.model.topic_names))])

    def send_topic_by_id(self, topic_id=None):
        if self.model.model and topic_id is not None:
            self.send(Output.TOPIC, self.model.get_topics_table_by_id(topic_id))


class TopicViewerTreeWidgetItem(QtGui.QTreeWidgetItem):
    def __init__(self, topic_id, words, parent):
        super().__init__(parent)
        self.topic_id = topic_id
        self.words = words

        self.setText(0, '{:d}'.format(topic_id + 1))
        self.setText(1, ', '.join(words))


class TopicViewer(QtGui.QTreeWidget):
    """ Just keeps stuff organized. Holds topic visualization widget and related functions.

    """

    columns = ['Topic', 'Topic keywords']
    topicSelected = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.setColumnCount(len(self.columns))
        self.setHeaderLabels(self.columns)
        self.resize_columns()
        self.itemSelectionChanged.connect(self.selected_topic_changed)

    def resize_columns(self):
        for i in range(self.columnCount()):
            self.resizeColumnToContents(i)

    def show_model(self, topic_model):
        self.clear()
        if topic_model.model:
            for i in range(topic_model.num_topics):
                words = topic_model.get_top_words_by_id(i)
                if words:
                    it = TopicViewerTreeWidgetItem(i, words, self)
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


class LearningThread(QtCore.QThread):
    def __init__(self, model, corpus, result_callback, **kwargs):
        super().__init__()
        self.result_callback = result_callback
        self.model = model
        self.corpus = corpus
        self.kwargs = kwargs

    def run(self):
        result = self.model.fit_transform(self.corpus, **self.kwargs)
        self.result_callback(result)

    def terminate(self):
        self.model.running = False


if __name__ == '__main__':
    from PyQt4.QtGui import QApplication

    app = QApplication([])
    widget = OWTopicModeling()
    # widget.set_data(Corpus.from_file('bookexcerpts'))
    widget.set_data(Corpus.from_file('deerwester'))
    widget.show()
    app.exec()
    widget.saveSettings()
