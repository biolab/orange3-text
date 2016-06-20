from PyQt4 import QtCore

from PyQt4.QtGui import (QApplication, QRadioButton, QVBoxLayout, QButtonGroup,
                         QGridLayout, QLabel, QWidget)

from Orange.widgets import gui
from Orange.widgets.gui import OWComponent
from Orange.widgets import settings
from Orange.widgets.widget import OWWidget
from orangecontrib.text import bagofowords
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.utils import widgets


class VectorizerWidget(QWidget, OWComponent):
    Vectorizer = NotImplementedError
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, widget, *args):
        QWidget.__init__(self, *args)
        OWComponent.__init__(self, widget)
        self.vectorizer = self.Vectorizer()

    def on_change(self):
        self.valueChanged.emit(self)

    @property
    def name(self):
        return self.Vectorizer.name


class CountVectorizerWidget(VectorizerWidget):
    binary = settings.Setting(False)
    ngram_range = settings.Setting((1, 1))

    Vectorizer = bagofowords.CountVectorizer

    def __init__(self, widget, *args):
        super().__init__(widget, *args)
        layout = QVBoxLayout(self)

        range_box = widgets.RangeWidget(self, 'ngram_range', minimum=1, maximum=10, step=1,
                                        min_label='N-grams range:', dtype=int)
        layout.addWidget(range_box)

        box = gui.checkBox(None, self, 'binary', 'Binary',
                           callback=self.on_change)
        layout.addWidget(box)

    def on_change(self):
        self.vectorizer = self.Vectorizer(binary=self.binary)
        super().on_change()


class TfidfVectorizerWidget(VectorizerWidget):
    ngram_range = settings.Setting((1, 1))
    binary = settings.Setting(False)
    normalization = settings.Setting('l2')
    use_idf = settings.Setting(True)
    smooth_idf = settings.Setting(True)
    sublinear_tf = settings.Setting(False)

    Vectorizer = bagofowords.TfidfVectorizer

    def __init__(self, widget, *args):
        super().__init__(widget, *args)
        layout = QGridLayout(self)
        layout.setSpacing(10)
        row = 0
        range_box = widgets.RangeWidget(self, 'ngram_range', minimum=1, maximum=10, step=1,
                                        dtype=int)
        layout.addWidget(QLabel('N-grams range:'), row, 0)
        layout.addWidget(range_box, row, 1)

        row += 1
        box = gui.checkBox(None, self, 'binary', 'Binary', callback=self.on_change)
        layout.addWidget(box, row, 0, 1, 2)

        row += 1
        combo = widgets.ComboBox(self, 'normalization',
                                 (('l1 norm', 'l1'), ('l2 norm', 'l2')))
        combo.currentIndexChanged.connect(self.on_change)
        layout.addWidget(QLabel('Regularization:'))
        layout.addWidget(combo, row, 1)

        row += 1
        box = gui.checkBox(None, self, 'use_idf', 'Use idf',
                           callback=self.on_change)
        layout.addWidget(box, row, 0, 1, 2)

        row += 1
        box = gui.checkBox(None, self, 'smooth_idf', 'Smooth idf',
                           callback=self.on_change)
        layout.addWidget(box, row, 0, 1, 2)

        row += 1
        box = gui.checkBox(None, self, 'sublinear_tf', 'Sublinear tf',
                           callback=self.on_change)
        layout.addWidget(box, row, 0, 1, 2)

    def on_change(self):
        self.vectorizer = self.Vectorizer(
            binary=self.binary,
            norm=self.normalization,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf
        )
        super().on_change()


class SimhashVectorizerWidget(VectorizerWidget):
    Vectorizer = bagofowords.SimhashVectorizer
    f = settings.Setting(64)
    shingle_len = settings.Setting(10)

    def __init__(self, widget, *args):
        super().__init__(widget, *args)
        layout = QVBoxLayout(self)

        spin = gui.spin(self, self, 'f', minv=1,
                        maxv=bagofowords.SimhashVectorizer.max_f,
                        label='Simhash size:')
        spin.editingFinished.connect(self.on_change)

        spin = gui.spin(self, self, 'shingle_len', minv=1, maxv=100,
                        label='Shingle length:')
        spin.editingFinished.connect(self.on_change)

    def on_change(self):
        self.vectorizer = self.Vectorizer(f=self.f, shingle_len=self.shingle_len)
        super().on_change()


class Input:
    CORPUS = 'Corpus'


class Output:
    CORPUS = 'Corpus'


class OWBagOfWords(OWWidget):
    name = 'Bag of Words'
    description = 'Generates a bag of words from the input corpus.'
    icon = 'icons/BagOfWords.svg'
    priority = 40

    # Input/output
    inputs = [
        (Input.CORPUS, Corpus, 'set_data'),
    ]
    outputs = [
        (Output.CORPUS, Corpus)
    ]

    want_main_area = True
    buttons_area_orientation = QtCore.Qt.Vertical
    control_area_width = 180

    method_widgets = [
        (CountVectorizerWidget, 'count'),
        (TfidfVectorizerWidget, 'tfidf'),
        (SimhashVectorizerWidget, 'simhash'),
    ]
    count = settings.SettingProvider(CountVectorizerWidget)
    tfidf = settings.SettingProvider(TfidfVectorizerWidget)
    simhash = settings.SettingProvider(SimhashVectorizerWidget)

    COUNT, TFIDF = 0, 1

    # Settings
    autocommit = settings.Setting(True)
    method_index = settings.Setting(0)

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.output = None

        self.mainArea.layout().setMargin(10)
        self.controlArea.layout().setMargin(10)

        # Info box.
        info_box = gui.widgetBox(self.controlArea, 'Info')
        info_box.setFixedWidth(self.control_area_width)
        self.info_label = gui.label(info_box, self, '')
        self.update_info()
        self.controlArea.layout().addStretch()

        self.group = QButtonGroup()
        self.group.buttonClicked.connect(self.on_change)

        method_layout = QVBoxLayout()

        self.methods = []
        for i, (Widget, attr_name) in enumerate(self.method_widgets):
            widget = Widget(self)
            setattr(self, attr_name, widget)
            rb = QRadioButton(self, text=widget.name)
            rb.setChecked(i == self.method_index)
            self.group.addButton(rb, i)
            widget.valueChanged.connect(self.vectorizer_changed)
            method_layout.addWidget(rb)
            method_layout.addWidget(widget)
            method_layout.addSpacing(10)
            self.methods.append(widget)

        method_layout.addStretch()
        self.mainArea.layout().addLayout(method_layout)

        self.report_button.setFixedWidth(self.control_area_width)
        commit_button = gui.auto_commit(self.buttonsArea, self, 'autocommit',
                                        'Commit', box=False)
        commit_button.setFixedWidth(self.control_area_width - 5)

    @property
    def method(self):
        return self.methods[self.method_index]

    def set_data(self, data):
        self.corpus = data
        self.output = None
        self.update_info()
        self.commit()

    def commit(self):
        self.apply()

    def apply(self):
        self.error(0, '')
        if self.corpus is not None:
            vectorizer = self.method.vectorizer
            new_corpus = vectorizer.fit_transform(self.corpus)
            self.output = new_corpus
            self.send(Output.CORPUS, new_corpus)

    def show_errors(self, error):
        self.error(0, '')
        self.error(0, str(error))

    def on_change(self):
        self.method_index = self.group.checkedId()
        self.commit()

    def vectorizer_changed(self, vectorizer):
        if vectorizer is self.method:
            self.on_change()

    def update_info(self):
        if self.corpus is None:
            new_info = 'No info available.'
        else:
            new_info = '{} documents.\n{} unique tokens.'.format(
                len(self.corpus),
                len(self.corpus.dictionary),
            )
        self.info_label.setText(new_info)

    def send_report(self):
        self.report_items(self.method.name, self.method.vectorizer.report())
        if self.corpus is not None:
            self.report_items('Corpus', (('Documents', len(self.corpus)),
                                         ('Unique tokens', len(self.corpus.dictionary))))
        if self.output is not None:
            self.report_items('Output', (('Attributes', len(self.output.domain)),))


if __name__ == '__main__':
    app = QApplication([])
    widget = OWBagOfWords()
    widget.show()
    corpus = Corpus.from_file('bookexcerpts')
    widget.set_data(corpus)
    app.exec()
