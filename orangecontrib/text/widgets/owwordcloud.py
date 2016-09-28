# coding: utf-8
from collections import Counter
from math import pi as PI
from os import path

import numpy as np
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QItemSelection, QItemSelectionModel

from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import PyTableModel
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import Topic


class SelectedWords(set):
    def __init__(self, widget):
        self.widget = widget

    def _update_webview(self):
        self.widget.webview.evalJS('SELECTED_WORDS = {};'.format(list(self)))

    def _update_filter(self):
        filter = set()
        if self.widget.corpus is not None:
            for i, doc in enumerate(self.widget.corpus.ngrams):
                if any(i in doc for i in self):
                    filter.add(i)
        if filter:
            self.widget.send(Output.CORPUS, self.widget.corpus[list(filter), :])
        else:
            self.widget.send(Output.CORPUS, None)

    def add(self, word):
        if word not in self:
            super().add(word)
            self._update_webview()
            self._update_filter()

    def remove(self, word):
        if word in self:
            super().remove(word)
            self._update_webview()
            self._update_filter()

    def clear(self):
        super().clear()
        self._update_webview()
        self._update_filter()
        self.widget.tableview.clearSelection()


class Output:
    CORPUS = 'Corpus'
    TOPIC = 'Topic'


class OWWordCloud(widget.OWWidget):
    name = "Word Cloud"
    priority = 10000
    icon = "icons/WordCloud.svg"
    inputs = [
        (Output.TOPIC, Topic, 'on_topic_change'),
        (Output.CORPUS, Corpus, 'on_corpus_change'),
    ]
    outputs = [('Corpus', Corpus)]

    graph_name = 'webview'

    selected_words = settings.ContextSetting(SelectedWords('whatevar (?)'))
    words_color = settings.Setting(True)
    words_tilt = settings.Setting(0)

    def __init__(self):
        super().__init__()
        self.n_topic_words = 0
        self.documents_info_str = ''
        self.selected_words = SelectedWords(self)
        self.webview = None
        self.topic = None
        self.corpus = None
        self.corpus_counter = None
        self.wordlist = None
        self._create_layout()
        self.on_corpus_change(None)

    @QtCore.pyqtSlot(str, result=str)
    def word_clicked(self, word):
        """Called from JavaScript"""
        if not word:
            self.selected_words.clear()
            return ''
        selection = QItemSelection()
        for i, row in enumerate(self.tablemodel):
            for j, val in enumerate(row):
                if val == word:
                    index = self.tablemodel.index(i, j)
                    selection.select(index, index)
        if word not in self.selected_words:
            self.selected_words.add(word)
            self.tableview.selectionModel().select(
                selection, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            return 'selected'
        else:
            self.selected_words.remove(word)
            self.tableview.selectionModel().select(
                selection, QItemSelectionModel.Deselect | QItemSelectionModel.Rows)
            return ''

    def _new_webview(self):
        HTML = '''
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {margin:0px;padding:0px;width:100%;height:100%;}
span:hover {color:OrangeRed !important}
span.selected {color:red !important}
</style>
</head>
<body id="canvas"></body>
</html>'''
        if self.webview:
            self.mainArea.layout().removeWidget(self.webview)
        webview = self.webview = gui.WebviewWidget(self.mainArea, self, debug=False)
        webview.setHtml(HTML)
        self.mainArea.layout().addWidget(webview)
        for script in ('wordcloud2.js',
                       'wordcloud-script.js'):
            self.webview.evalJS(open(path.join(path.dirname(__file__), 'resources', script), encoding='utf-8').read())

    def _create_layout(self):
        self._new_webview()
        box = gui.widgetBox(self.controlArea, 'Info')
        self.topic_info = gui.label(box, self, '%(n_topic_words)d words in a topic')
        gui.label(box, self, '%(documents_info_str)s')

        box = gui.widgetBox(self.controlArea, 'Cloud preferences')
        gui.checkBox(box, self, 'words_color', 'Color words', callback=self.on_cloud_pref_change)
        TILT_VALUES = ('no', 'slight', 'more', 'full')
        gui.valueSlider(box, self, 'words_tilt', label='Words tilt:',
                        values=list(range(len(TILT_VALUES))),
                        callback=self.on_cloud_pref_change,
                        labelFormat=lambda x: TILT_VALUES[x])
        gui.button(box, None, 'Regenerate word cloud', callback=self.on_cloud_pref_change)

        box = gui.widgetBox(self.controlArea, 'Words && weights')

        class TableView(gui.TableView):
            def __init__(self, parent):
                super().__init__(parent)
                self._parent = parent

            def selectionChanged(self, selected, deselected):
                super().selectionChanged(selected, deselected)
                parent = self._parent
                for index in deselected.indexes():
                    data = parent.tablemodel[index.row()][1]
                    self._parent.selected_words.remove(data)
                for index in selected.indexes():
                    data = parent.tablemodel[index.row()][1]
                    self._parent.selected_words.add(data)
                parent.cloud_reselect()

        view = self.tableview = TableView(self)
        model = self.tablemodel = PyTableModel()
        model.setHorizontalHeaderLabels(['Weight', 'Word'])
        view.setModel(model)
        box.layout().addWidget(view)

    def cloud_reselect(self):
        self.webview.evalJS('selectWords();')

    def on_cloud_pref_change(self):
        if self.wordlist is None:
            return
        self._new_webview()
        self.webview.evalJS('OPTIONS["color"] = "{}"'.format(
            'random-dark' if self.words_color else 'black'))
        tilt_ratio, tilt_amount = {
            0: (0, 0),
            1: (.5,  PI/12),
            2: (.75, PI/5),
            3: (.9,  PI/2),
        }[self.words_tilt]
        self.webview.evalJS('OPTIONS["minRotation"] = {}; \
                             OPTIONS["maxRotation"] = {};'.format(-tilt_amount, tilt_amount))
        self.webview.evalJS('OPTIONS["rotateRatio"] = {};'.format(tilt_ratio))
        # Trigger cloud redrawing by constructing new webview, because everything else fail Macintosh
        self.webview.evalJS('OPTIONS["list"] = {};'.format(self.wordlist))
        self.webview.evalJS('redrawWordCloud();')

    def _repopulate_wordcloud(self, words, weights):
        N_BEST = 200
        words, weights = words[:N_BEST], weights[:N_BEST]
        # Repopulate table
        self.tablemodel.clear()
        for word, weight in zip(words, weights):
            self.tablemodel.append([weight, word])
        self.tableview.sortByColumn(0, Qt.DescendingOrder)
        # Reset wordcloud
        mean = np.mean(weights)
        MIN_SIZE, MEAN_SIZE, MAX_SIZE = 8, 18, 40

        def _size(w):
            return np.clip(w/mean*MEAN_SIZE, MIN_SIZE, MAX_SIZE)

        self.wordlist = [[word, _size(weight)] for word, weight in zip(words, weights)]
        self.on_cloud_pref_change()

    def on_topic_change(self, data):
        self.topic = data
        self.topic_info.setVisible(bool(data))
        if not data and self.corpus:  # Topics aren't, but raw corpus is available
            return self._apply_corpus()
        metas = data.domain.metas if data else []
        try:
            col = next(i for i,var in enumerate(metas) if var.is_string)
        except StopIteration:
            words = np.zeros((0, 1))
        else:
            words = data.metas[:, col]
            self.n_topic_words = data.metas.shape[0]
        if not data:
            weights = np.ones(len(words))
        if data and data.W.any():
            weights = data.W[:]
        elif data and 'weights' in data.domain:
            weights = data.get_column_view(data.domain['weights'])[0]
        else:
            weights = np.ones(len(words))

        self._repopulate_wordcloud(words, weights)

    def _apply_corpus(self):
        counts = self.corpus_counter.most_common()
        words, freq = zip(*counts) if counts else ([], [])
        self._repopulate_wordcloud(words, freq)

    def on_corpus_change(self, data):
        self.corpus = data

        self.corpus_counter = Counter()
        if data is not None:
            self.corpus_counter = Counter(w for doc in data.ngrams for w in doc)
            n_docs, n_words = len(data), len(self.corpus_counter)

        self.documents_info_str = ('{} documents with {} words'.format(n_docs, n_words)
                                   if data else '(no documents on input)')
        if not self.topic:
            self._apply_corpus()


def main():
    from Orange.data import Table, Domain, ContinuousVariable, StringVariable

    words = 'hey~mr. tallyman tally~me banana daylight come and me wanna go home'
    words = np.array([w.replace('~', ' ') for w in words.split()], dtype=object, ndmin=2).T
    weights = np.random.random((len(words), 1))

    data = np.zeros((len(words), 0))
    metas = []
    for i, w in enumerate(weights.T):
        data = np.column_stack((data, words, w))
        metas = metas + [StringVariable('Topic' + str(i)),
                         ContinuousVariable('weights')]
    domain = Domain([], metas=metas)
    table = Table.from_numpy(domain,
                             X=np.zeros((len(words), 0)),
                             metas=data)
    app = QtGui.QApplication([''])
    w = OWWordCloud()
    w.on_topic_change(table)
    domain = Domain([], metas=[StringVariable('text')])
    data = Corpus(domain=domain, metas=np.array([[' '.join(words.flat)]]))
    # data = Corpus.from_numpy(domain, X=np.zeros((1, 0)), metas=np.array([[' '.join(words.flat)]]))
    w.on_corpus_change(data)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
