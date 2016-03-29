# coding: utf-8
from collections import OrderedDict
from os import path
from math import pi as PI

import numpy as np

from PyQt4 import QtCore, QtGui

from Orange.widgets import widget, gui, settings
from orangecontrib.text.topics import Topics
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import Preprocessor


class SelectedWords(set):
    def __init__(self, widget):
        self.widget = widget

    def _update_webview(self):
        self.widget.webview.evalJS('SELECTED_WORDS = {};'.format(list(self)))

    def _update_filter(self):
        filter = set()
        if self.widget.bow is not None:
            for word in self:
                index = self.widget.PREPROCESS.cv.vocabulary_.get(word)
                if index is None: continue
                filter |= set(self.widget.bow[:, index].nonzero()[0])
        if filter:
            self.widget.send(Output.CORPUS, self.widget.corpus[sorted(filter), :])
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
        self.widget.table.clearSelection()


class Output:
    CORPUS = 'Corpus'


class OWWordCloud(widget.OWWidget):
    name = "Word Cloud"
    priority = 10000
    icon = "icons/WordCloud.svg"
    inputs = [
        ('Topics', Topics, 'on_topics_change'),
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
        self.bow = None  # bag-of-words, obviously
        self.topics = None
        self.corpus = None
        self.PREPROCESS = Preprocessor()
        self._create_layout()

    @QtCore.pyqtSlot(str, result=str)
    def word_clicked(self, word):
        """Called from JavaScript"""
        if not word:
            self.selected_words.clear()
            return ''
        if word not in self.selected_words:
            self.selected_words.add(word)
            self.table.selectRowsWhere(1, word)
            return 'selected'
        else:
            self.selected_words.remove(word)
            self.table.deselectRowsWhere(1, word)
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
        self.webview = gui.WebviewWidget(self.mainArea, self, HTML, debug=True)
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
        self.table = gui.TableWidget(box,
                                     col_labels=['Weight', 'Word'],
                                     multi_selection=True,
                                     select_rows=True)

        def _selection_changed(selected, deselected):
            for index in deselected.indexes():
                data = self.table.rowData(index.row())
                self.selected_words.remove(data)
            for index in selected.indexes():
                data = self.table.rowData(index.row())
                self.selected_words.add(data)
            self.cloud_reselect()

        self.table.selectionChanged = _selection_changed

    def cloud_reselect(self):
        self.webview.evalJS('selectWords();')

    def on_cloud_pref_change(self):
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
        self.table.clear()
        for word, weight in zip(words, weights):
            self.table.addRow((weight, word), data=word)
        self.table.sortByColumn(0, QtCore.Qt.DescendingOrder)
        # Reset wordcloud
        mean = np.mean(weights)
        MIN_SIZE, MEAN_SIZE, MAX_SIZE = 8, 18, 40

        def _size(w):
            return np.clip(w/mean*MEAN_SIZE, MIN_SIZE, MAX_SIZE)

        self.wordlist = [[word, _size(weight)] for word, weight in zip(words, weights)]
        self.webview.evalJS('OPTIONS["list"] = {};'.format(self.wordlist))
        self.on_cloud_pref_change()

    def on_topics_change(self, data):
        self.topics = data
        self.topic_info.setVisible(bool(data))
        if not data and self.corpus:  # Topics aren't, but raw corpus is avaiable
            return self._count_words_in_corpus()
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

    def _get_text_column(self):
        col = None
        try:
            # Take the first string meta variable
            col = next(i for i,var in enumerate(self.corpus.domain.metas) if var.is_string)
            # But prefer any that has name 'text'
            col = next(i for i,var in enumerate(self.corpus.domain.metas)
                       if var.is_string and var.name == 'text')
        except (StopIteration, AttributeError): pass
        return col

    def _bag_of_words_from_corpus(self):
        self.bow = None
        col = self._get_text_column()
        if col is None: return 0, 0
        texts = self.corpus.metas[:, col]
        self.bow = self.PREPROCESS.cv.fit_transform(texts).tocsc()
        return self.bow.shape

    def _count_words_in_corpus(self):
        col = self._get_text_column()
        if col is None: return
        words = self.PREPROCESS.cv.get_feature_names()
        freqs = np.array(self.bow.sum(axis=0))[0]
        self._repopulate_wordcloud(words, freqs)

    def on_corpus_change(self, data):
        self.corpus = data
        n_docs, n_words = self._bag_of_words_from_corpus()
        self.documents_info_str = ('{} documents with {} words'.format(n_docs, n_words)
                                   if n_docs else '(no documents on input)')
        if not self.topics:
            self._count_words_in_corpus()


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
    w.on_topics_change(table)
    domain = Domain([], metas=[StringVariable('text')])
    data = Corpus(None, None, np.array([[' '.join(words.flat)]]), domain)
    # data = Corpus.from_numpy(domain, X=np.zeros((1, 0)), metas=np.array([[' '.join(words.flat)]]))
    w.on_corpus_change(data)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
