# coding: utf-8
from collections import Counter
from math import pi as PI

import numpy as np
from AnyQt.QtCore import Qt, QItemSelection, QItemSelectionModel, pyqtSlot, \
    QObject, QSortFilterProxyModel
from AnyQt.QtWidgets import QApplication

from Orange.data import StringVariable, ContinuousVariable, Domain, Table
from Orange.data.util import scale
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import PyTableModel
from Orange.widgets.widget import Input, Output
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.topics import Topic


class OWWordCloud(widget.OWWidget):
    name = "Word Cloud"
    priority = 510
    icon = "icons/WordCloud.svg"

    class Inputs:
        corpus = Input("Corpus", Corpus, default=True)
        topic = Input("Topic", Topic)

    class Outputs:
        corpus = Output("Corpus", Corpus)
        selected_words = Output("Selected Words", Topic, dynamic=False)
        word_counts = Output("Word Counts", Table)

    graph_name = 'webview'

    selected_words = settings.Setting(set(), schema_only=True)

    words_color = settings.Setting(True)
    words_tilt = settings.Setting(0)

    class Warning(widget.OWWidget.Warning):
        topic_precedence = widget.Msg('Input signal Topic takes priority over Corpus')

    def __init__(self):
        super().__init__()
        self.n_topic_words = 0
        self.documents_info_str = ''
        self.webview = None
        self.topic = None
        self.corpus = None
        self.corpus_counter = None
        self.wordlist = None
        self._create_layout()
        self.on_corpus_change(None)

    def _new_webview(self):
        HTML = '''
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {margin:0px;padding:0px;width:100%;height:100%;
            cursor:default; -webkit-user-select: none; user-select: none; }
span:hover {color:OrangeRed !important}
span.selected {color:red !important}
</style>
</head>
<body id="canvas">
<script src="resources/wordcloud2.js"></script>
<script src="resources/wordcloud-script.js"></script>
</body>
</html>'''
        if self.webview:
            self.mainArea.layout().removeWidget(self.webview)

        class Bridge(QObject):
            @pyqtSlot('QVariantList')
            def update_selection(_, words):
                nonlocal webview
                self.update_selection(words, webview)

        class Webview(gui.WebviewWidget):
            def update_selection(self, words):
                self.evalJS('SELECTED_WORDS = {}; selectWords();'.format(list(words)))

        webview = self.webview = Webview(self.mainArea, Bridge())
        webview.setHtml(HTML, webview.toFileURL(__file__))
        self.mainArea.layout().addWidget(webview)

    def _create_layout(self):
        self._new_webview()
        box = gui.widgetBox(self.controlArea, 'Info')
        self.topic_info = gui.label(box, self, '%(n_topic_words)d words in a topic')
        gui.label(box, self, '%(documents_info_str)s')

        box = gui.widgetBox(self.controlArea, 'Cloud preferences')
        gui.checkBox(box, self, 'words_color', 'Color words', callback=self.on_cloud_pref_change)
        TILT_VALUES = ('no', '30°', '45°', '60°')
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
                self.__nope = False

            def setModel(self, model):
                """Otherwise QTableView.setModel() calls
                QAbstractItemView.setSelectionModel() which resets selection,
                calling selectionChanged() and overwriting any selected_words
                setting that may have been saved."""
                self.__nope = True
                super().setModel(model)
                self.__nope = False

            def selectionChanged(self, selected, deselected):
                nonlocal model, proxymodel
                super().selectionChanged(selected, deselected)
                if not self.__nope:
                    words = {model[proxymodel.mapToSource(index).row()][1]
                             for index in self.selectionModel().selectedIndexes()}
                    self._parent.update_selection(words, self)

            def update_selection(self, words):
                nonlocal model, proxymodel
                selection = QItemSelection()
                for i, (_, word) in enumerate(model):
                    if word in words:
                        index = proxymodel.mapFromSource(model.index(i, 1))
                        selection.select(index, index)
                self.__nope = True
                self.clearSelection()
                self.selectionModel().select(
                    selection,
                    QItemSelectionModel.Select | QItemSelectionModel.Rows)
                self.__nope = False

        view = self.tableview = TableView(self)
        model = self.tablemodel = PyTableModel(parent=self)
        proxymodel = QSortFilterProxyModel(self, dynamicSortFilter=True,
                                           sortCaseSensitivity=Qt.CaseInsensitive,
                                           sortRole=Qt.EditRole)
        proxymodel.setSourceModel(model)
        model.setHorizontalHeaderLabels(['Weight', 'Word'])
        view.setModel(proxymodel)
        box.layout().addWidget(view)

    def on_cloud_pref_change(self):
        if self.wordlist is None:
            return
        self._new_webview()
        self.webview.evalJS('''OPTIONS["color"] = function () {
            return %s[Math.floor(3 * Math.random())];;
        }''' % (["#da1", "#629", "#787"]
                if self.words_color else
                ['#000', '#444', '#777', '#aaa']))
        tilt_ratio, tilt_amount = {
            0: (0, 0),
            1: (1, PI / 6),
            2: (1, PI / 4),
            3: (.67, PI / 3),
        }[self.words_tilt]
        self.webview.evalJS('OPTIONS["minRotation"] = {}; \
                             OPTIONS["maxRotation"] = {};'.format(-tilt_amount, tilt_amount))
        self.webview.evalJS('OPTIONS["rotateRatio"] = {};'.format(tilt_ratio))
        self.webview.evalJS('''
        OPTIONS["gridSize"] = function () {
          return Math.round( 
            Math.min(
              document.getElementById("canvas").clientWidth,
              document.getElementById("canvas").clientHeight
            ) / 48
          );
        };''')
        self.webview.evalJS('''
        OPTIONS["weightFactor"] = function (size) {
          return size * 
            Math.min(
              document.getElementById("canvas").clientWidth,
              document.getElementById("canvas").clientHeight
            ) / 512;
        };''')
        # Trigger cloud redrawing by constructing new webview, because everything else fail Macintosh
        self.webview.evalJS('OPTIONS["list"] = {};'.format(self.wordlist))
        self.webview.evalJS('redrawWordCloud();')
        self.webview.update_selection(self.selected_words)

    def _repopulate_wordcloud(self, words, weights):
        N_BEST = 200
        words, weights = words[:N_BEST], weights[:N_BEST]
        # Repopulate table
        self.tablemodel.wrap(list(zip(weights, words)))
        self.tableview.sortByColumn(0, Qt.DescendingOrder)
        # Reset wordcloud
        weights = np.clip(weights, *np.percentile(weights, [2, 98]))
        weights = scale(weights, 8, 40)
        self.wordlist = np.c_[words, weights].tolist()
        self.on_cloud_pref_change()

    @Inputs.topic
    def on_topic_change(self, data):
        self.topic = data
        self.topic_info.setVisible(data is not None)

    def _apply_topic(self):
        data = self.topic
        metas = data.domain.metas if data else []
        try:
            col = next(i for i,var in enumerate(metas) if var.is_string)
        except StopIteration:
            words = np.zeros((0, 1))
        else:
            words = data.metas[:, col]
            self.n_topic_words = data.metas.shape[0]
        if data and data.W.any():
            weights = data.W[:]
        elif data and 'weights' in data.domain:
            weights = data.get_column_view(data.domain['weights'])[0]
        else:
            weights = np.ones(len(words))

        self._repopulate_wordcloud(words, weights)

    def _apply_corpus(self):
        words, freq = self.word_frequencies()
        self._repopulate_wordcloud(words, freq)

    def word_frequencies(self):
        counts = self.corpus_counter.most_common()
        words, freq = zip(*counts) if counts else ([], [])
        return words, freq

    def create_weight_list(self):
        wc_table = None
        if self.corpus is not None:
            words, freq = self.word_frequencies()
            words = np.array(words)[:, None]
            w_count = np.array(freq)[:, None]
            domain = Domain([ContinuousVariable('Word Count')],
                            metas=[StringVariable('Word')])
            wc_table = Table.from_numpy(domain, X=w_count, metas=words)
            wc_table.name = 'Word Counts'
        self.Outputs.word_counts.send(wc_table)

    @Inputs.corpus
    def on_corpus_change(self, data):
        self.corpus = data

        self.corpus_counter = Counter()
        if data is not None:
            self.corpus_counter = Counter(w for doc in data.ngrams for w in doc)
            n_docs, n_words = len(data), len(self.corpus_counter)

        self.documents_info_str = ('{} documents with {} words'.format(n_docs, n_words)
                                   if data else '(no documents on input)')

        self.create_weight_list()

    def handleNewSignals(self):
        if self.topic is not None and len(self.topic):
            self._apply_topic()
        elif self.corpus is not None and len(self.corpus):
            self._apply_corpus()
        else:
            self.clear()
            return

        self.Warning.topic_precedence(
            shown=self.corpus is not None and self.topic is not None)

        if self.topic is not None or self.corpus is not None:
            if self.selected_words:
                self.update_selection(self.selected_words)

    def clear(self):
        self._new_webview()
        self.tablemodel.clear()
        self.wordlist = None
        self.commit()

    def update_selection(self, words, skip=None):
        assert skip is None or skip in (self.webview, self.tableview)
        self.selected_words = words = set(words)

        if self.tableview != skip:
            self.tableview.update_selection(words)
        if self.webview != skip:
            self.webview.update_selection(words)

        self.commit()

    def commit(self):
        out = None
        if self.corpus is not None:
            rows = [i for i, doc in enumerate(self.corpus.ngrams)
                    if any(word in doc for word in self.selected_words)]
            out = self.corpus[rows]
        self.Outputs.corpus.send(out)

        topic = None
        words = list(self.selected_words)
        if words:
            topic = Topic.from_numpy(Domain([], metas=[StringVariable('Words')]),
                                     X=np.empty((len(words), 0)),
                                     metas=np.c_[words].astype(object))
            topic.name = 'Selected Words'
        self.Outputs.selected_words.send(topic)

    def send_report(self):
        html = self.webview.html()
        start = html.index('>', html.index('<body')) + 1
        end = html.index('</body>')
        body = html[start:end]
        # create an empty div of appropriate height to compensate for
        # absolute positioning of words in the html
        height = self.webview._evalJS("document.getElementById('canvas').clientHeight")
        self.report_html += '<div style="position: relative; height: {}px;">{}</div>'.format(
            height, body)

        self.report_table(self.tableview)


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
    app = QApplication([''])
    w = OWWordCloud()
    w.on_topic_change(table)
    domain = Domain([], metas=[StringVariable('text')])
    data = Corpus(domain=domain, metas=np.array([[' '.join(words.flat)]]))
    # data = Corpus.from_numpy(domain, X=np.zeros((1, 0)), metas=np.array([[' '.join(words.flat)]]))
    w.on_corpus_change(data)
    w.handleNewSignals()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
