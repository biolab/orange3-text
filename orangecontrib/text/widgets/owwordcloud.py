# coding: utf-8
from collections import OrderedDict
from os import path
from math import pi as PI

import numpy as np

from PyQt4 import QtCore, QtGui

from Orange.widgets import widget, gui, settings
from orangecontrib.text.topics import Topics


class SelectedWords(set):
    def __init__(self, widget):
        self.widget = widget

    def _update_webview(self):
        self.widget.webview.evalJS('SELECTED_WORDS = {};'.format(list(self)))

    def add(self, item):
        if item not in self:
            super().add(item)
            self._update_webview()

    def remove(self, item):
        if item in self:
            super().remove(item)
            self._update_webview()

    def clear(self):
        super().clear()
        self._update_webview()
        self.widget.table.clearSelection()


class OWWordCloud(widget.OWWidget):
    name = "Word Cloud"
    priority = 10000
    icon = "icons/WordCloud.svg"
    inputs = [("Topics", Topics, "on_topic_change")]
    outputs = []

    selected_words = settings.ContextSetting(SelectedWords('whatevar (?)'))
    words_color = settings.Setting(True)
    words_tilt = settings.Setting(1)

    def __init__(self):
        super().__init__()
        self.n_words = 0
        self.mean_weight = 0
        self.selected_words = SelectedWords(self)
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

    def _create_layout(self):
        html = '''
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
<body id="canvas">&nbsp;
</body>
</html>'''
        self.webview = webview = gui.WebviewWidget(self.mainArea, self, html)
        for script in ('wordcloud2.js',
                       'wordcloud-script.js'):
            self.webview.evalJS(open(path.join(path.dirname(__file__), 'resources', script), encoding='utf-8').read())

        box = gui.widgetBox(self.controlArea, 'Info')
        gui.label(box, self, '%(n_words)d words')
        gui.label(box, self, 'Mean weight: %(mean_weight).4f')

        box = gui.widgetBox(self.controlArea, 'Cloud preferences')
        gui.checkBox(box, self, 'words_color', 'Color words', callback=self.on_cloud_pref_change)
        TILT_VALUES = ('no', 'slight', 'more', 'full')
        gui.valueSlider(box, self, 'words_tilt', label='Words tilt:',
                        values=list(range(len(TILT_VALUES))),
                        callback=self.on_cloud_pref_change,
                        labelFormat=lambda x: TILT_VALUES[x])
        gui.button(box, None, 'Regenerate word cloud', callback=self.cloud_redraw)

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

    def cloud_redraw(self):
        self.webview.evalJS('redrawWordCloud();')

    def cloud_reselect(self):
        self.webview.evalJS('selectWords();')

    def on_cloud_pref_change(self):
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
        self.cloud_redraw()

    def _repopulate_table(self, words, weights):
        self.table.clear()
        for word, weight in zip(words, weights):
            self.table.addRow((weight, word), data=word)
        self.table.sortByColumn(0, QtCore.Qt.DescendingOrder)

    def _repopulate_wordcloud(self, words, weights):
        self.mean_weight = mean = np.mean(weights)
        MIN_SIZE, MEAN_SIZE, MAX_SIZE = 8, 18, 40

        def _size(w):
            return np.clip(w/mean*MEAN_SIZE, MIN_SIZE, MAX_SIZE)

        wordlist = [[word, _size(weight)] for word, weight in zip(words, weights)]
        self.webview.evalJS('OPTIONS["list"] = {};'.format(wordlist))
        self.on_cloud_pref_change()

    def on_topic_change(self, data):
        self.data = data
        metas = data.domain.metas if data else []
        try: self.topic = next(i for i,var in enumerate(metas) if var.is_string)
        except StopIteration:
            self.topic = None
        col = self.topic
        if col is None:
            return
        N_BEST = 200
        words = self.data.metas[:N_BEST, col] if self.topic is not None else np.zeros((0, 1))
        if self.data.W.any():
            weights = self.data.W[:N_BEST]
        elif 'weights' in self.data.domain:
            weights = self.data.get_column_view(self.data.domain['weights'])[0][:N_BEST]
        else:
            weights = np.ones(N_BEST)
        self.n_words = words.shape[0]
        self._repopulate_table(words, weights)
        self._repopulate_wordcloud(words, weights)


def main():
    from Orange.data import Table, Domain, ContinuousVariable, StringVariable

    words = 'hey~mr. tallyman tally~me banana daylight come and me wanna go home'
    words = np.array([w.replace('~', ' ') for w in words.split()], dtype=object, ndmin=2).T
    weights = np.random.random((len(words), 3))

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
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
