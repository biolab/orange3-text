# coding: utf-8
from collections import defaultdict, OrderedDict
from os import path
from math import pi as PI

import numpy as np

from PyQt4 import QtCore, QtGui, QtWebKit

from Orange.widgets import widget, gui, settings
from Orange.data import Table


JS_WORDCLOUD = open(path.join(path.dirname(__file__), 'wordcloud2.js'), encoding='utf-8').read()
JS_SCRIPT = open(path.join(path.dirname(__file__), 'wordcloud_script.js'), encoding='utf-8').read()


class WebviewWidget(QtWebKit.QWebView):
    def __init__(self, parent, bridge, debug=False):
        """
        Parameters
        ----------
        parent: QObject
            Parent QObject, classic QT.
        bridge: QObject
            The "bridge" object exposed as ``window.pybridge`` in JavaScript.
        debug: bool
            If True, enable context menu and inspector.
        """
        super().__init__(parent)
        self._bridge = bridge
        parent.layout().addWidget(self)
        settings = self.settings()
        settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)
        if debug:
            settings.setAttribute(settings.DeveloperExtrasEnabled, True)
        else:
            self.setContextMenuPolicy(QtCore.Qt.NoContextMenu)

    def setContent(self, *args):
        super().setContent(*args)
        if self._bridge:
            self.page().mainFrame().addToJavaScriptWindowObject('pybridge', self._bridge)

    def sizeHint(self):
        return QtCore.QSize(600, 500)

    def evalJS(self, javascript):
        self.page().mainFrame().evaluateJavaScript(javascript)


class SimpleTableWidget(QtGui.QTableWidget):
    """ A wrapper around QTableWidget """

    ROW_DATA_ROLE = QtCore.Qt.UserRole + 131
    ITEM_DATA_ROLE = QtCore.Qt.UserRole + 132

    class TableWidgetNumericItem(QtGui.QTableWidgetItem):
        """TableWidgetItem that sorts numbers correctly!"""
        def __lt__(self, other):
            return (self.data(SimpleTableWidget.ITEM_DATA_ROLE) <
                    other.data(SimpleTableWidget.ITEM_DATA_ROLE))

    def _update_headers(func):
        """ Decorator to update certain table features after method calls
            because Qt sucks ass.
        """
        def _f(self, *args, **kwargs):
            func(self, *args, **kwargs)
            if self.col_labels is not None:
                self.setHorizontalHeaderLabels(self.col_labels)
            if self.row_labels is not None:
                self.setVerticalHeaderLabels(self.row_labels)
            if self.stretch_last_section:
                self.horizontalHeader().setStretchLastSection(True)
        return _f

    @_update_headers
    def __init__(self,
                 parent=None,
                 col_labels=None,
                 row_labels=None,
                 stretch_last_section=True,
                 multi_selection=False,
                 select_rows=False):
        """`callback` is a function that accepts first selected row item"""
        super().__init__(parent)
        if parent and hasattr(parent, 'layout') and parent.layout():
            parent.layout().addWidget(self)
        self.col_labels = col_labels
        self.row_labels = row_labels
        self.stretch_last_section = stretch_last_section
        if col_labels is None:
            self.horizontalHeader().setVisible(False)
        if row_labels is None:
            self.verticalHeader().setVisible(False)
        if multi_selection:
            self.setSelectionMode(self.MultiSelection)
        if select_rows:
            self.setSelectionBehavior(self.SelectRows)
        # Default preferences
        self.setHorizontalScrollMode(self.ScrollPerPixel)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.setEditTriggers(self.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setSortingEnabled(True)

    @_update_headers
    def add(self, items, data=None):
        """Appends iterable of `items` as the next row."""
        row_data = data
        row = self.rowCount()
        self.insertRow(row)
        col_count = max(len(items), self.columnCount())
        if col_count != self.columnCount():
            self.setColumnCount(col_count)
        for col, item_data in enumerate(items):
            if isinstance(item_data, str):
                name = item_data
            elif hasattr(item_data, '__iter__') and len(item_data) == 2:
                name, item_data = item_data
            elif isinstance(item_data, float):
                name = '{:.4f}'.format(item_data)
            else:
                name = str(item_data)
            if isinstance(item_data, (float, int)):
                item = self.TableWidgetNumericItem(name)
            else:
                item = QtGui.QTableWidgetItem(name)
            item.setData(self.ITEM_DATA_ROLE, item_data)
            self.setItem(row, col, item)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        if row_data is not None:
            self.setRowData(row, row_data)

    def rowData(self, row):
        return self.item(row, 0).data(self.ROW_DATA_ROLE)

    def setRowData(self, row, data):
        self.item(row, 0).setData(self.ROW_DATA_ROLE, data)

    def clear(self):
        super().clear()
        self.setRowCount(0)
        self.setColumnCount(0)

    def selectFirstRow(self):
        if self.rowCount() > 0:
            self.selectRow(0)

    def selectRowsWhere(self, col, value, n_hits=-1,
                        flags=QtCore.Qt.MatchExactly, _select=True):
        """
        Selects (also return) at most `n_hits` rows where column `col`
        has value (``data()``) `value`.
        """
        model = self.model()
        matches = model.match(model.index(0, col),
                              self.ITEM_DATA_ROLE,
                              value,
                              n_hits,
                              flags)
        model = self.selectionModel()
        selection_flag = model.Select if _select else model.Deselect
        for index in matches:
            if _select ^ model.isSelected(index):
                model.select(index, selection_flag | model.Rows)
        return matches

    def deselectRowsWhere(self, col, value, n_hits=-1,
                          flags=QtCore.Qt.MatchExactly):
        """
        Deselect (also return) at most `n_hits` rows where column `col`
        has value (``data()``) `value`.
        """
        return self.selectRowsWhere(col, value, n_hits, flags, False)


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
    inputs = [("Topics", Table, "on_data")]
    outputs = []

    selected_words = settings.ContextSetting(SelectedWords('whatevar (?)'))
    selected_topic = settings.ContextSetting('')
    words_color = settings.Setting(True)
    words_tilt = settings.Setting(1)

    def __init__(self):
        super().__init__()
        self.n_words = 0
        self.n_topics = 0
        self.mean_weight = 0
        self.selected_words = SelectedWords(self)
        self.webview = webview = WebviewWidget(self.mainArea, self, True)
        script = '''
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
        webview.setContent(script.encode('utf-8'), 'text/html')
        webview.evalJS(JS_WORDCLOUD)
        webview.evalJS(JS_SCRIPT)
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
        box = gui.widgetBox(self.controlArea, 'Info')
        gui.label(box, self, '%(n_topics)d topics')
        gui.label(box, self, '%(n_words)d words per topic')
        gui.label(box, self, 'Mean weight in selected topic: %(mean_weight).4f')

        box = gui.widgetBox(self.controlArea, 'Cloud preferences')
        gui.checkBox(box, self, 'words_color', 'Color words', callback=self.on_cloud_pref_change)
        TILT_VALUES = ('no', 'slight', 'more', 'full')
        gui.valueSlider(box, self, 'words_tilt', label='Words tilt:',
                        values=list(range(len(TILT_VALUES))),
                        callback=self.on_cloud_pref_change,
                        labelFormat=lambda x: TILT_VALUES[x])
        gui.button(box, None, 'Regenerate word cloud', callback=self.cloud_redraw)

        box = gui.widgetBox(self.controlArea, 'Words && weights')
        self.topic_combo = gui.comboBox(box, self, 'selected_topic',
                                        callback=self.on_topic_change,
                                        sendSelectedValue=True)
        self.table = SimpleTableWidget(box,
                                       col_labels=['Weight', 'Word'],
                                       multi_selection=True,
                                       select_rows=True)

        def _selection_changed(selected, deselected,
                              table_selection_changed=self.table.selectionChanged):
            table_selection_changed(selected, deselected)
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

    def _repopulate_topic_combo(self, data):
        metas = data.domain.metas if data else []
        self.topics = OrderedDict([(var.name, col)
                                   for col, var in enumerate(metas)
                                   if var.is_string])
        self.topic_combo.clear()
        self.topic_combo.addItems(list(self.topics.keys()))
        self.n_topics = len(self.topics)

    def on_data(self, data):
        self.data = data
        self._repopulate_topic_combo(data)
        self.on_topic_change()

    def _repopulate_table(self, data):
        self.table.clear()
        for word, weight in data:
            self.table.add((weight, word), data=word)
        self.table.sortByColumn(0, QtCore.Qt.DescendingOrder)

    def _repopulate_wordcloud(self, data):
        self.mean_weight = mean = np.mean(data[:, 1])
        MIN_SIZE, MEAN_SIZE, MAX_SIZE = 8, 18, 40

        def _size(w):
            return np.clip(w/mean*MEAN_SIZE, MIN_SIZE, MAX_SIZE)

        wordlist = [[word, _size(weight)] for word, weight in data]
        self.webview.evalJS('OPTIONS["list"] = {};'.format(wordlist))
        self.on_cloud_pref_change()

    def on_topic_change(self):
        col = self.topics.get(self.selected_topic, None)
        if col is None and self.topics:
            col = 0
            topic = next(iter(self.topics))
            self.topic_combo.setCurrentIndex(self.topic_combo.findText(topic))
        N_BEST = 200
        data = self.data.metas[:N_BEST, col:col+2] if self.topics else np.zeros((0, 2))
        self.n_words = data.shape[0]
        self._repopulate_table(data)
        self._repopulate_wordcloud(data)


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
    w.on_data(table)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
