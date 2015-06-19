from PyQt4 import QtGui

from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.data import Table
from orangecontrib.text.stats import false_discovery_rate, hypergeom_p_values


class OWWordEnrichment(OWWidget):
    # Basic widget info
    name = "Word Enrichment"
    description = "Word enrichment analysis for selected documents."
    icon = "icons/SetEnrichment.svg"
    priority = 60

    # Input/output
    inputs = [("Selected Data", Table, "set_data_selected"),
              ("Other Data", Table, "set_data_other")]
    want_main_area = True

    # Settings
    filter_by_p = Setting(False)
    filter_p_value = Setting(0.01)
    filter_by_fdr = Setting(False)
    filter_fdr_value = Setting(0.01)

    def __init__(self):
        super().__init__()

        # Init data
        self.selected_data = None
        self.other_data = None
        self.words = []
        self.p_values = []
        self.fdr_values = []

        # Filtering settings
        fbox = gui.widgetBox(self.controlArea, "Filter words")

        gui.checkBox(fbox, self, "filter_by_p", "p-value",
                     callback=self.filter_and_display,
                     tooltip="Filter by word p-value")
        gui.doubleSpin(gui.indentedBox(fbox), self, 'filter_p_value',
                       1e-4, 1, step=1e-4,  label='p:', labelWidth=15,
                       callback=self.filter_and_display,
                       callbackOnReturn=True,
                       tooltip="Max p-value for word")

        gui.checkBox(fbox, self, "filter_by_fdr", "FDR",
                     callback=self.filter_and_display,
                     tooltip="Filter by word FDR")
        gui.doubleSpin(gui.indentedBox(fbox), self, 'filter_fdr_value',
                       1e-4, 1, step=1e-4,  label='p:', labelWidth=15,
                       callback=self.filter_and_display,
                       callbackOnReturn=True,
                       tooltip="Max p-value for word")

        gui.rubber(self.controlArea)

        # Word's list view
        self.cols = ['Word', 'p-value', 'FDR']
        self.sig_words = QtGui.QTreeWidget()
        self.sig_words.setColumnCount(len(self.cols))
        self.sig_words.setHeaderLabels(self.cols)
        self.sig_words.setSortingEnabled(True)
        self.sig_words.setSelectionMode(QtGui.QTreeView.ExtendedSelection)
        self.sig_words.sortByColumn(1, 0)   # 0 is ascending order
        for i in range(len(self.cols)):
            self.sig_words.resizeColumnToContents(i)
        self.mainArea.layout().addWidget(self.sig_words)

    def set_data_selected(self, data=None):
        self.selected_data = data

    def set_data_other(self, data=None):
        self.other_data = data

    def handleNewSignals(self):
        self.check_data()

    def check_data(self):
        self.error(1)
        if isinstance(self.selected_data, Table) and \
                isinstance(self.other_data, Table):
            if self.selected_data.domain == self.other_data.domain:
                self.apply()
            else:
                self.clear()
                self.error(1, 'The domains do not match')
        else:
            self.clear()

    def clear(self):
        self.sig_words.clear()
        self.sig_words.sortByColumn(1, 0)   # 0 is ascending order

    def filter_and_display(self, resize_columns=False):
        self.sig_words.clear()
        if self.words:
            for word, pval, fval in zip(self.words, self.p_values, self.fdr_values):
                if (not self.filter_by_p or pval <= self.filter_p_value) and \
                        (not self.filter_by_fdr or fval <= self.filter_fdr_value):
                    it = EATreeWidgetItem(word, pval, fval, self.sig_words)
                    self.sig_words.addTopLevelItem(it)

        if resize_columns:
            for i in range(len(self.cols)):
                self.sig_words.resizeColumnToContents(i)

    def progress(self, p):
        self.progressBarSet(p)

    def apply(self):
        if self.selected_data.X.size > 0:
            self.progressBarInit()
            self.words = [i.name for i in self.selected_data.domain.attributes]
            self.p_values = hypergeom_p_values(self.selected_data.X,
                                               self.other_data.X,
                                               callback=self.progress)
            self.fdr_values = false_discovery_rate(self.p_values)
            self.filter_and_display(resize_columns=True)
            self.progressBarFinished()
        else:
            self.clear()


fp = lambda score: "%0.5f" % score if score > 10e-3 else "%0.1e" % score
fpt = lambda score: "%0.9f" % score if score > 10e-3 else "%0.5e" % score


class EATreeWidgetItem(QtGui.QTreeWidgetItem):
    def __init__(self, word, p_value, f_value, parent):
        super().__init__(parent)
        self.data = [word, p_value, f_value]
        self.setText(0, word)
        self.setText(1, fp(p_value))
        self.setToolTip(1, fpt(p_value))
        self.setText(2, fp(f_value))
        self.setToolTip(2, fpt(f_value))

    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        return self.data[col] < other.data[col]
