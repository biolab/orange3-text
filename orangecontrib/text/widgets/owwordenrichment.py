from types import SimpleNamespace
from typing import List, Optional, Any

import numpy as np
from AnyQt.QtWidgets import QTreeWidget, QTreeView, QTreeWidgetItem

from Orange.data import Table, Domain
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import OWWidget, Msg, Input
from Orange.statistics.util import FDR
from PyQt5.QtCore import QSize
from orangecontrib.text import Corpus
from orangecontrib.text.util import np_sp_sum
from orangecontrib.text.stats import hypergeom_p_values


class Result(SimpleNamespace):
    words: Optional[List[str]] = None
    p_values: Optional[List[float]] = None
    fdr_values: Optional[List[float]] = None


class Runner:
    @staticmethod
    def run(
            selected_data_transformed: Table,
            data: Table,
            result: Result,
            state: TaskState
    ) -> None:
        state.set_status("Listing words")
        result.words = [
            i.name for i in selected_data_transformed.domain.attributes]
        state.set_status("Computing p-values")
        result.p_values = hypergeom_p_values(
            data.X, selected_data_transformed.X,
            callback=state.set_progress_value
        )
        state.set_status("Computing FDR values")
        result.fdr_values = FDR(result.p_values)


class OWWordEnrichment(OWWidget, ConcurrentWidgetMixin):
    # Basic widget info
    name = "Word Enrichment"
    description = "Word enrichment analysis for selected documents."
    icon = "icons/SetEnrichment.svg"
    priority = 600

    # Input/output
    class Inputs:
        selected_data = Input("Selected Data", Table)
        data = Input("Data", Table)

    want_main_area = True

    class Error(OWWidget.Error):
        no_bow_features = Msg('No bag-of-words features!')
        no_words_overlap = Msg('No words overlap!')
        empty_selection = Msg('Selected data is empty!')
        all_selected = Msg('All examples can not be selected!')

    # Settings
    filter_by_p: bool = Setting(False)
    filter_p_value: float = Setting(0.01)
    filter_by_fdr: bool = Setting(True)
    filter_fdr_value: float = Setting(0.2)

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        # Init data
        self.data = None
        self.selected_data = None
        # used for transforming the 'selected data' into the 'data' domain
        self.selected_data_transformed = None

        self.results = Result()

        # info box
        fbox = gui.widgetBox(self.controlArea, "Info")
        self.info_fil = gui.label(fbox, self, 'Words displayed: 0')

        # Filtering settings
        fbox = gui.widgetBox(self.controlArea, "Filter")
        hbox = gui.widgetBox(fbox, orientation=0)

        self.chb_p = gui.checkBox(hbox, self, "filter_by_p", "p-value",
                                  callback=self.filter_and_display,
                                  tooltip="Filter by word p-value")
        self.spin_p = gui.doubleSpin(hbox, self, 'filter_p_value',
                                     1e-4, 1, step=1e-4, labelWidth=15,
                                     callback=self.filter_and_display,
                                     tooltip="Max p-value for word")
        self.spin_p.setEnabled(self.filter_by_p)

        hbox = gui.widgetBox(fbox, orientation=0)
        self.chb_fdr = gui.checkBox(hbox, self, "filter_by_fdr", "FDR",
                                    callback=self.filter_and_display,
                                    tooltip="Filter by word FDR")
        self.spin_fdr = gui.doubleSpin(hbox, self, 'filter_fdr_value',
                                       1e-4, 1, step=1e-4, labelWidth=15,
                                       callback=self.filter_and_display,
                                       tooltip="Max p-value for word")
        self.spin_fdr.setEnabled(self.filter_by_fdr)
        gui.rubber(self.controlArea)

        # Word's list view
        self.cols = ['Word', 'p-value', 'FDR']
        self.sig_words = QTreeWidget()
        self.sig_words.setColumnCount(len(self.cols))
        self.sig_words.setHeaderLabels(self.cols)
        self.sig_words.setSortingEnabled(True)
        self.sig_words.setSelectionMode(QTreeView.NoSelection)
        self.sig_words.sortByColumn(1, 0)   # 0 is ascending order
        for i in range(len(self.cols)):
            self.sig_words.resizeColumnToContents(i)
        self.mainArea.layout().addWidget(self.sig_words)

    def sizeHint(self):
        return QSize(450, 240)

    @Inputs.data
    def set_data(self, data=None):
        self.data = data
        # selected data transformed depends on data domain
        self.selected_data_transformed = None

    @Inputs.selected_data
    def set_data_selected(self, data=None):
        self.selected_data = data

    def handleNewSignals(self):
        self.check_data()

    def get_bow_domain(self):
        domain = self.data.domain
        return Domain(
            attributes=[a for a in domain.attributes
                        if a.attributes.get('bow-feature', False)],
            class_vars=domain.class_vars,
            metas=domain.metas,
            source=domain)

    def check_data(self):
        self.Error.clear()
        if isinstance(self.data, Table) and \
                isinstance(self.selected_data, Table):
            if len(self.selected_data) == 0:
                self.Error.empty_selection()
                self.clear()
                return

            # keep only BoW features
            bow_domain = self.get_bow_domain()
            if len(bow_domain.attributes) == 0:
                self.Error.no_bow_features()
                self.clear()
                return
            self.data = Corpus.from_table(bow_domain, self.data)
            self.selected_data_transformed = Corpus.from_table(
                bow_domain, self.selected_data)

            if np_sp_sum(self.selected_data_transformed.X) == 0:
                self.Error.no_words_overlap()
                self.clear()
            elif len(self.data) == len(self.selected_data):
                self.Error.all_selected()
                self.clear()
            else:
                self.set_input_info()
                self.apply()
        else:
            self.clear()

    def clear(self):
        self.sig_words.clear()
        self.info.set_input_summary(self.info.NoInput)
        self.set_displayed_info(0)

    def filter_enabled(self, b):
        self.chb_p.setEnabled(b)
        self.chb_fdr.setEnabled(b)
        self.spin_p.setEnabled(b)
        self.spin_fdr.setEnabled(b)

    def filter_and_display(self):
        self.spin_p.setEnabled(self.filter_by_p)
        self.spin_fdr.setEnabled(self.filter_by_fdr)
        self.sig_words.clear()

        if self.selected_data_transformed is None:  # do nothing when no Data
            return

        if self.results.words:
            count = self.build_tree()
        else:
            count = 0

        for i in range(len(self.cols)):
            self.sig_words.resizeColumnToContents(i)
        self.set_displayed_info(count)

    def build_tree(self) -> int:
        count = 0
        for word, pval, fval in zip(
                self.results.words,
                self.results.p_values,
                self.results.fdr_values
        ):
            if ((not self.filter_by_p or pval <= self.filter_p_value) and
                    (not self.filter_by_fdr or fval <= self.filter_fdr_value)):
                it = EATreeWidgetItem(word, pval, fval, self.sig_words)
                self.sig_words.addTopLevelItem(it)
                count += 1
        return count

    def set_input_info(self) -> None:
        cluster_words = len(self.selected_data_transformed.domain.attributes)
        selected_words = np.count_nonzero(np_sp_sum(
            self.selected_data_transformed.X, axis=0))

        self.info.set_input_summary(
            f"{cluster_words}|{selected_words}",
            f"Total words: {cluster_words}\n"
            f"Words in subset: {selected_words}")

    def set_displayed_info(self, count: int) -> None:
        self.info_fil.setText(f"Words displayed: {count}")

    def apply(self):
        self.sig_words.clear()
        self.filter_enabled(False)
        self.start(
            Runner.run,
            self.selected_data_transformed,
            self.data,
            self.results
        )

    def on_done(self, result: Result) -> None:
        self.filter_and_display()
        self.filter_enabled(True)

    def on_exception(self, ex: Exception) -> None:
        self.filter_enabled(True)

    def tree_to_table(self):
        view = [self.cols]
        items = self.sig_words.topLevelItemCount()
        for i in range(items):
            line = []
            for j in range(3):
                line.append(self.sig_words.topLevelItem(i).text(j))
            view.append(line)
        return view

    def send_report(self):
        if self.results.words:
            self.report_table("Enriched words", self.tree_to_table())


fp = lambda score: "%0.5f" % score if score > 10e-3 else "%0.1e" % score
fpt = lambda score: "%0.9f" % score if score > 10e-3 else "%0.5e" % score


class EATreeWidgetItem(QTreeWidgetItem):
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


if __name__ == '__main__':
    from orangewidget.utils.widgetpreview import WidgetPreview
    from orangecontrib.text.vectorization import BowVectorizer

    corpus = Corpus.from_file('book-excerpts')
    vect = BowVectorizer()
    corpus_vect = vect.transform(corpus)
    WidgetPreview(OWWordEnrichment).run(
        set_data_selected=corpus_vect[:10], set_data=corpus_vect)
