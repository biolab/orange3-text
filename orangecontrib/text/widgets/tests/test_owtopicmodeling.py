import unittest

from AnyQt.QtCore import QItemSelection, QItemSelectionRange, \
    QItemSelectionModel, QModelIndex
from Orange.widgets.tests.base import WidgetTest
from PyQt5 import Qt
from PyQt5.QtTest import QTest
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owtopicmodeling import OWTopicModeling


class TestTopicModeling(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWTopicModeling)
        self.corpus = Corpus.from_file('deerwester')

    def test_data(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        # otherwise fails in learning task as if it was sent None
        self.wait_until_finished()
        self.send_signal(self.widget.Inputs.corpus, None)
        self.wait_until_finished()
        
    def test_saved_selection(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        id = self.widget.topic_desc.model().index(2, 0)
        self.widget.topic_desc.selectionModel().select(id,
                                                       QItemSelectionModel.Rows |
                                                       QItemSelectionModel.Select)
        ids_1 = self.get_output(self.widget.Outputs.selected_topic).ids
        state = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(
            OWTopicModeling, stored_settings=state
        )
        self.send_signal(w.Inputs.corpus, self.corpus, widget=w)
        ids_2 = self.get_output(w.Outputs.selected_topic, widget=w).ids
        self.assertSequenceEqual(list(ids_1), list(ids_2))



if __name__ == "__main__":
    unittest.main()