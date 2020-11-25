from unittest.mock import Mock
from unittest import skipIf

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

try:
    from orangecontrib.network import Network
    from orangecontrib.text.widgets.owcorpustonetwork import OWCorpusToNetwork
    SKIP = False
except Exception:
    SKIP = True
from orangecontrib.text import Corpus


@skipIf(SKIP, "Network add-on is not installed.")
class TestOWCorpusToNetwork(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWCorpusToNetwork)
        self.corpus = Corpus.from_file('deerwester')
        self.larger_corpus = Corpus.from_file('book-excerpts')

    def test_input(self):
        set_data = self.widget.set_data = Mock()
        self.send_signal("Corpus", None)
        set_data.assert_called_with(None)
        self.send_signal("Corpus", self.corpus[:0])
        set_data.assert_called_with(self.corpus[:0])
        self.send_signal("Corpus", self.corpus)
        set_data.assert_called_with(self.corpus)

    def test_output(self):
        self.send_signal("Corpus", None)
        self.assertIsNone(self.get_output(self.widget.Outputs.network))
        self.assertIsNone(self.get_output(self.widget.Outputs.items))

        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.network))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.items))

    def test_input_summary(self):
        input_summary = self.widget.info.set_input_summary = Mock()
        self.send_signal("Corpus", None)
        input_summary.assert_called_with(self.widget.info.NoInput)

        self.send_signal("Corpus", self.corpus)
        input_summary.assert_called_with(str(len(self.corpus)),
                                         "Corpus with 9 documents.")

    def test_output_summary(self):
        cbox = self.widget.controls.node_type
        simulate.combobox_activate_index(cbox, 0)
        output_summary = self.widget.info.set_output_summary = Mock()
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        summary = "Undirected network with 9 nodes and 26 edges."
        output_summary.assert_called_with("9 / 26", summary)

    def test_disabled(self):
        cbox = self.widget.controls.node_type
        simulate.combobox_activate_index(cbox, 0)
        self.assertFalse(self.widget.controls.window_size.isEnabled())
        self.assertFalse(self.widget.controls.freq_threshold.isEnabled())
        simulate.combobox_activate_index(cbox, 1)
        self.assertTrue(self.widget.controls.window_size.isEnabled())
        self.assertTrue(self.widget.controls.freq_threshold.isEnabled())
