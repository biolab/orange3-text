from unittest import mock

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owpreprocess import OWPreprocess


class TestPreprocessWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.corpus = Corpus.from_file('deerwester')

    def test_set_corpus(self):
        """
        Just basic test.
        """
        self.send_signal("Corpus", self.corpus)

    def test_multiple_instances(self):
        # GH-327
        widget2 = self.create_widget(OWPreprocess)
        widget3 = self.create_widget(OWPreprocess)

    def test_udpipe_offline(self):
        """Test if preprocess works with offline udpipe"""
        with mock.patch('orangecontrib.text.preprocess.normalize.UDPipeModels.online',
                        new_callable=mock.PropertyMock, return_value=False):
            widget = self.create_widget(OWPreprocess)
            normalization = widget.stages[2]
            normalization.on_off_button.click()
            self.assertTrue(widget.Warning.udpipe_offline.is_shown())

    def test_udpipe_no_models(self):
        """Test if preprocess disables udpipe if no models"""
        with mock.patch(
                'orangecontrib.text.preprocess.normalize.UDPipeModels.online',
                new_callable=mock.PropertyMock, return_value=False):
            with mock.patch('orangecontrib.text.preprocess.normalize.UDPipeModels.supported_languages',
                            new_callable=mock.PropertyMock, return_value=[]):
                widget = self.create_widget(OWPreprocess)
                normalization = widget.stages[2]
                normalization.on_off_button.click()
                self.assertFalse(normalization.group.button(normalization.UDPIPE)
                                 .isEnabled())
                self.assertFalse(normalization.udpipe_tokenizer_box.isEnabled())
                self.assertTrue(widget.Warning.udpipe_offline_no_models.is_shown())
