import unittest
from unittest.mock import patch, PropertyMock, MagicMock

from orangewidget.utils.filedialogs import RecentPath
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import RegexpTokenizer, WhitespaceTokenizer, \
    LowercaseTransformer, HtmlTransformer, PorterStemmer, SnowballStemmer, \
    UDPipeLemmatizer, StopwordsFilter, MostFrequentTokensFilter, NGrams
from orangecontrib.text.tag import AveragedPerceptronTagger, StanfordPOSTagger, \
    MaxEntTagger
from orangecontrib.text.widgets.owpreprocess import OWPreprocess, \
    TransformationModule, TokenizerModule, NormalizationModule, \
    FilteringModule, NgramsModule, POSTaggingModule


class TestOWPreprocess(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPreprocess)
        self.corpus = Corpus.from_file("deerwester")

    def test_outputs(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        pp_data = self.get_output(self.widget.Outputs.corpus)
        self.assertIsInstance(pp_data, Corpus)
        self.assertIsNot(pp_data, self.corpus)
        self.assertNotEqual(pp_data, self.corpus)

    def test_available_preprocessors(self):
        self.assertEqual(self.widget.preprocessors.rowCount(), 6)

    def test_default_preprocessors(self):
        self.assertEqual(self.widget.preprocessormodel.rowCount(), 3)

    def test_multiple_instances(self):
        # GH-327
        self.create_widget(OWPreprocess)
        self.create_widget(OWPreprocess)

    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess."
           "storedsettings",
           PropertyMock(return_value={
               "preprocessors": [("preprocess.normalize",
                                  {"method": NormalizationModule.Porter}),
                                 ("preprocess.tokenize",
                                  {"method": TokenizerModule.Word})]
           }))
    def test_tokenizer_propagated(self):
        widget = self.create_widget(OWPreprocess)
        self.send_signal(widget.Inputs.corpus, self.corpus)
        self.assertTrue(widget.Warning.tokenizer_propagated.is_shown())

    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess."
           "storedsettings",
           PropertyMock(return_value={
               "preprocessors": [("preprocess.tokenize",
                                  {"method": TokenizerModule.Word}),
                                 ("preprocess.normalize",
                                  {"method": NormalizationModule.UDPipe,
                                   "udpipe_tokenizer": True})]
           }))
    def test_tokenizer_ignored(self):
        widget = self.create_widget(OWPreprocess)
        self.send_signal(widget.Inputs.corpus, self.corpus)
        self.assertTrue(widget.Warning.tokenizer_ignored.is_shown())

    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess."
           "storedsettings",
           PropertyMock(return_value={
               "preprocessors": [("preprocess.filter",
                                  {"method": FilteringModule.Stopwords}),
                                 ("preprocess.normalize",
                                  {"method": NormalizationModule.UDPipe,
                                   "udpipe_tokenizer": True})]
           }))
    def test_filtering_ignored(self):
        widget = self.create_widget(OWPreprocess)
        self.send_signal(widget.Inputs.corpus, self.corpus)
        self.assertTrue(widget.Warning.filtering_ignored.is_shown())

    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess."
           "storedsettings",
           PropertyMock(return_value={
               "preprocessors": [("preprocess.normalize",
                                  {"method": NormalizationModule.UDPipe})]
           }))
    @patch("orangecontrib.text.preprocess.normalize.UDPipeModels.online",
           PropertyMock(return_value=False))
    @patch("orangecontrib.text.preprocess.normalize.UDPipeModels.model_files",
           PropertyMock(return_value=["English"]))
    def test_udpipe_offline(self):
        widget = self.create_widget(OWPreprocess)
        self.send_signal(widget.Inputs.corpus, self.corpus)
        self.assertTrue(widget.Warning.udpipe_offline.is_shown())
        self.assertFalse(widget.Warning.udpipe_offline_no_models.is_shown())

    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess."
           "storedsettings",
           PropertyMock(return_value={
               "preprocessors": [("preprocess.normalize",
                                  {"method": NormalizationModule.UDPipe})]
           }))
    @patch("orangecontrib.text.preprocess.normalize.UDPipeModels.online",
           PropertyMock(return_value=False))
    @patch("orangecontrib.text.preprocess.normalize.UDPipeModels.model_files",
           PropertyMock(return_value=[]))
    def test_udpipe_no_models(self):
        widget = self.create_widget(OWPreprocess)
        self.send_signal(widget.Inputs.corpus, self.corpus)
        self.assertTrue(widget.Warning.udpipe_offline_no_models.is_shown())
        self.assertFalse(widget.Warning.udpipe_offline.is_shown())

    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess."
           "storedsettings",
           PropertyMock(return_value={
               "preprocessors": [("preprocess.filter",
                                  {"method": FilteringModule.Stopwords})]
           }))
    @patch("orangecontrib.text.preprocess.filter.FileWordListMixin.from_file")
    def test_unicode_error(self, from_file: MagicMock):
        def fun(*_):
            raise UnicodeError

        from_file.side_effect = fun
        widget = self.create_widget(OWPreprocess)
        self.assertTrue(widget.Error.invalid_encoding.is_shown())

    # TODO - implement StanfordPOSTagger
    # @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess."
    #        "storedsettings",
    #        PropertyMock(return_value={
    #            "preprocessors": [("tag.pos",
    #                               {"method": POSTaggingModule.Stanford})]
    #        }))
    # def test_stanford_tagger_error(self):
    #     widget = self.create_widget(OWPreprocess)
    #     self.assertTrue(widget.Error.stanford_tagger.is_shown())


class TestOWPreprocessMigrateSettings(WidgetTest):
    def test_migrate_settings_transform(self):
        settings = {"__version__": 1,
                    "transformers": {"checked": [0, 2, 3], "enabled": True}}
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        params = [("preprocess.transform", {"methods": [0, 2, 3]})]
        self.assertEqual(widget.storedsettings["preprocessors"], params)

    def test_migrate_settings_tokenize(self):
        settings = {"__version__": 1,
                    "tokenizer": {"method_index": 1, "pattern": "foo",
                                  "enabled": True}}
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        params = [("preprocess.tokenize", {"method": 1, "pattern": "foo"})]
        self.assertEqual(widget.storedsettings["preprocessors"], params)

    def test_migrate_settings_normalize(self):
        settings = {"__version__": 1,
                    "normalizer": {"enabled": True, "method_index": 2,
                                   "snowball_language": "French",
                                   "udpipe_language": "German",
                                   "udpipe_tokenizer": True}}
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        params = [("preprocess.normalize",
                   {"method": 2, "snowball_language": "French",
                    "udpipe_language": "German", "udpipe_tokenizer": True})]
        self.assertEqual(widget.storedsettings["preprocessors"], params)

    def test_migrate_settings_filter(self):
        settings = {"__version__": 1,
                    "filters": {"checked": [0, 2, 4], "enabled": True,
                                "keep_n": 50, "max_df": 0.5, "min_df": 0.3,
                                "pattern": "foo",
                                "recent_lexicon_files": ["(none)"],
                                "recent_sw_files": ["(none)"],
                                "stopwords_language": "Finnish",
                                "use_df": False, "use_keep_n": False}}
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        params = [("preprocess.filter",
                   {"methods": [0, 2, 4], "language": "Finnish",
                    "sw_path": None, "sw_list": [],
                    "lx_path": None, "lx_list": [],
                    "pattern": "foo", "rel_start": 0.3,
                    "rel_end": 0.5, "n_tokens": 50}
                   )]
        self.assertEqual(widget.storedsettings["preprocessors"], params)

    def test_migrate_settings_ngrams(self):
        settings = {"__version__": 1,
                    "ngrams_range": {"enabled": True, "ngrams_range": (5, 6)}}
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        params = [("preprocess.ngrams", {"start": 5, "end": 6})]
        self.assertEqual(widget.storedsettings["preprocessors"], params)

    def test_migrate_settings_tagger(self):
        settings = {"__version__": 1,
                    "pos_tagger": {"enabled": True, "method_index": 1,
                                   "stanford": {"recent_files": ["(none)"],
                                                "recent_provider": ["(none)"],
                                                "resource_path": ""}}}
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        params = [("tag.pos", {"method": 1})]
        self.assertEqual(widget.storedsettings["preprocessors"], params)

    def test_migrate_settings(self):
        settings = {
            "__version__": 1,
            "filters": {"checked": [0, 1, 2, 3, 4], "enabled": True,
                        "keep_n": 50, "max_df": 0.5, "min_df": 0.3,
                        "pattern": "foo", "recent_lexicon_files": [
                    "/Users/vesna/Desktop/computer.txt", "(none)"],
                        "recent_sw_files": ["/Users/vesna/Desktop/test.txt",
                                            "(none)"],
                        "stopwords_language": "Finnish", "use_df": False,
                        "use_keep_n": False},
            "ngrams_range": {"enabled": True, "ngrams_range": (5, 6)},
            "normalizer": {"enabled": True, "method_index": 2,
                           "snowball_language": "French",
                           "udpipe_language": "German",
                           "udpipe_tokenizer": True},
            "pos_tagger": {"enabled": True, "method_index": 1,
                           "stanford": {"recent_files": ["(none)"],
                                        "recent_provider": ["(none)"],
                                        "resource_path": ""}},
            "tokenizer": {"enabled": False, "method_index": 0,
                          "pattern": "foo"},
            "transformers": {"checked": [0, 1, 2, 3], "enabled": True}
        }
        self.create_widget(OWPreprocess, stored_settings=settings)


class TestTransformationModule(WidgetTest):
    def setUp(self):
        self.editor = TransformationModule()

    @property
    def check_boxes(self):
        return [cb for i, cb in self.editor._MultipleMethodModule__cbs]

    def test_init(self):
        check_boxes = self.check_boxes
        self.assertTrue(check_boxes[0].isChecked())
        self.assertGreater(len(check_boxes[0].toolTip()), 0)
        for i in range(1, len(check_boxes)):
            self.assertFalse(check_boxes[i].isChecked())
            self.assertGreater(len(check_boxes[i].toolTip()), 0)

    def test_parameters(self):
        params = {"methods": [TransformationModule.Lowercase]}
        self.assertDictEqual(self.editor.parameters(), params)

    def test_set_parameters(self):
        params = {"methods": [TransformationModule.Accents,
                              TransformationModule.Parse]}
        self.editor.setParameters(params)
        self.assertDictEqual(self.editor.parameters(), params)

        check_boxes = self.check_boxes
        self.assertFalse(check_boxes[0].isChecked())
        self.assertTrue(check_boxes[1].isChecked())
        self.assertTrue(check_boxes[2].isChecked())
        self.assertFalse(check_boxes[3].isChecked())

    def test_createinstance(self):
        pp = self.editor.createinstance(self.editor.parameters())
        self.assertIsInstance(pp[0], LowercaseTransformer)

        params = {"methods": [TransformationModule.Lowercase,
                              TransformationModule.Parse]}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp[0], LowercaseTransformer)
        self.assertIsInstance(pp[1], HtmlTransformer)

    def test_repr(self):
        self.assertEqual(str(self.editor), "Lowercase")
        params = {"methods": [TransformationModule.Lowercase,
                              TransformationModule.Parse]}
        self.editor.setParameters(params)
        self.assertEqual(str(self.editor), "Lowercase, Parse html")


class TestTokenizerModule(WidgetTest):
    def setUp(self):
        self.editor = TokenizerModule()

    @property
    def buttons(self):
        return self.editor._SingleMethodModule__group.buttons()

    @property
    def line_edit(self):
        return self.editor._TokenizerModule__edit

    def test_init(self):
        self.assertTrue(self.buttons[3].isChecked())
        self.assertGreater(len(self.buttons[3].toolTip()), 0)
        for i in [0, 1, 2, 4]:
            self.assertFalse(self.buttons[i].isChecked())
            self.assertGreater(len(self.buttons[i].toolTip()), 0)

        self.assertEqual(self.line_edit.text(), "\w+")

    def test_parameters(self):
        params = {"method": TokenizerModule.Regexp, "pattern": "\w+"}
        self.assertDictEqual(self.editor.parameters(), params)

    def test_set_parameters(self):
        params = {"method": TokenizerModule.Whitespace, "pattern": "foo"}
        self.editor.setParameters(params)
        self.assertDictEqual(self.editor.parameters(), params)

        self.assertTrue(self.buttons[1].isChecked())
        for i in [0, 2, 3, 4]:
            self.assertFalse(self.buttons[i].isChecked())
        self.assertEqual(self.line_edit.text(), "foo")

    def test_createinstance(self):
        pp = self.editor.createinstance(self.editor.parameters())
        self.assertIsInstance(pp, RegexpTokenizer)
        self.assertEqual(pp._RegexpTokenizer__pattern, "\w+")

        params = {"pattern": "foo"}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp, RegexpTokenizer)
        self.assertEqual(pp._RegexpTokenizer__pattern, "foo")

        params = {"method": TokenizerModule.Whitespace}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp, WhitespaceTokenizer)

    def test_repr(self):
        self.assertEqual(str(self.editor), "Regexp (\\w+)")


class TestNormalizationModule(WidgetTest):
    def setUp(self):
        self.editor = NormalizationModule()

    @property
    def buttons(self):
        return self.editor._SingleMethodModule__group.buttons()

    @property
    def combo_sbl(self):
        return self.editor._NormalizationModule__combo_sbl

    @property
    def combo_udl(self):
        return self.editor._NormalizationModule__combo_udl

    @property
    def check_use(self):
        return self.editor._NormalizationModule__check_use

    def test_init(self):
        self.assertTrue(self.buttons[0].isChecked())
        for i in range(1, 4):
            self.assertFalse(self.buttons[i].isChecked())
        self.assertEqual(self.combo_sbl.currentText(), "English")
        self.assertEqual(self.combo_udl.currentText(), "English")
        self.assertFalse(self.check_use.isChecked())

    def test_parameters(self):
        params = {"method": NormalizationModule.Porter,
                  "snowball_language": "English",
                  "udpipe_language": "English",
                  "udpipe_tokenizer": False}
        self.assertDictEqual(self.editor.parameters(), params)

    def test_set_parameters(self):
        params = {"method": NormalizationModule.UDPipe,
                  "snowball_language": "Dutch",
                  "udpipe_language": "Finnish",
                  "udpipe_tokenizer": True}
        self.editor.setParameters(params)
        self.assertDictEqual(self.editor.parameters(), params)
        self.assertEqual(self.combo_sbl.currentText(), "Dutch")
        self.assertEqual(self.combo_udl.currentText(), "Finnish")
        self.assertTrue(self.check_use.isChecked())

    def test_createinstance(self):
        pp = self.editor.createinstance(self.editor.parameters())
        self.assertIsInstance(pp, PorterStemmer)

        params = {"method": NormalizationModule.Snowball}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp, SnowballStemmer)
        self.assertEqual(str(pp.normalizer.stemmer), "<EnglishStemmer>")

        params = {"method": NormalizationModule.Snowball,
                  "snowball_language": "Dutch"}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp, SnowballStemmer)
        self.assertEqual(str(pp.normalizer.stemmer), "<DutchStemmer>")

        params = {"method": NormalizationModule.UDPipe,
                  "udpipe_language": "Finnish",
                  "udpipe_tokenizer": True}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp, UDPipeLemmatizer)
        self.assertEqual(pp._UDPipeLemmatizer__language, "Finnish")
        self.assertEqual(pp._UDPipeLemmatizer__use_tokenizer, True)

    def test_repr(self):
        self.assertEqual(str(self.editor), "Porter Stemmer")

    @patch("orangecontrib.text.preprocess.normalize.UDPipeModels.online",
           PropertyMock(return_value=False))
    @patch("orangecontrib.text.preprocess.normalize.UDPipeModels.model_files",
           PropertyMock(return_value=[]))
    def test_udpipe_no_models(self):
        editor = NormalizationModule()
        button = editor._SingleMethodModule__group.button(editor.UDPipe)
        self.assertFalse(button.isEnabled())
        combo = editor._NormalizationModule__combo_udl
        self.assertFalse(combo.isEnabled())
        check = editor._NormalizationModule__check_use
        self.assertFalse(check.isEnabled())


class TestFilterModule(WidgetTest):
    def setUp(self):
        self.editor = FilteringModule()

    @property
    def check_boxes(self):
        return [cb for i, cb in self.editor._MultipleMethodModule__cbs]

    @property
    def combo(self):
        return self.editor._FilteringModule__combo

    @property
    def sw_combo(self):
        return self.editor._FilteringModule__sw_loader.file_combo

    @property
    def lx_combo(self):
        return self.editor._FilteringModule__lx_loader.file_combo

    @property
    def line_edit(self):
        return self.editor._FilteringModule__edit

    @property
    def group_buttons(self):
        return self.editor._FilteringModule__freq_group.buttons()

    @property
    def rel_spins(self):
        return self.editor._FilteringModule__rel_range_spins.spins()

    @property
    def abs_spins(self):
        return self.editor._FilteringModule__abs_range_spins.spins()

    @property
    def spin(self):
        return self.editor._FilteringModule__spin_n

    def test_init(self):
        check_boxes = self.check_boxes
        self.assertTrue(check_boxes[0].isChecked())
        self.assertGreater(len(check_boxes[0].toolTip()), 0)
        for i in range(1, len(check_boxes)):
            self.assertFalse(check_boxes[i].isChecked())
            self.assertGreater(len(check_boxes[i].toolTip()), 0)
        self.assertEqual(self.combo.currentText(), "English")
        self.assertEqual(self.sw_combo.currentText(), "(none)")
        self.assertEqual(self.lx_combo.currentText(), "(none)")
        self.assertEqual(self.line_edit.text(), FilteringModule.DEFAULT_PATTERN)
        self.assertTrue(self.group_buttons[0].isChecked())
        self.assertFalse(self.group_buttons[1].isChecked())
        self.assertEqual(self.rel_spins[0].value(), 0.1)
        self.assertEqual(self.rel_spins[1].value(), 0.9)
        self.assertEqual(self.abs_spins[0].value(), 1)
        self.assertEqual(self.abs_spins[1].value(), 10)
        self.assertEqual(self.spin.value(), 100)

    def test_parameters(self):
        params = {"methods": [FilteringModule.Stopwords],
                  "language": "English", "sw_path": None, "lx_path": None,
                  "sw_list": [], "lx_list": [],
                  "pattern": FilteringModule.DEFAULT_PATTERN,
                  "freq_type": 0,
                  "rel_start": 0.1, "rel_end": 0.9,
                  "abs_start": 1, "abs_end": 10,
                  "n_tokens": 100, "invalidated": False}
        self.assertDictEqual(self.editor.parameters(), params)

    def test_set_parameters(self):
        sw_path = RecentPath.create("Foo", [])
        lx_path = RecentPath.create("Bar", [])
        params = {"methods": [FilteringModule.Lexicon, FilteringModule.Regexp],
                  "language": "Finnish",
                  "sw_path": sw_path, "lx_path": lx_path,
                  "sw_list": [sw_path], "lx_list": [lx_path],
                  "pattern": "foo",
                  "freq_type": 1,
                  "rel_start": 0.2, "rel_end": 0.7,
                  "abs_start": 2, "abs_end": 15,
                  "n_tokens": 10, "invalidated": False}
        self.editor.setParameters(params)
        self.assertDictEqual(self.editor.parameters(), params)

        check_boxes = self.check_boxes
        self.assertFalse(check_boxes[0].isChecked())
        self.assertTrue(check_boxes[1].isChecked())
        self.assertTrue(check_boxes[2].isChecked())
        self.assertFalse(check_boxes[3].isChecked())
        self.assertFalse(check_boxes[4].isChecked())

        self.assertEqual(self.combo.currentText(), "Finnish")
        self.assertEqual(self.sw_combo.currentText(), "Foo")
        self.assertEqual(self.lx_combo.currentText(), "Bar")
        self.assertEqual(self.line_edit.text(), "foo")
        self.assertFalse(self.group_buttons[0].isChecked())
        self.assertTrue(self.group_buttons[1].isChecked())
        self.assertEqual(self.rel_spins[0].value(), 0.2)
        self.assertEqual(self.rel_spins[1].value(), 0.7)
        self.assertEqual(self.abs_spins[0].value(), 2)
        self.assertEqual(self.abs_spins[1].value(), 15)
        self.assertEqual(self.spin.value(), 10)

    def test_createinstance(self):
        pp = self.editor.createinstance(self.editor.parameters())
        self.assertIsInstance(pp[0], StopwordsFilter)

        params = {"methods": [FilteringModule.Stopwords,
                              FilteringModule.MostFreq]}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp[0], StopwordsFilter)
        self.assertIsInstance(pp[1], MostFrequentTokensFilter)

    def test_repr(self):
        self.assertEqual(str(self.editor),
                         "Stopwords (Language: English, File: None)")
        params = {"methods": [FilteringModule.Lexicon,
                              FilteringModule.Regexp]}
        self.editor.setParameters(params)
        self.assertEqual(
            str(self.editor),
            f"Lexicon (File: None), Regexp ({FilteringModule.DEFAULT_PATTERN})"
        )


class TestNgramsModule(WidgetTest):
    def setUp(self):
        self.editor = NgramsModule()

    @property
    def spins(self):
        range = self.editor._NgramsModule__range_spins
        return range._spin_start, range._spin_end

    def test_init(self):
        self.assertEqual(self.spins[0].value(), 1)
        self.assertEqual(self.spins[1].value(), 2)

    def test_parameters(self):
        params = {"start": 1, "end": 2}
        self.assertDictEqual(self.editor.parameters(), params)

    def test_set_parameters(self):
        self.editor.setParameters({"start": 3, "end": 7})
        self.assertDictEqual(self.editor.parameters(), {"start": 3, "end": 7})

        self.assertEqual(self.spins[0].value(), 3)
        self.assertEqual(self.spins[1].value(), 7)

    def test_createinstance(self):
        pp = self.editor.createinstance(self.editor.parameters())
        self.assertIsInstance(pp, NGrams)
        self.assertEqual(pp._NGrams__range, (1, 2))

        params = {"start": 4}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp, NGrams)
        self.assertEqual(pp._NGrams__range, (4, 2))

    def test_repr(self):
        self.assertEqual(str(self.editor), "(1, 2)")


class TestPOSTaggerModule(WidgetTest):
    def setUp(self):
        self.editor = POSTaggingModule()

    @property
    def buttons(self):
        return self.editor._SingleMethodModule__group.buttons()

    def test_init(self):
        self.assertTrue(self.buttons[0].isChecked())
        for i in range(1, 2):
            self.assertFalse(self.buttons[i].isChecked())

    def test_parameters(self):
        params = {"method": POSTaggingModule.Averaged}
        self.assertDictEqual(self.editor.parameters(), params)

    def test_set_parameters(self):
        params = {"method": POSTaggingModule.MaxEnt}
        self.editor.setParameters(params)
        self.assertDictEqual(self.editor.parameters(), params)

        self.assertTrue(self.buttons[1].isChecked())
        for i in range(1):
            self.assertFalse(self.buttons[i].isChecked())

    def test_createinstance(self):
        pp = self.editor.createinstance(self.editor.parameters())
        self.assertIsInstance(pp, AveragedPerceptronTagger)

        pp = self.editor.createinstance({"method": POSTaggingModule.MaxEnt})
        self.assertIsInstance(pp, MaxEntTagger)

        # TODO - implement StanfordPOSTagger
        # pp = self.editor.createinstance({"method": POSTaggingModule.Stanford})
        # self.assertIsInstance(pp, StanfordPOSTagger)

    def test_repr(self):
        self.assertEqual(str(self.editor), "Averaged Perceptron Tagger")


if __name__ == "__main__":
    unittest.main()
