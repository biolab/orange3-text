import unittest
from unittest.mock import patch, PropertyMock, MagicMock, Mock

import numpy as np
from AnyQt.QtGui import QStandardItem, QIcon
from Orange.data import Domain, StringVariable
from Orange.widgets.data.utils.preprocess import DescriptionRole, ParametersRole
from orangewidget.utils.filedialogs import RecentPath
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import RegexpTokenizer, WhitespaceTokenizer, \
    LowercaseTransformer, HtmlTransformer, PorterStemmer, SnowballStemmer, \
    UDPipeLemmatizer, StopwordsFilter, MostFrequentTokensFilter, NGrams
from orangecontrib.text.tag import (AveragedPerceptronTagger, MaxEntTagger,
                                    SpacyPOSTagger)
from orangecontrib.text.tests.test_preprocess import SF_LIST, SERVER_FILES
from orangecontrib.text.widgets.owpreprocess import (
    OWPreprocess,
    TransformationModule,
    TokenizerModule,
    NormalizationModule,
    FilteringModule,
    NgramsModule,
    POSTaggingModule,
    LanguageComboBox,
    _DEFAULT_NONE,
    UDPipeComboBox,
)


@patch(SF_LIST, new=Mock(return_value=SERVER_FILES))
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

    def test_previews(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.assertEqual("human, machine, interface, lab, abc", self.widget.preview)
        self.assertEqual("Tokens: 52\nTypes: 35", self.widget.output_info)
        self.send_signal(self.widget.Inputs.corpus, None)
        self.wait_until_finished()
        self.assertFalse(self.widget.preview)
        self.assertFalse(self.widget.output_info)

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
    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess.start", Mock())
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
    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess.start", Mock())
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
    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess.start", Mock())
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
           PropertyMock(return_value={}))
    @patch("orangecontrib.text.widgets.owpreprocess.OWPreprocess.start", Mock())
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

    @patch(
        "orangecontrib.text.widgets.owpreprocess.OWPreprocess.storedsettings",
        PropertyMock(
            return_value={
                "preprocessors": [
                    ("preprocess.tokenize", {"method": TokenizerModule.Word}),
                    ("preprocess.filter", {"method": FilteringModule.Stopwords}),
                ]
            }
        ),
    )
    def test_no_tokens_left(self):
        # make corpus with only stop words to get no_token_left warning
        domain = Domain([], metas=[StringVariable("Text")])
        corpus = Corpus.from_numpy(
            domain,
            np.empty((2, 0)),
            metas=np.array([["is are"], ["been"]]),
            text_features=domain.metas,
            language="en",
        )
        self.send_signal(self.widget.Inputs.corpus, corpus)
        self.wait_until_finished()
        self.assertTrue(self.widget.Warning.no_token_left.is_shown())

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.assertFalse(self.widget.Warning.no_token_left.is_shown())

    def test_language_from_corpus(self):
        """Test language from corpus is set correctly"""
        initial = {
            "name": "",
            "preprocessors": [("preprocess.normalize", {}), ("preprocess.filter", {})],
        }
        self.widget.storedsettings = initial
        self.widget._initialize()
        self.assertDictEqual(initial, self.widget.storedsettings)
        combos = self.widget.mainArea.findChildren(LanguageComboBox)
        self.assertEqual(
            ["English", "English", "English", "English"],
            [c.currentText() for c in combos]
        )

        # test with Slovenian - language should set for all preprocessors except
        # Snowball that doesn't support Slovenian
        self.corpus.attributes["language"] = "sl"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(
            ["English", "Slovenian", "Slovenian", "Slovenian"],
            [c.currentText() for c in combos]
        )
        settings = self.widget.storedsettings["preprocessors"]
        self.assertEqual("sl", settings[0][1]["udpipe_language"])
        self.assertEqual("sl", settings[0][1]["lemmagen_language"])
        self.assertEqual("sl", settings[1][1]["language"])

        # test with Lithuanian that is support by one preprocessors
        self.corpus.attributes["language"] = "lt"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(
            ["English", "Lithuanian", "Slovenian", "Slovenian"],
            [c.currentText() for c in combos]
        )
        settings = self.widget.storedsettings["preprocessors"]
        self.assertEqual("lt", settings[0][1]["udpipe_language"])
        self.assertEqual("sl", settings[0][1]["lemmagen_language"])
        self.assertEqual("sl", settings[1][1]["language"])

        self.corpus.attributes["language"] = "pt"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(
            ["Portuguese", "Portuguese", "Slovenian", "Portuguese"],
            [c.currentText() for c in combos]
        )
        settings = self.widget.storedsettings["preprocessors"]
        self.assertEqual("pt", settings[0][1]["snowball_language"])
        self.assertEqual("pt", settings[0][1]["udpipe_language"])
        self.assertEqual("sl", settings[0][1]["lemmagen_language"])
        self.assertEqual("pt", settings[1][1]["language"])

        # language not supported by any preprocessor - language shouldn't change
        self.corpus.attributes["language"] = "bo"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(
            ["Portuguese", "Portuguese", "Slovenian", "Portuguese"],
            [c.currentText() for c in combos]
        )
        settings = self.widget.storedsettings["preprocessors"]
        self.assertEqual("pt", settings[0][1]["snowball_language"])
        self.assertEqual("pt", settings[0][1]["udpipe_language"])
        self.assertEqual("sl", settings[0][1]["lemmagen_language"])
        self.assertEqual("pt", settings[1][1]["language"])

        # test with missing language - language shouldn't change
        self.corpus.attributes["language"] = None
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(
            ["Portuguese", "Portuguese", "Slovenian", "Portuguese"],
            [c.currentText() for c in combos]
        )
        settings = self.widget.storedsettings["preprocessors"]
        self.assertEqual("pt", settings[0][1]["snowball_language"])
        self.assertEqual("pt", settings[0][1]["udpipe_language"])
        self.assertEqual("sl", settings[0][1]["lemmagen_language"])
        self.assertEqual("pt", settings[1][1]["language"])

    def test_language_from_schema(self):
        """Test language from schema/workflow is retained"""
        initial = {
            "name": "",
            "preprocessors": [
                (
                    "preprocess.normalize",
                    {
                        "lemmagen_language": "sl",
                        "snowball_language": "nl",
                        "udpipe_language": "lt",
                    },
                ),
                ("preprocess.filter", {"language": "nl"}),
            ],
        }
        self.widget.storedsettings = initial

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        self.send_signal(widget.Inputs.corpus, self.corpus, widget=widget)
        self.assertDictEqual(initial, widget.storedsettings)
        combos = widget.mainArea.findChildren(LanguageComboBox)
        self.assertEqual(
            ["Dutch", "Lithuanian", "Slovenian", "Dutch"],
            [c.currentText() for c in combos]
        )

    def test_language_from_corpus_editor_inserted(self):
        """Test language from corpus is set to new editor too"""
        initial = {
            "name": "",
            "preprocessors": [("preprocess.filter", {})],
        }
        self.widget.storedsettings = initial
        self.widget._initialize()
        self.assertDictEqual(initial, self.widget.storedsettings)
        combos = self.widget.mainArea.findChildren(LanguageComboBox)
        self.assertEqual(
            ["English"],
            [c.currentText() for c in combos]
        )

        # insert data - language of stopwords combo should change to italian
        self.corpus.attributes["language"] = "sl"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(
            ["Slovenian"],
            [c.currentText() for c in combos]
        )

        # insert new editor - all languages except snowball should be set to Slovenian
        pp_def = self.widget._qname2ppdef["preprocess.normalize"]
        description = pp_def.description
        item = QStandardItem(description.title)
        icon = QIcon(description.icon)
        item.setIcon(icon)
        item.setToolTip(description.summary)
        item.setData(pp_def, DescriptionRole)
        item.setData({}, ParametersRole)
        self.widget.preprocessormodel.insertRow(0, [item])
        self.wait_until_finished()

        combos = self.widget.mainArea.findChildren(LanguageComboBox)
        self.assertEqual(
            ['Slovenian', 'English', 'Slovenian', 'Slovenian'],
            [c.currentText() for c in combos]
        )


@patch(SF_LIST, new=Mock(return_value=SERVER_FILES))
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
                                   "udpipe_language": "Portuguese",
                                   "udpipe_tokenizer": True}}
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        params = [("preprocess.normalize",
                   {"method": 2, "snowball_language": "fr",
                    "udpipe_language": "pt", "udpipe_tokenizer": True})]
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
                   {"methods": [0, 2, 4], "language": "fi",
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

    def test_migrate_filter_language_settings(self):
        """Test migration to iso langauge codes"""
        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [("preprocess.filter", {"language": "Finnish"})]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        filter_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertEqual("fi", filter_settings["language"])

        # NLTK uses Slovene instead of Slovenian, this is also the reason
        # that preprocess widget stored language as Slovene before
        # check if it is mapped correctly
        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [("preprocess.filter", {"language": "Slovene"})]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        filter_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertEqual("sl", filter_settings["language"])

        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [("preprocess.filter", {"language": _DEFAULT_NONE})]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        filter_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertIsNone(filter_settings["language"])

    def test_migrate_lemmagen_language_settings(self):
        """Test migration to iso langauge codes"""
        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [
                    ("preprocess.normalize", {"lemmagen_language": "Slovenian"}),
                ]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        normalize_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertEqual("sl", normalize_settings["lemmagen_language"])

        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [
                    ("preprocess.normalize", {"lemmagen_language": "English"}),
                ]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        normalize_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertEqual("en", normalize_settings["lemmagen_language"])

    def test_migrate_snowball_language_settings(self):
        """Test migration to iso langauge codes"""
        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [
                    ("preprocess.normalize", {"snowball_language": "Swedish"}),
                ]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        normalize_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertEqual("sv", normalize_settings["snowball_language"])

        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [
                    ("preprocess.normalize", {"snowball_language": "English"}),
                ]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        normalize_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertEqual("en", normalize_settings["snowball_language"])

    def test_migrate_udpipe_language_settings(self):
        """Test migration to iso langauge codes"""
        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [
                    ("preprocess.normalize", {"udpipe_language": "Slovenian"}),
                ]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        normalize_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertEqual("sl", normalize_settings["udpipe_language"])

        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [
                    ("preprocess.normalize", {"udpipe_language": "English (lines)"}),
                ]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        normalize_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertEqual("en_lines", normalize_settings["udpipe_language"])

        settings = {
            "__version__": 3,
            "storedsettings": {
                "preprocessors": [
                    ("preprocess.normalize", {"udpipe_language": "Abc"}),
                ]
            },
        }
        widget = self.create_widget(OWPreprocess, stored_settings=settings)
        normalize_settings = widget.storedsettings["preprocessors"][0][1]
        self.assertIsNone(normalize_settings["udpipe_language"])

    @unittest.skip("Very slow test")
    def test_migrate_udpipe_language_settings_slow(self):
        """
        Test migration to iso langauge codes. To run it successfully remove
        patch on the TestOWPreprocessMigrateSettings class
        """
        migrations = [
            ("Ancient greek proiel", "grc_proiel"),
            ("Ancient greek", "grc"),
            ("Arabic", "ar"),
            ("Basque", "eu"),
            ("Belarusian", "be"),
            ("Bulgarian", "bg"),
            ("Catalan", "ca"),
            ("Chinese", "zh"),
            ("Coptic", "cop"),
            ("Croatian", "hr"),
            ("Czech cac", "cs_cac"),
            ("Czech cltt", "cs_cltt"),
            ("Czech", "cs"),
            ("Danish", "da"),
            ("Dutch lassysmall", "nl_lassysmall"),
            ("Dutch", "nl"),
            ("English lines", "en_lines"),
            ("English partut", "en_partut"),
            ("English", "en"),
            ("Estonian", "et"),
            ("Finnish ftb", "fi_ftb"),
            ("Finnish", "fi"),
            ("French partut", "fr_partut"),
            ("French sequoia", "fr_sequoia"),
            ("French", "fr"),
            ("Galician treegal", "gl_treegal"),
            ("Galician", "gl"),
            ("German", "de"),
            ("Gothic", "got"),
            ("Greek", "el"),
            ("Hebrew", "he"),
            ("Hindi", "hi"),
            ("Hungarian", "hu"),
            ("Indonesian", "id"),
            ("Irish", "ga"),
            ("Italian", "it"),
            ("Japanese", "ja"),
            ("Kazakh", "kk"),
            ("Korean", "ko"),
            ("Latin ittb", "la_ittb"),
            ("Latin proiel", "la_proiel"),
            ("Latin", "la"),
            ("Latvian", "lv"),
            ("Lithuanian", "lt"),
            ("Norwegian bokmaal", "nb"),
            ("Norwegian nynorsk", "nn"),
            ("Old church slavonic", "cu"),
            ("Persian", "fa"),
            ("Polish", "pl"),
            ("Portuguese br", "pt_br"),
            ("Portuguese", "pt"),
            ("Romanian", "ro"),
            ("Russian syntagrus", "ru_syntagrus"),
            ("Russian", "ru"),
            ("Sanskrit", "sa"),
            ("Slovak", "sk"),
            ("Slovenian sst", "sl_sst"),
            ("Slovenian", "sl"),
            ("Spanish ancora", "es_ancora"),
            ("Spanish", "es"),
            ("Swedish lines", "sv_lines"),
            ("Swedish", "sv"),
            ("Tamil", "ta"),
            ("Turkish", "tr"),
            ("Ukrainian", "uk"),
            ("Urdu", "ur"),
            ("Uyghur", "ug"),
            ("Vietnamese", "vi"),
        ]
        for old_value, new_value in migrations:
            settings = {
                "__version__": 3,
                "storedsettings": {
                    "preprocessors": [
                        ("preprocess.normalize", {"udpipe_language": old_value}),
                    ]
                },
            }
            widget = self.create_widget(OWPreprocess, stored_settings=settings)
            normalize_settings = widget.storedsettings["preprocessors"][0][1]
            self.assertEqual(new_value, normalize_settings["udpipe_language"])


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


@patch(SF_LIST, new=Mock(return_value=SERVER_FILES))
class TestNormalizationModule(WidgetTest):
    @patch(SF_LIST, new=Mock(return_value=SERVER_FILES))
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
    def combo_lemm(self):
        return self.editor._NormalizationModule__combo_lemm

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
        params = {
            "method": NormalizationModule.Porter,
            "snowball_language": "en",
            "udpipe_language": "en",
            "lemmagen_language": "en",
            "udpipe_tokenizer": False,
        }
        self.assertDictEqual(self.editor.parameters(), params)

    def test_set_parameters(self):
        params = {
            "method": NormalizationModule.UDPipe,
            "snowball_language": "nl",
            "udpipe_language": "sl",
            "lemmagen_language": "bg",
            "udpipe_tokenizer": True,
        }
        self.editor.setParameters(params)
        self.assertDictEqual(self.editor.parameters(), params)
        self.assertEqual(self.combo_sbl.currentText(), "Dutch")
        self.assertEqual(self.combo_udl.currentText(), "Slovenian")
        self.assertEqual(self.combo_lemm.currentText(), "Bulgarian")
        self.assertTrue(self.check_use.isChecked())

    def test_createinstance(self):
        pp = self.editor.createinstance(self.editor.parameters())
        self.assertIsInstance(pp, PorterStemmer)

        params = {"method": NormalizationModule.Snowball}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp, SnowballStemmer)
        self.assertIn("<EnglishStemmer>", str(pp.normalizer))

        params = {"method": NormalizationModule.Snowball, "snowball_language": "nl"}
        pp = self.editor.createinstance(params)
        self.assertIsInstance(pp, SnowballStemmer)
        self.assertIn("<DutchStemmer>", str(pp.normalizer))

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
           PropertyMock(return_value={}))
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
                  "language": "en", "sw_path": None, "lx_path": None,
                  "sw_list": [], "lx_list": [],
                  "incl_num": False,
                  "pattern": FilteringModule.DEFAULT_PATTERN,
                  "freq_type": 0,
                  "rel_start": 0.1, "rel_end": 0.9,
                  "abs_start": 1, "abs_end": 10,
                  "n_tokens": 100, "pos_tags": "NOUN,VERB",
                  "invalidated": False}
        self.assertDictEqual(self.editor.parameters(), params)

    def test_set_parameters(self):
        sw_path = RecentPath.create("Foo", [])
        lx_path = RecentPath.create("Bar", [])
        params = {"methods": [FilteringModule.Lexicon, FilteringModule.Regexp],
                  "language": "fi",
                  "sw_path": sw_path, "lx_path": lx_path,
                  "sw_list": [sw_path], "lx_list": [lx_path],
                  "incl_num": False,
                  "pattern": "foo",
                  "freq_type": 1,
                  "rel_start": 0.2, "rel_end": 0.7,
                  "abs_start": 2, "abs_end": 15,
                  "n_tokens": 10,  "pos_tags": "JJ",
                  "invalidated": False}
        self.editor.setParameters(params)
        self.assertDictEqual(self.editor.parameters(), params)

        check_boxes = self.check_boxes
        self.assertFalse(check_boxes[0].isChecked())
        self.assertTrue(check_boxes[1].isChecked())
        self.assertFalse(check_boxes[2].isChecked())
        self.assertTrue(check_boxes[3].isChecked())
        self.assertFalse(check_boxes[4].isChecked())
        self.assertFalse(check_boxes[5].isChecked())

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
        self.assertEqual(str(self.editor), "Stopwords (Language: English, File: None)")
        params = self.editor.parameters()
        params["language"] = None
        self.editor.setParameters(params)
        self.assertEqual(str(self.editor), "Stopwords (Language: None, File: None)")

        params = {"methods": [FilteringModule.Lexicon, FilteringModule.Regexp]}
        self.editor.setParameters(params)
        self.assertEqual(
            str(self.editor),
            f"Lexicon (File: None), Regexp ({FilteringModule.DEFAULT_PATTERN})"
        )

    def test_abs_spins(self):
        self.abs_spins[0].setValue(5)
        self.abs_spins[1].setValue(3)
        self.assertEqual(5, self.abs_spins[0].value())
        self.assertEqual(5, self.abs_spins[1].value())
        self.assertEqual("max", self.abs_spins[1].text())
        self.abs_spins[1].setValue(6)
        self.assertEqual(5, self.abs_spins[0].value())
        self.assertEqual(6, self.abs_spins[1].value())
        self.abs_spins[0].setValue(7)
        self.assertEqual(7, self.abs_spins[0].value())
        self.assertEqual(7, self.abs_spins[1].value())
        self.assertEqual("max", self.abs_spins[1].text())


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
        for i in range(1, 3):
            self.assertFalse(self.buttons[i].isChecked())

    def test_parameters(self):
        params = {"method": POSTaggingModule.Averaged, "spacy_language":
                  POSTaggingModule.DEFAULT_LANGUAGE}
        self.assertDictEqual(self.editor.parameters(), params)

    def test_set_parameters(self):
        params = {"method": POSTaggingModule.Spacy, "spacy_language": "sl"}
        self.editor.setParameters(params)
        self.assertDictEqual(self.editor.parameters(), params)

        self.assertTrue(self.buttons[2].isChecked())
        for i in range(0, 2):
            self.assertFalse(self.buttons[i].isChecked())

    def test_createinstance(self):
        pp = self.editor.createinstance(self.editor.parameters())
        self.assertIsInstance(pp, AveragedPerceptronTagger)

        pp = self.editor.createinstance({"method": POSTaggingModule.MaxEnt})
        self.assertIsInstance(pp, MaxEntTagger)

        pp = self.editor.createinstance({"method": POSTaggingModule.Spacy})
        self.assertIsInstance(pp, SpacyPOSTagger)

    def test_repr(self):
        self.assertEqual(str(self.editor), "Averaged Perceptron Tagger")

        params = {"method": POSTaggingModule.Spacy, "spacy_language":
                  POSTaggingModule.DEFAULT_LANGUAGE}
        self.editor.setParameters(params)
        self.assertEqual(str(self.editor),
                         f"Spacy POS Tagger ({params['spacy_language']})")


class TestLanguageComboBox(WidgetTest):
    def test_basic_setup(self):
        mock = Mock()
        cb = LanguageComboBox(None, ["sl", "en", "sv", "fi"], "fi", False, mock)
        self.assertEqual(4, cb.count())
        self.assertEqual(
            ["English", "Finnish", "Slovenian", "Swedish"],
            [cb.itemText(i) for i in range(cb.count())],
        )
        self.assertEqual("Finnish", cb.currentText())

    def test_include_none(self):
        mock = Mock()
        cb = LanguageComboBox(None, ["sl", "en", "sv", "fi"], "fi", True, mock)
        self.assertEqual(5, cb.count())
        self.assertEqual(
            [_DEFAULT_NONE, "English", "Finnish", "Slovenian", "Swedish"],
            [cb.itemText(i) for i in range(cb.count())],
        )
        self.assertEqual("Finnish", cb.currentText())

        # test with current item None
        cb = LanguageComboBox(None, ["sl", "en", "sv", "fi"], None, True, mock)
        self.assertEqual(5, cb.count())
        self.assertEqual(
            [_DEFAULT_NONE, "English", "Finnish", "Slovenian", "Swedish"],
            [cb.itemText(i) for i in range(cb.count())],
        )
        self.assertEqual(_DEFAULT_NONE, cb.currentText())

    def test_set_current_language(self):
        mock = Mock()
        cb = LanguageComboBox(None, ["sl", "en", "sv", "fi"], "fi", True, mock)
        self.assertEqual("Finnish", cb.currentText())
        cb.set_current_language("sl")
        self.assertEqual("Slovenian", cb.currentText())
        cb.set_current_language(None)
        self.assertEqual(_DEFAULT_NONE, cb.currentText())

    def test_change_item(self):
        mock = Mock()
        cb = LanguageComboBox(None, ["sl", "en", "sv", "fi"], "fi", True, mock)
        self.assertEqual(
            [_DEFAULT_NONE, "English", "Finnish", "Slovenian", "Swedish"],
            [cb.itemText(i) for i in range(cb.count())],
        )
        mock.assert_not_called()
        simulate.combobox_activate_item(cb, "Slovenian")
        mock.assert_called_once_with("sl")
        mock.reset_mock()
        simulate.combobox_activate_item(cb, _DEFAULT_NONE)
        mock.assert_called_once_with(None)


@patch(SF_LIST, new=Mock(return_value=SERVER_FILES))
class TestUDPipeComboBox(WidgetTest):
    ITEMS = ["English", "English (lines)", "English (partut)", "Lithuanian",
             "Portuguese", "Slovenian", "Slovenian (sst)"]

    def test_basic_setup(self):
        mock = Mock()
        cb = UDPipeComboBox(None, "pt", "en", mock)
        self.assertEqual(7, cb.count())
        self.assertEqual(self.ITEMS, [cb.itemText(i) for i in range(cb.count())])
        self.assertEqual("Portuguese", cb.currentText())

    def test_set_current_language(self):
        mock = Mock()
        cb = UDPipeComboBox(None, "pt", "en", mock)
        self.assertEqual("Portuguese", cb.currentText())
        cb.set_current_language("sl")
        self.assertEqual("Slovenian", cb.currentText())
        cb.set_current_language("abc")  # language not in list - keep current seleciton
        self.assertEqual("Slovenian", cb.currentText())

    def test_set_language_to_default(self):
        """In case current item not in dropdown anymore set language to default"""
        mock = Mock()
        cb = UDPipeComboBox(None, "pt", "en", mock)
        self.assertEqual("Portuguese", cb.currentText())
        # when no default language in the dropdown set to first
        cb.removeItem(0)
        x = cb._UDPipeComboBox__items
        cb._UDPipeComboBox__items = x[:3] + x[4:]
        cb.showPopup()
        self.assertEqual("English", cb.currentText())

    def test_change_item(self):
        mock = Mock()
        cb = UDPipeComboBox(None, "pt", "en", mock)
        self.assertEqual(self.ITEMS, [cb.itemText(i) for i in range(cb.count())])
        mock.assert_not_called()
        simulate.combobox_activate_item(cb, "Slovenian")
        mock.assert_called_once_with("sl")


if __name__ == "__main__":
    unittest.main()
