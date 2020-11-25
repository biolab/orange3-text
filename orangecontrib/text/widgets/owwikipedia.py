from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication, QGridLayout, QLabel

from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.widget import OWWidget, Msg, Output
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.language_codes import lang2code, code2lang
from orangecontrib.text.widgets.utils import ComboBox, ListEdit, CheckListLayout, asynchronous

from orangecontrib.text.wikipedia import WikipediaAPI


class OWWikipedia(OWWidget):
    """ Get articles from wikipedia. """
    name = 'Wikipedia'
    priority = 160
    icon = 'icons/Wikipedia.svg'

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False
    resizing_enabled = False

    label_width = 1
    widgets_width = 2

    attributes = [feat.name for feat in WikipediaAPI.string_attributes]
    text_includes = settings.Setting([feat.name for feat in WikipediaAPI.string_attributes])

    query_list = settings.Setting([])
    language = settings.Setting('en')
    articles_per_query = settings.Setting(10)

    info_label = 'Articles count {:d}'

    class Error(OWWidget.Error):
        api_error = Msg('API error: {}')

    class Warning(OWWidget.Warning):
        no_text_fields = Msg('Text features are inferred when none are selected.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.api = WikipediaAPI(on_error=self.Error.api_error)
        self.result = None

        query_box = gui.hBox(self.controlArea, 'Query')

        # Queries configuration
        layout = QGridLayout()
        layout.setSpacing(7)

        row = 0
        self.query_edit = ListEdit(self, 'query_list', "Each line represents a "
                                                  "separate query.", 100, self)
        layout.addWidget(QLabel('Query word list:'), row, 0, 1, self.label_width)
        layout.addWidget(self.query_edit, row, self.label_width, 1,
                         self.widgets_width)

        # Language
        row += 1
        language_edit = ComboBox(self, 'language', tuple(sorted(lang2code.items())))
        layout.addWidget(QLabel('Language:'), row, 0, 1, self.label_width)
        layout.addWidget(language_edit, row, self.label_width, 1, self.widgets_width)

        # Articles per query
        row += 1
        layout.addWidget(QLabel('Articles per query:'), row, 0, 1, self.label_width)
        slider = gui.valueSlider(query_box, self, 'articles_per_query', box='',
                                 values=[1, 3, 5, 10, 25])
        layout.addWidget(slider.box, row, 1, 1, self.widgets_width)

        query_box.layout().addLayout(layout)
        self.controlArea.layout().addWidget(query_box)

        self.controlArea.layout().addWidget(
            CheckListLayout('Text includes', self, 'text_includes', self.attributes, cols=2,
                            callback=self.set_text_features))

        self.info_box = gui.hBox(self.controlArea, 'Info')
        self.result_label = gui.label(self.info_box, self, self.info_label.format(0))

        self.button_box = gui.hBox(self.controlArea)

        self.search_button = gui.button(self.button_box, self, 'Search', self.start_stop)
        self.search_button.setFocusPolicy(Qt.NoFocus)

    def start_stop(self):
        if self.search.running:
            self.search.stop()
        else:
            self.search()

    @asynchronous
    def search(self):
        return self.api.search(lang=self.language, queries=self.query_list,
                               articles_per_query=self.articles_per_query,
                               on_progress=self.progress_with_info,
                               should_break=self.search.should_break)

    @search.callback(should_raise=False)
    def progress_with_info(self, progress, n_retrieved):
        self.progressBarSet(100 * progress)
        self.result_label.setText(self.info_label.format(n_retrieved))

    @search.on_start
    def on_start(self):
        self.Error.api_error.clear()
        self.progressBarInit()
        self.search_button.setText('Stop')
        self.result_label.setText(self.info_label.format(0))
        self.Outputs.corpus.send(None)

    @search.on_result
    def on_result(self, result):
        self.result = result
        self.result_label.setText(self.info_label.format(len(result) if result else 0))
        self.search_button.setText('Search')
        self.set_text_features()
        self.progressBarFinished()

    def set_text_features(self):
        self.Warning.no_text_fields.clear()
        if not self.text_includes:
            self.Warning.no_text_fields()

        if self.result is not None:
            vars_ = [var for var in self.result.domain.metas if var.name in self.text_includes]
            self.result.set_text_features(vars_ or None)
            self.Outputs.corpus.send(self.result)

    def send_report(self):
        if self.result:
            items = (('Language', code2lang[self.language]),
                     ('Query', self.query_edit.toPlainText()),
                     ('Articles count', len(self.result)))
            self.report_items('Query', items)


if __name__ == '__main__':
    app = QApplication([])
    widget = OWWikipedia()
    widget.show()
    app.exec()
    widget.saveSettings()
