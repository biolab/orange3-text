from AnyQt.QtWidgets import QGroupBox, QVBoxLayout

from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.widget import OWWidget, Input, Output
from orangecontrib.text.corpus import Corpus


class OWBaseVectorizer(OWWidget):
    """ A base class for feature extraction methods.

    Notes:
        Ensure that `create_configuration_layout` and `update_method` are overwritten.
    """
    # Input/output
    class Inputs:
        corpus = Input("Corpus", Corpus)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False
    resizing_enabled = False

    # Settings
    autocommit = settings.Setting(True)
    hidden_cb = settings.Setting(True)

    Method = NotImplemented

    def __init__(self):
        super().__init__()
        self.corpus = None
        self.method = None
        self.new_corpus = None
        self.new_attrs = None

        box = QGroupBox(title='Options')
        box.setLayout(self.create_configuration_layout())
        self.controlArea.layout().addWidget(box)

        output_layout = gui.hBox(self.controlArea)
        gui.checkBox(output_layout, self, "hidden_cb", "Hide bow attributes",
                     callback=self.hide_attrs)

        buttons_layout = gui.hBox(self.controlArea)
        gui.auto_commit(buttons_layout, self, 'autocommit', 'Commit', box=False)
        self.update_method()

    @Inputs.corpus
    def set_data(self, data):
        self.corpus = data
        self.invalidate()

    def hide_attrs(self):
        if self.new_corpus:
            new_domain = self.new_corpus.domain
            for f in new_domain.attributes:
                if f.name in self.new_attrs:
                    f.attributes['hidden'] = self.hidden_cb
            self.new_corpus = self.new_corpus.transform(new_domain)
            self.commit()

    def commit(self):
        self.Outputs.corpus.send(self.new_corpus)

    def apply(self):
        if self.corpus is not None:
            self.new_corpus = self.method.transform(self.corpus)
            self.new_attrs = {f.name for f in self.new_corpus.domain.attributes} \
                - {f.name for f in self.corpus.domain.attributes}

    def invalidate(self):
        self.apply()
        self.hide_attrs()
        self.commit()

    def update_method(self):
        self.method = self.Method()

    def on_change(self):
        self.update_method()
        self.invalidate()

    def send_report(self):
        self.report_items(self.method.report())

    def create_configuration_layout(self):
        return QVBoxLayout()
