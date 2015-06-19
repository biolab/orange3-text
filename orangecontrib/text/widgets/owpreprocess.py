from PyQt4 import QtCore, QtGui

from Orange.widgets.widget import OWWidget
from Orange.widgets.settings import Setting
from Orange.widgets import gui
from orangecontrib.text.preprocess import Preprocessor


class Output:
    PREPROCESSOR = "Preprocessor"

class OWPreprocess(OWWidget):
    name = "Preprocess"
    description = "Choose the pre-processing options and return a Preprocessor object."
    icon = "icons/TextPreprocess.svg"
    priority = 30

    outputs = [(Output.PREPROCESSOR, Preprocessor)]
    want_main_area = False

    include_punctuation = Setting(False)
    lowercase = Setting(True)
    remove_stpwrds = Setting(True)
    transformation_opt = Setting(["(none)", "Stemmer", "Lemmatizer"])

    def __init__(self):
        super().__init__()

        self.transformation = None

        # Settings.
        settings_box = gui.widgetBox(self.controlArea, "Basic options", addSpace=True)

        gui.checkBox(settings_box, self, "include_punctuation", "Include punctuation")
        gui.checkBox(settings_box, self, "lowercase", "To lowercase")
        gui.checkBox(settings_box, self, "remove_stpwrds", "Remove stopwords")

        # Transformation.
        transformation_box = gui.widgetBox(self.controlArea, "Transformation", addSpace=True)

        self.trans_combo = QtGui.QComboBox(settings_box)
        transformation_box.layout().addWidget(self.trans_combo, QtCore.Qt.AlignRight)
        self.fill_transformation_options()  # Add available options.
        self.trans_combo.activated[int].connect(self.select_transformation)

        gui.button(self.controlArea, self, "&Apply", callback=self.apply, default=True)

        self.apply()

    def select_transformation(self, n):
        if n > 0:
            self.transformation = self.transformation_opt[n]
        else:
            self.transformation = None

    def fill_transformation_options(self):
        self.trans_combo.clear()
        for trans in self.transformation_opt:
            if trans == "(none)":
                self.trans_combo.addItem("(none)")
            else:
                self.trans_combo.addItem(trans)

    def apply(self):
        # TODO change this to custom stopwords
        if self.remove_stpwrds:
            sw = 'english'
        pp = Preprocessor(incl_punct=self.include_punctuation, trans=self.transformation,
                          lowercase=self.lowercase, stop_words=sw)
        self.send(Output.PREPROCESSOR, pp)