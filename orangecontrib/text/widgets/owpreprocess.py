import bisect
import os
import pkg_resources

from PyQt4.QtGui import (
    QWidget, QButtonGroup, QRadioButton, QListView, QPushButton, QFrame,
    QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QCheckBox, QLabel,
    QCursor, QIcon,  QStandardItemModel, QStandardItem, QStyle,
    QStylePainter, QStyleOptionFrame, QPixmap,
    QApplication, QDrag
)

from PyQt4 import QtGui
from PyQt4.QtCore import (
    Qt, QObject, QEvent, QSize, QModelIndex, QMimeData
)
from PyQt4.QtCore import pyqtSignal as Signal, pyqtSlot as Slot

from Orange.widgets.widget import OWWidget
from orangecontrib.text.preprocess import Preprocessor, \
    PorterStemmer as ps, Lemmatizer as lm, SnowballStemmer as ss
from orangecontrib.text.corpus import Corpus

from Orange.widgets import gui, settings


class Input:
    CORPUS = "Corpus"


class Output:
    PP_CORPUS = "Corpus"


def remove_values_from_list(input, val):
    return [x for i, x in enumerate(input) if x == val]


WIDGET_TITLES = ['Tokenization', 'Token filtering', 'Stemming', 'Token capitalization']


class BaseEditor(QWidget):
    """
    Base widget for editing preprocessor's parameters.
    """
    # Emitted when parameters have changed.
    changed = Signal()
    # Emitted when parameters were edited/changed as a result of user interaction.
    edited = Signal()

    def set_parameters(self, parameters):
        """
        Set parameters.
        :param parameters: Parameters as a dictionary. It is up
            to subclasses to properly parse the contents.
        :type parameters: dict
        """
        raise NotImplementedError

    def parameters(self):
        """
        Return the parameters as a dictionary.
        """
        raise NotImplementedError

    @staticmethod
    def create_instance(params):
        """
        Create the Preprocessor instance given the stored parameters dict.
        :param params: Parameters as returned by `parameters`.
        :type params: dict
        """
        raise NotImplementedError


class TokenizerEditor(BaseEditor):
    """
    Editor for orangecontrib.text.preprocess tokenizing.
    """
    # Tokenizing methods
    NLTKTokenizer, TwitterTokenizer = 0, 1

    # Keywords for the pp objects.
    TokenizerObjects = {
        NLTKTokenizer: False,
        TwitterTokenizer: True
    }
    # Names of the pp objects.
    Names = {
        NLTKTokenizer: "NLTK tokenizer",
        TwitterTokenizer: "Twitter tokenizer"
    }

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)

        # Default tokenizer. Cannot be None.
        self.__method = TokenizerEditor.NLTKTokenizer

        # Layout.
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Set the radio buttons.
        self.__group = group = QButtonGroup(self, exclusive=True)
        for method in [self.NLTKTokenizer, self.TwitterTokenizer]:
            rb = QRadioButton(
                self, text=self.Names[method],
                checked=self.__method == method
            )
            if method == self.TwitterTokenizer:
                rb.setEnabled(False)    # Disable until the Twitter tokenizer is available.
            layout.addWidget(rb)
            group.addButton(rb, method)
        group.buttonClicked.connect(self.__on_button_clicked)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def set_method(self, method):
        """
        Checks the corresponding button and emits a change signal.
        """
        if self.__method != method:
            self.__method = method
            b = self.__group.button(method)
            b.setChecked(True)
            self.changed.emit()

    def set_parameters(self, params):
        method = params.get("method", 0)
        self.set_method(method)

    def parameters(self):
        return {"method": self.__method}

    def __on_button_clicked(self):
        """
        Whenever user clicks a button.
        """
        method = self.__group.checkedId()
        if method != self.__method:
            self.set_method(self.__group.checkedId())
            self.edited.emit()

    @staticmethod
    def create_instance(params):
        """
        Reads the parameters and generates a setting.
        """
        method = params.pop("method", TokenizerEditor.NLTKTokenizer)
        used_method = TokenizerEditor.TokenizerObjects[method]
        return {"use_twitter_tokenizer": used_method}


class TransformationEditor(BaseEditor):
    """
    Editor for orangecontrib.text.preprocess morphological transformations.
    """
    # Transformation methods
    NoTrans, PorterStemmer, SnowballStemmer, Lemmatizer = 0, 1, 2, 3

    TransformationObjects = {
        NoTrans: None,
        PorterStemmer: ps,
        SnowballStemmer: ss,
        Lemmatizer: lm
    }
    Names = {
        NoTrans: "None",
        PorterStemmer: "Porter stemmer",
        SnowballStemmer: "Snowball stemmer",
        Lemmatizer: "Lemmatizer"
    }

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        # Default transformation, should we choose to transform at all.
        self.__method = TransformationEditor.PorterStemmer

        # Layout.
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Set the radio buttons.
        self.__group = group = QButtonGroup(self, exclusive=True)
        for method in [self.PorterStemmer, self.SnowballStemmer, self.Lemmatizer]:
            rb = QRadioButton(
                self, text=self.Names[method],
                checked=self.__method == method
            )
            layout.addWidget(rb)
            group.addButton(rb, method)
        group.buttonClicked.connect(self.__on_button_clicked)
        """
        # HLine divider.
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(h_line)

        # Set the lowercase checkbox.
        self.lowercase_checkbox = QCheckBox("Lowercase")
        self.lowercase_checkbox.clicked.connect(self.__on_chbox_clicked)
        layout.addWidget(self.lowercase_checkbox)
        """
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def set_method(self, method):
        if self.__method != method:
            self.__method = method
            b = self.__group.button(method)
            b.setChecked(True)
            self.changed.emit()
    """
    def set_lowercase_flag(self, flag):
        if self.__use_lowercase != flag:
            self.__use_lowercase = flag
            self.lowercase_checkbox.setChecked(flag)
            self.changed.emit()
    """

    def set_parameters(self, params):
        method = params.get("method", 1)
        #lwcase = params.get("lwcase", False)
        #self.set_lowercase_flag(lwcase)
        self.set_method(method)

    def parameters(self):
        return {"method": self.__method}#, "lwcase": self.__use_lowercase}
    """
    def __on_chbox_clicked(self):
        # Whenever user clicks the checkbox.
        flag = self.lowercase_checkbox.isChecked()
        if flag != self.__use_lowercase:
            self.set_lowercase_flag(self.lowercase_checkbox.isChecked())
            self.edited.emit()
    """

    def __on_button_clicked(self):
        # On user 'method' button click.
        method = self.__group.checkedId()
        if method != self.__method:
            self.set_method(self.__group.checkedId())
            self.edited.emit()

    @staticmethod
    def create_instance(params):
        """
        Reads the parameters and generates a setting.
        """
        method = params.pop("method", TransformationEditor.PorterStemmer)
        #lwcase = params.pop("lwcase", False)
        used_method = TransformationEditor.TransformationObjects[method]
        return {"transformation": used_method}#, "lowercase": lwcase}


class StopWordEditor(BaseEditor):
    """
    Editor for orangecontrib.text.preprocess stop word removal.
    """
    recent_sw_files = []

    # Stop word sources.
    Default, Custom = 0, 1
    StopWordSources = {
        Default: "english",
        Custom: {'recent_sw_files': [], 'loaded_sw_file': []}
    }
    Names = {
        Default: "Default stop words set (English)",
        Custom: "Custom stop words set"
    }

    dlgFormats = "Only text files (*.txt)"
    current_file_path = None

    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)
        # Default source.
        self.__source = StopWordEditor.Default

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Radio buttons.
        self.__group = group = QButtonGroup(self, exclusive=True)
        for source in [self.Default, self.Custom]:
            rb = QRadioButton(
                self, text=self.Names[source],
                checked=self.__source == source
            )
            layout.addWidget(rb)
            group.addButton(rb, source)
        group.buttonClicked.connect(self.__on_button_clicked)

        # Custom sw browser.
        h_box = QHBoxLayout()
        h_box.setContentsMargins(20, 0, 0, 0)

        self.sw_file_combo = QtGui.QComboBox()
        h_box.addWidget(self.sw_file_combo)
        self.sw_file_combo.activated[int].connect(self.select_file)
        self.set_file_list()

        self.browse_button = QPushButton(self)
        self.browse_button.clicked.connect(self.browse_file)
        self.browse_button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DirOpenIcon))
        self.browse_button.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        h_box.addWidget(self.browse_button)

        # Reload button
        reload_button = QPushButton(self)
        reload_button.clicked.connect(self.__on_reload_button_clicked)
        reload_button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        h_box.addWidget(reload_button)

        layout.addLayout(h_box)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def set_source(self, source):
        # Determine whether the browse controls should be enabled.
        self.enable_browsing(source)

        if self.__source != source:
            self.__source = source
            b = self.__group.button(source)
            b.setChecked(True)

            self.changed.emit()

    def set_parameters(self, params):
        """
        Parses the handled parameters and updates the control accordingly.
        :param params: Dictionary of the control's parameters.
        :Type params: dict
        :return: None
        """
        source_index = params.get("source", 0)
        # Check the appropriate radio button.
        self.set_source(source_index)

        # If custom sw file.
        if source_index == 1:
            self.set_file_list()

    def parameters(self):
        return {"source": self.__source}

    # --- File selection.
    def select_file(self, n):
        if n < len(self.recent_sw_files):
            name = self.recent_sw_files[n]
            del self.recent_sw_files[n]
            self.recent_sw_files.insert(0, name)

        if len(self.recent_sw_files) > 0:
            self.set_file_list()
            self.open_file(self.recent_sw_files[0])

    def set_file_list(self):
        self.sw_file_combo.clear()
        if not self.recent_sw_files:
            self.sw_file_combo.addItem("(none)")
            self.update_custom_source()
        else:
            for file in self.recent_sw_files:
                self.sw_file_combo.addItem(os.path.split(file)[1])
            self.update_custom_source()

    def browse_file(self):
        """
        Opens the file browser, starting at the home directory.
        :return: None
        """
        start_file = os.path.expanduser("~/")
        # Get the file path from the browser window.
        path = QtGui.QFileDialog.getOpenFileName(
            self, 'Open a stop words source', start_file, self.dlgFormats)
        if not path:
            return

        if path in self.recent_sw_files:
            self.recent_sw_files.remove(path)
        self.recent_sw_files.insert(0, path)
        self.set_file_list()
        self.open_file(path)

    def update_custom_source(self, stop_words=None):
        self.StopWordSources[self.Custom]['recent_sw_files'] = self.recent_sw_files
        if stop_words is not None:
            self.StopWordSources[self.Custom]['loaded_sw_file'] = stop_words

    def open_file(self, path):
        try:
            with open(path) as f:     # Read most recent.
                new_stop_words = [sw.strip() for sw in f.read().splitlines()]
                self.update_custom_source(new_stop_words)
        except Exception as err:    # Raise an exception otherwise.
            self.update_custom_source(err)

    def set_recent_sw_files(self, files):
        """
        Sets this pp element's recently used files.
        :param files: A list of file paths recently used.
        :type files: list
        :return: None
        """
        self.recent_sw_files = files
        self.set_file_list()
    # END File selection.

    def enable_browsing(self, source):
        browse_flag = source == 1
        self.browse_button.setEnabled(browse_flag)
        self.sw_file_combo.setEnabled(browse_flag)

        # If available, load the custom file.
        self.select_file(0)

    def __on_reload_button_clicked(self):
        # When the user clicks the reload button.
        if self.recent_sw_files:
            self.open_file(self.recent_sw_files[0])
            self.changed.emit()
            self.edited.emit()


    def __on_button_clicked(self):
        # on user 'source' button click
        source = self.__group.checkedId()
        self.enable_browsing(source)

        if source != self.__source:
            self.set_source(self.__group.checkedId())
            self.edited.emit()

    @staticmethod
    def create_instance(params):
        source = params.pop("source")
        used_source = StopWordEditor.StopWordSources[source]

        if isinstance(used_source, dict):   # If custom, use the loaded file.
            used_source = used_source['loaded_sw_file']

        return {"stop_words": used_source}


class CasingEditor(BaseEditor):
    """
    Editor for orangecontrib.text.preprocess lowercasing.
    """
    def __init__(self, parent=None, **kwargs):
        BaseEditor.__init__(self, parent, **kwargs)

        # Layout.
        layout = QVBoxLayout()
        layout.addWidget(QLabel('All characters in tokens, will be lower case.'))
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

    def set_method(self):
        self.changed.emit()

    def set_parameters(self, params):
        self.set_method()

    def parameters(self):
        return {'lowercase': True}

    @staticmethod
    def create_instance(params):
        """
        Reads the parameters and generates a setting.
        """
        return {'lowercase': True}


class PreprocessAction(object):
    """
    This object holds the info about a single pre-processing option
    from the list.
    """
    def __init__(self, qualname, title, icon, summary, view_class):
        self.qualname = qualname        # Unique identifier.
        self.title = title              # Control title.
        self.icon = icon                # Control icon.
        self.summary = summary          # Control summary.
        self.view_class = view_class    # A class that extends BaseEditor.


def icon_path(basename):
    return pkg_resources.resource_filename(__name__, "icons/" + basename)


# Mandatory preprocessors.
DEFAULT_PREPROCESSORS = [
    PreprocessAction(
        "orangecontrib.text.preprocess.tokenizer", WIDGET_TITLES[0], icon_path("TextPreprocess.svg"),
        "Splits the text into single word tokens.",
        TokenizerEditor
    )
]

PREPROCESSORS = [
    PreprocessAction(
        "orangecontrib.text.preprocess.stemmer", WIDGET_TITLES[2], icon_path("TextPreprocess.svg"),
        "Performs the chosen morphological transformation on the text tokens.",
        TransformationEditor
    ),
    PreprocessAction(
        "orangecontrib.text.preprocess.stopwords", WIDGET_TITLES[1], icon_path("TextPreprocess.svg"),
        "Removes stop words from text.",
        StopWordEditor
    ),
    PreprocessAction(
        "orangecontrib.text.preprocess.lowercase", WIDGET_TITLES[3], icon_path("TextPreprocess.svg"),
        "Lowercases all characters in the tokens.",
        CasingEditor
    )
]

#: Qt.ItemRole holding the PreprocessAction instance
DescriptionRole = Qt.UserRole
#: Qt.ItemRole storing the preprocess parameters
ParametersRole = Qt.UserRole + 1


class Controller(QObject):
    """
    Controller for displaying/editing QAbstractItemModel using SequenceFlow.
    It creates/deletes updates the widgets in the view when the model
    changes, as well as interprets drop events (with appropriate mime data)
    onto the view, modifying the model appropriately.
    :param view: The view to control (required).
    :type view: SequenceFlow
    :param model: A list model.
    :type model: QAbstractItemModel
    :param parent: The controller's parent.
    :type parent: QObject
    """
    MimeType = "application/x-qwidget-ref"

    def __init__(self, view, model=None, parent=None):
        super().__init__(parent)
        self._model = None

        self.view = view
        view.installEventFilter(self)
        view.widgetCloseRequested.connect(self._closeRequested)
        view.widgetMoved.connect(self._widgetMoved)

        # gruesome
        self._setDropIndicatorAt = view._SequenceFlow__setDropIndicatorAt
        self._insertIndexAt = view._SequenceFlow__insertIndexAt

        if model is not None:
            self.setModel(model)

    def __connect(self, model):
        model.dataChanged.connect(self._dataChanged)
        model.rowsInserted.connect(self._rowsInserted)
        model.rowsRemoved.connect(self._rowsRemoved)
        model.rowsMoved.connect(self._rowsMoved)

    def __disconnect(self, model):
        model.dataChanged.disconnect(self._dataChanged)
        model.rowsInserted.disconnect(self._rowsInserted)
        model.rowsRemoved.disconnect(self._rowsRemoved)
        model.rowsMoved.disconnect(self._rowsMoved)

    def setModel(self, model):
        """
        Set the model for the view.
        :param model: Model of the view.
        :type model: QAbstractItemModel.
        """
        if self._model is model:
            return

        if self._model is not None:
            self.__disconnect(self._model)

        self._clear()
        self._model = model

        if self._model is not None:
            self._initialize(model)
            self.__connect(model)

    def model(self):
        """
        Return the model.
        """
        return self._model

    def _initialize(self, model):
        for i in range(model.rowCount()):
            index = model.index(i, 0)
            self._insertWidgetFor(i, index)

    def _clear(self):
        self.view.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(self.MimeType) and \
                self.model() is not None:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            return True
        else:
            return False

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(self.MimeType) and \
                self.model() is not None:
            event.accept()
            self._setDropIndicatorAt(event.pos())
            return True
        else:
            return False

    def dragLeaveEvent(self, event):
        return False

    def dropEvent(self, event):
        if event.mimeData().hasFormat(self.MimeType) and \
                self.model() is not None:
            model = self.model()

            # Do a check for duplicates. Reject drop if found.
            for i in range(0, model.rowCount()):
                if event.mimeData().data("application/x-qwidget-ref") == \
                        model.index(i, 0).data(DescriptionRole).qualname:
                    return False

            # Create and insert appropriate widget.
            self._setDropIndicatorAt(None)
            row = self._insertIndexAt(event.pos())

            diddrop = model.dropMimeData(
                event.mimeData(), Qt.CopyAction, row, 0, QModelIndex())

            if diddrop:
                event.accept()
            return True
        else:
            return False

    def eventFilter(self, view, event):
        if view is not self.view:
            return False

        if event.type() == QEvent.DragEnter:
            return self.dragEnterEvent(event)
        elif event.type() == QEvent.DragMove:
            return self.dragMoveEvent(event)
        elif event.type() == QEvent.DragLeave:
            return self.dragLeaveEvent(event)
        elif event.type() == QEvent.Drop:
            return self.dropEvent(event)
        else:
            return super().eventFilter(view, event)

    def _dataChanged(self, topleft, bottomright):
        model = self.model()
        widgets = self.view.widgets()

        top, left = topleft.row(), topleft.column()
        bottom, right = bottomright.row(), bottomright.column()
        assert left == 0 and right == 0

        for row in range(top, bottom + 1):
            self.setWidgetData(widgets[row], model.index(row, 0))

    def _rowsInserted(self, parent, start, end):
        model = self.model()

        for row in range(start, end + 1):
            index = model.index(row, 0, parent)
            self._insertWidgetFor(row, index)

    def _rowsRemoved(self, parent, start, end):
        for row in reversed(range(start, end + 1)):
            self._removeWidgetFor(row, None)

    def _rowsMoved(self, srcparetn, srcstart, srcend,
                   dstparent, dststart, dstend):
        raise NotImplementedError

    def _closeRequested(self, row):
        model = self.model()
        assert 0 <= row < model.rowCount()
        model.removeRows(row, 1, QModelIndex())

    def _widgetMoved(self, from_, to):
        # The widget in the view were already swaped, so
        # we must disconnect from the model when moving the rows.
        # It would be better if this class would also filter and
        # handle internal widget moves.
        model = self.model()
        self.__disconnect(model)
        try:
            model.moveRow
        except AttributeError:
            data = model.itemData(model.index(from_, 0))
            model.removeRow(from_, QModelIndex())
            model.insertRow(to, QModelIndex())

            model.setItemData(model.index(to, 0), data)
            assert model.rowCount() == len(self.view.widgets())
        else:
            model.moveRow(QModelIndex(), from_, QModelIndex(), to)
        finally:
            self.__connect(model)

    def _insertWidgetFor(self, row, index):
        widget = self.createWidgetFor(index)
        self.view.insertWidget(row, widget, title=index.data(Qt.DisplayRole))
        self.view.setIcon(row, index.data(Qt.DecorationRole))
        self.setWidgetData(widget, index)
        widget.edited.connect(self.__edited)

    def _removeWidgetFor(self, row, index):
        widget = self.view.widgets()[row]
        self.view.removeWidget(widget)
        widget.edited.disconnect(self.__edited)
        widget.deleteLater()

    def createWidgetFor(self, index):
        """
        Create a QWidget instance for the index (:class:`QModelIndex`)
        """
        definition = index.data(DescriptionRole)
        widget = definition.view_class()
        return widget

    def setWidgetData(self, widget, index):
        """
        Set/update the widget state from the model at index.
        """
        params = index.data(ParametersRole)
        if not isinstance(params, dict):
            params = {}
        widget.set_parameters(params)

    def setModelData(self, widget, index):
        """
        Get the data from the widget state and set/update the model at index.
        """
        params = widget.parameters()
        assert isinstance(params, dict)
        self._model.setData(index, params, ParametersRole)

    @Slot()
    def __edited(self,):
        widget = self.sender()
        row = self.view.indexOf(widget)
        index = self.model().index(row, 0)
        self.setModelData(widget, index)


class SequenceFlow(QWidget):
    """
    A re-orderable list of widgets.
    """
    # Emitted when the user clicks the Close button in the header
    widgetCloseRequested = Signal(int)
    # Emitted when the user moves/drags a widget to a new location.
    widgetMoved = Signal(int, int)
    # A list holding recent sw files.
    recent_sw_files = []

    class Frame(QtGui.QDockWidget):
        """
        Widget frame with a handle.
        """
        closeRequested = Signal()

        def __init__(self, parent=None, widget=None, title=None, is_closable=True, **kwargs):

            super().__init__(parent, **kwargs)
            if is_closable:
                self.setFeatures(QtGui.QDockWidget.DockWidgetClosable)
            else:
                self.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
            self.setAllowedAreas(Qt.NoDockWidgetArea)

            self.__title = ""
            self.__icon = ""
            self.__focusframe = None

            self.__deleteaction = QtGui.QAction(
                "Remove", self, shortcut=QtGui.QKeySequence.Delete,
                enabled=False, triggered=self.closeRequested
            )
            self.addAction(self.__deleteaction)

            if widget is not None:
                self.setWidget(widget)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

            if title:
                self.setTitle(title)

            self.setFocusPolicy(Qt.ClickFocus | Qt.TabFocus)

        def setTitle(self, title):
            if self.__title != title:
                self.__title = title
                self.setWindowTitle(title)
                self.update()

        def getTitle(self):
            if self.__title:
                return self.__title
            return None

        def setIcon(self, icon):
            icon = QIcon(icon)
            if self.__icon != icon:
                self.__icon = icon
                self.setWindowIcon(icon)
                self.update()

        def paintEvent(self, event):
            painter = QStylePainter(self)
            opt = QStyleOptionFrame()
            opt.init(self)
            painter.drawPrimitive(QStyle.PE_FrameDockWidget, opt)
            painter.end()

            super().paintEvent(event)

        def focusInEvent(self, event):
            event.accept()
            self.__focusframe = QtGui.QFocusFrame(self)
            self.__focusframe.setWidget(self)
            self.__deleteaction.setEnabled(True)

        def focusOutEvent(self, event):
            event.accept()
            self.__focusframe.deleteLater()
            self.__focusframe = None
            self.__deleteaction.setEnabled(False)

        def closeEvent(self, event):
            super().closeEvent(event)
            event.ignore()
            self.closeRequested.emit()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.__dropindicator = QSpacerItem(
            16, 16, QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.__dragstart = (None, None, None)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.__flowlayout = QVBoxLayout()
        layout.addLayout(self.__flowlayout)
        layout.addSpacerItem(
            QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.setLayout(layout)
        self.setAcceptDrops(True)

    def set_recent_sw_files(self, files):
        self.recent_sw_files = files

    def sizeHint(self):
        if self.widgets():
            return super().sizeHint()
        else:
            return QSize(150, 100)

    def addWidget(self, widget, title):
        """
        Add `widget` with `title` to list of widgets (in the last position).
        :param widget: Widget instance to add.
        :type widget: QWidget
        :param title: The title of the widget.
        :type title: str
        """
        index = len(self.widgets())
        self.insertWidget(index, widget, title)

    def insertWidget(self, index, widget, title):
        """
        Insert `widget` with `title` at `index`.
        :param index: The index at which to insert the new widget.
        :type index: int
        :param widget: Widget instance to add.
        :type widget: QWidget
        :param title: The title of the widget.
        :type title: str
        """
        layout = self.__flowlayout
        frames = [item.widget() for item in self.layout_iter(layout)
                  if item.widget()]

        if title == WIDGET_TITLES[1]:
            # Set the recent files.
            widget.set_recent_sw_files(self.recent_sw_files)

        if title == WIDGET_TITLES[0]:    # Mandatory widget -> Disable removal.
            frame = SequenceFlow.Frame(widget=widget, title=title, is_closable=False)
        else:
            frame = SequenceFlow.Frame(widget=widget, title=title)
            frame.closeRequested.connect(self.__closeRequested)

        if 0 < index < len(frames):
            # find the layout index of a widget occupying the current
            # index'th slot.
            insert_index = layout.indexOf(frames[index])
        elif index == 0:
            insert_index = 0
        elif index < 0 or index >= len(frames):
            insert_index = layout.count()
        else:
            assert False

        layout.insertWidget(insert_index, frame)

        frame.installEventFilter(self)

    def removeWidget(self, widget):
        """
        Remove widget from the list.
        :param widget: Widget to remove.
        :type widget: QWidget
        """
        layout = self.__flowlayout
        frame = self.__widgetFrame(widget)
        if frame is not None:
            frame.setWidget(None)
            widget.setVisible(False)
            widget.setParent(None)
            layout.takeAt(layout.indexOf(frame))
            frame.hide()
            frame.deleteLater()

    def clear(self):
        """
        Clear the list (remove all widgets).
        """
        for w in reversed(self.widgets()):
            self.removeWidget(w)

    def widgets(self):
        """
        Return a list of all `widgets`.
        """
        layout = self.__flowlayout
        items = (layout.itemAt(i) for i in range(layout.count()))
        return [item.widget().widget()
                for item in items if item.widget() is not None]

    def indexOf(self, widget):
        """
        Return the index (logical position) of `widget`
        """
        widgets = self.widgets()
        return widgets.index(widget)

    def setTitle(self, index, title):
        """
        Set title for `widget` at `index`.
        """
        widget = self.widgets()[index]
        frame = self.__widgetFrame(widget)
        frame.setTitle(title)

    def setIcon(self, index, icon):
        widget = self.widgets()[index]
        frame = self.__widgetFrame(widget)
        frame.setIcon(icon)

    def dropEvent(self, event):
        layout = self.__flowlayout
        index = self.__insertIndexAt(self.mapFromGlobal(QCursor.pos()))

        if event.mimeData().hasFormat("application/x-internal-move") and \
                event.source() is self:
            # Complete the internal move
            frame, oldindex, _ = self.__dragstart
            # Remove the drop indicator spacer item before re-inserting
            # the frame
            self.__setDropIndicatorAt(None)
            if oldindex != index:
                layout.insertWidget(index, frame)
                if index > oldindex:
                    self.widgetMoved.emit(oldindex, index - 1)
                else:
                    self.widgetMoved.emit(oldindex, index)

                event.accept()

            self.__dragstart = None, None, None

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-internal-move") and \
                event.source() is self:
            assert self.__dragstart[0] is not None
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        pos = self.mapFromGlobal(QCursor.pos())
        self.__setDropIndicatorAt(pos)

    def dragLeaveEvent(self, event):
        self.__setDropIndicatorAt(None)

    def __setDropIndicatorAt(self, pos):
        # find the index where drop at pos would insert.
        index = -1
        layout = self.__flowlayout
        if pos is not None:
            index = self.__insertIndexAt(pos)
        spacer = self.__dropindicator
        currentindex = self.layout_index_of(layout, spacer)

        if currentindex != -1:
            item = layout.takeAt(currentindex)
            assert item is spacer
            if currentindex < index:
                index -= 1

        if index != -1:
            layout.insertItem(index, spacer)

    def __insertIndexAt(self, pos):
        y = pos.y()
        midpoints = [item.widget().geometry().center().y()
                     for item in self.layout_iter(self.__flowlayout)
                     if item.widget() is not None]
        index = bisect.bisect_left(midpoints, y)
        return index

    def __startInternalDrag(self, frame, hotSpot=None):
        drag = QDrag(self)
        pixmap = QPixmap(frame.size())
        frame.render(pixmap)

        transparent = QPixmap(pixmap.size())
        transparent.fill(Qt.transparent)
        painter = QtGui.QPainter(transparent)
        painter.setOpacity(0.35)
        painter.drawPixmap(0, 0, pixmap.width(), pixmap.height(), pixmap)
        painter.end()

        drag.setPixmap(transparent)
        if hotSpot is not None:
            drag.setHotSpot(hotSpot)
        mime = QMimeData()
        mime.setData("application/x-internal-move", "")
        drag.setMimeData(mime)
        return drag.exec_(Qt.MoveAction)

    def __widgetFrame(self, widget):
        layout = self.__flowlayout
        for item in self.layout_iter(layout):
            if item.widget() is not None and \
                    isinstance(item.widget(), SequenceFlow.Frame) and \
                    item.widget().widget() is widget:
                return item.widget()
        else:
            return None

    def __closeRequested(self):
        frame = self.sender()
        index = self.indexOf(frame.widget())
        self.widgetCloseRequested.emit(index)

    @staticmethod
    def layout_iter(layout):
        return (layout.itemAt(i) for i in range(layout.count()))

    @staticmethod
    def layout_index_of(layout, item):
        for i, item1 in enumerate(SequenceFlow.layout_iter(layout)):
            if item == item1:
                return i
        return -1


def get_mime_data(index_list):
    """
    Returns the mime data for the list item at that index.
    :param index_list: The list containing a single list index.
    :type index_list: list
    :return: QMimeData
    """
    assert len(index_list) == 1  # Check if there is only one index.
    index = index_list[0]
    q_name = index.data(DescriptionRole).qualname
    m = QMimeData()
    m.setData("application/x-qwidget-ref", q_name)
    return m


class OWPreprocess(OWWidget):
    name = "Preprocess text"
    description = "Construct a text pre-processing pipeline."
    icon = "icons/TextPreprocess.svg"
    priority = 30

    inputs = [(Input.CORPUS, Corpus, "set_data")]
    outputs = [(Output.PP_CORPUS, Corpus)]

    # Settings from previous runs. Actually chosen preprocessors.
    stored_settings = settings.Setting({})
    autocommit = settings.Setting(False)
    recent_sw_files = settings.Setting([])

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self._invalidated = False

        # List of available preprocessors (DescriptionRole : Description)
        self.preprocessors = QStandardItemModel()
        self.preprocessors.mimeData = get_mime_data

        box = gui.widgetBox(self.controlArea, "Preprocessors")
        view = QListView(selectionMode=QListView.SingleSelection,
                         dragEnabled=True,
                         dragDropMode=QListView.DragOnly)
        view.setModel(self.preprocessors)
        view.activated.connect(self.__activated)
        box.layout().addWidget(view)

        # Store key, value pairs for qual_name : pp_definition.
        self._pp_def_mapping = {pp_def.qualname: pp_def for pp_def in PREPROCESSORS + DEFAULT_PREPROCESSORS}

        # List of chosen preprocessors.
        self.preprocessormodel = None
        self.flow_view = SequenceFlow()
        self.flow_view.set_recent_sw_files(self.recent_sw_files)
        self.controller = Controller(self.flow_view, parent=self)

        self.scroll_area = QtGui.QScrollArea(
            verticalScrollBarPolicy=Qt.ScrollBarAsNeeded
        )
        self.scroll_area.viewport().setAcceptDrops(True)
        self.scroll_area.setWidget(self.flow_view)
        self.scroll_area.setWidgetResizable(True)
        self.mainArea.layout().addWidget(self.scroll_area)
        self.flow_view.installEventFilter(self)

        # Commit checkbox and commit button.
        box = gui.widgetBox(self.controlArea, "Output")
        gui.auto_commit(box, self, "autocommit", "Commit", box=False)

        # After the GUI is set up, initialize the pre-processing options.
        self.initialize_pp_options()
        # Restore the previous state if possible.
        self.restore_pp_pipeline()

    def initialize_pp_options(self):
        # Build the list of all available preprocessors.
        for pp in PREPROCESSORS:
            if pp.icon:
                icon = QIcon(pp.icon)
            else:
                icon = QIcon()
            item = QStandardItem(icon, pp.title)
            item.setToolTip(pp.summary or "")
            item.setData(pp, DescriptionRole)
            item.setFlags(Qt.ItemIsEnabled |
                          Qt.ItemIsSelectable |
                          Qt.ItemIsDragEnabled)
            self.preprocessors.appendRow([item])

    def restore_pp_pipeline(self):
        # Attempt to reload the previous state of the widget,
        # i.e. previously selected preprocessors.
        try:
            model = self.load(self.stored_settings)
        except Exception:
            model = self.load({})

        # Set the loaded pp model to the view.
        self.set_model(model)

        # Enforce default width constraint if no preprocessors
        # are instantiated (if the model is not empty the constraints
        # will be triggered by LayoutRequest event on the `flow_view`)
        if not model.rowCount():
            self.__update_size_constraint()

    def set_model(self, pp_model):
        if self.preprocessormodel:
            self.preprocessormodel.dataChanged.disconnect(self.commit)
            self.preprocessormodel.rowsInserted.disconnect(self.commit)
            self.preprocessormodel.rowsRemoved.disconnect(self.commit)
            self.preprocessormodel.deleteLater()

        self.preprocessormodel = pp_model
        self.controller.setModel(pp_model)
        if pp_model is not None:
            self.preprocessormodel.dataChanged.connect(self.commit)
            self.preprocessormodel.rowsInserted.connect(self.commit)
            self.preprocessormodel.rowsRemoved.connect(self.commit)

    def set_data(self, data=None):
        # Set the input data set.
        self.data = data

    def drop_mime_data(self, data, action, row, column, parent):
        if data.hasFormat("application/x-qwidget-ref") and action == Qt.CopyAction:
            # Get the pp definition from qual name.
            q_name = bytes(data.data("application/x-qwidget-ref")).decode()
            pp_def = self._pp_def_mapping[q_name]

            item = QStandardItem(pp_def.title)
            item.setData({}, ParametersRole)
            item.setData(pp_def.title, Qt.DisplayRole)
            item.setData(pp_def, DescriptionRole)
            self.preprocessormodel.insertRow(row, [item])
            return True
        else:
            return False

    def save(self, model):
        # Save the preprocessor list to a dict.
        d = {}
        preprocessors = []
        for i in range(model.rowCount()):
            item = model.item(i)
            if item is not None:
                pp_def = item.data(DescriptionRole)
                params = item.data(ParametersRole)
                preprocessors.append((pp_def.qualname, params))

        d["preprocessors"] = preprocessors
        return d

    def load(self, saved):
        # Load the list of chosen preprocessors from a dict.
        preprocessors = saved.get("preprocessors", [])

        # Init an empty model.
        model = QStandardItemModel()
        model.dropMimeData = self.drop_mime_data

        # If the pp list is empty, insert a default tokenizer.
        if not preprocessors:
            q_name = "orangecontrib.text.preprocess.tokenizer"
            pp_def = self._pp_def_mapping[q_name]

            item = QStandardItem(pp_def.title)
            item.setData({}, ParametersRole)
            item.setData(pp_def.title, Qt.DisplayRole)
            item.setData(pp_def, DescriptionRole)
            model.appendRow(item)

        # Insert the rest.
        for q_name, params in preprocessors:
            # Get the pp definition from the qual name.
            pp_def = self._pp_def_mapping[q_name]
            if pp_def.icon:
                icon = QIcon(pp_def.icon)
            else:
                icon = QIcon()
            item = QStandardItem(icon, pp_def.title)
            item.setToolTip(pp_def.summary)
            item.setData(pp_def, DescriptionRole)
            item.setData(params, ParametersRole)

            model.appendRow(item)
        return model

    def __activated(self, index):
        # On pp option double click.
        # Get the pp item from the list.
        list_item = self.preprocessors.itemFromIndex(index)
        # Extract the pp action class instance.
        pp_action = list_item.data(DescriptionRole)

        # Reject the double click add if duplicate.
        for i in range(0, self.preprocessormodel.rowCount()):
            if pp_action.qualname == self.preprocessormodel.index(i, 0).data(DescriptionRole).qualname:
                return

        item = QStandardItem()
        item.setData({}, ParametersRole)
        item.setData(pp_action.title, Qt.DisplayRole)
        item.setData(pp_action, DescriptionRole)
        self.preprocessormodel.appendRow([item])

    def storeSpecificSettings(self):
        self.stored_settings = self.save(self.preprocessormodel)
        super().storeSpecificSettings()

    def commit(self, *args, **kwargs):
        # Sync the model into stored_settings on every change commit.
        self.storeSpecificSettings()
        if not self._invalidated:
            self._invalidated = True
            QApplication.postEvent(self, QEvent(QEvent.User))

    def handleNewSignals(self):
        self.apply()

    def customEvent(self, event):
        if event.type() == QEvent.User and self._invalidated:
            self._invalidated = False
            self.apply()

    def apply(self):
        self.error(1, "")
        # Nothing to do, if there is no input corpus.
        if self.data is None:
            return

        # Build a preprocessor instance with the selected parameters.
        preprocessor = self.build_preproc()
        if not preprocessor:
            self.error(1, "Failed to create a Preprocessor object.")
            return

        # Pre-process the documents and present feedback via the progress bar.
        tokens = []
        self.progressBarInit()
        for i, document in enumerate(self.data.documents):
            tokens.append(preprocessor._preprocess_document(document))
            print(tokens[-1])
            self.progressBarSet(100.0 * (i/len(self.data.documents)))
        print("=================================================================")
        self.progressBarFinished()

        # Update the input corpus with a list of tokens.
        self.data.tokens = tokens
        self.send(Output.PP_CORPUS, self.data)

        # Sync the model into stored_settings on every apply.
        self.storeSpecificSettings()

    def build_preproc(self):
        self.warning(1, "")
        # Preprocessor parameters that we require.
        p = {"use_twitter_tokenizer": False, "lowercase": False, "stop_words": None, "transformation": None}
        # For each row in the self.preprocessormodel, acquire
        # the info for the preprocessor.
        for i in range(self.preprocessormodel.rowCount()):
            pp_item = self.preprocessormodel.item(i)
            desc = pp_item.data(DescriptionRole)
            params = pp_item.data(ParametersRole)

            if not isinstance(params, dict):
                params = {}

            # Get the method from the BaseEditor class.
            pp_info = desc.view_class.create_instance(params)

            # If stop words, check for a possible exception.
            if isinstance(pp_info.get("stop_words", None), Exception):
                self.warning(1, "Stop word removal failed: {}".format(pp_info["stop_words"]))
                pp_info["stop_words"] = None

            p.update(pp_info)

        return Preprocessor(**p)

    @Slot()
    def __update_size_constraint(self):
        # Update minimum width constraint on the scroll area containing
        # the 'instantiated' preprocessor list (to avoid the horizontal
        # scroll bar).
        sh = self.flow_view.minimumSizeHint()
        scroll_width = self.scroll_area.verticalScrollBar().width()
        self.scroll_area.setMinimumWidth(min(max(sh.width() + scroll_width + 2,
                                                 self.controlArea.width()), 520))
