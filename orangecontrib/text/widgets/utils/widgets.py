import os

from AnyQt.QtWidgets import (QComboBox, QWidget, QHBoxLayout,
                         QSizePolicy, QLineEdit, QDoubleSpinBox,
                         QSpinBox, QTextEdit, QDateEdit, QGroupBox,
                             QPushButton, QStyle, QFileDialog, QLabel,
                             QGridLayout, QCheckBox, QStackedLayout)
from AnyQt.QtGui import QColor
from AnyQt.QtCore import QDate, pyqtSignal, Qt, QSize
from Orange.data import DiscreteVariable, ContinuousVariable, TimeVariable, \
    StringVariable

from Orange.widgets.gui import OWComponent, hBox
from Orange.widgets import settings

from orangecontrib.text.corpus import get_sample_corpora_dir


class ListEdit(QTextEdit):
    PLACEHOLDER_COLOR = QColor(128, 128, 128)
    USER_TEXT_COLOR = QColor(0, 0, 0)

    def __init__(self, master=None, attr=None, placeholder_text=None,
                 fixed_height=None, *args):
        super().__init__(*args)
        self.master = master
        self.attr = attr
        self.placeholder_text = placeholder_text

        if self.master and self.attr:
            self.setText('\n'.join(getattr(self.master, self.attr, [])))

        self.set_placeholder()
        self.textChanged.connect(self.synchronize)

        if fixed_height:
            self.setFixedHeight(fixed_height)

    def set_placeholder(self):
        """ Set placeholder if there is no user input. """
        if self.toPlainText() == '':
            self.setFontItalic(True)
            self.setTextColor(self.PLACEHOLDER_COLOR)
            self.setText(self.placeholder_text)

    def toPlainText(self):
        """ Return only text input from user. """
        text = super().toPlainText()
        if self.placeholder_text is not None and text == self.placeholder_text:
            text = ''
        return text

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self.toPlainText() == '':
            self.clear()
            self.setFontItalic(False)
            self.setTextColor(self.USER_TEXT_COLOR)

    def focusOutEvent(self, event):
        self.set_placeholder()
        QTextEdit.focusOutEvent(self, event)

    def synchronize(self):
        if self.master and self.attr:
            setattr(self.master, self.attr, self.value())

    def value(self):
        return self.text.split('\n') if self.text else []

    @property
    def text(self):
        return self.toPlainText().strip()


class QueryBox(QComboBox):
    def __init__(self, widget, master, history, callback, min_width=150):
        super().__init__()
        self.master = master
        self.history = history
        self.callback = callback

        self.setMinimumWidth(min_width)
        self.setEditable(True)
        self.activated[int].connect(self.synchronize)   # triggered for enter and drop-down
        widget.layout().addWidget(self)
        self.refresh()

    def synchronize(self, n=None, silent=False):
        if n is not None and n < len(self.history):     # selecting from drop-down
            name = self.history[n]
            del self.history[n]
            self.history.insert(0, name)
        else:                                           # enter pressed
            query = self.currentText()
            if query != '':
                if query in self.history:
                    self.history.remove(query)
                self.history.insert(0, self.currentText())

        self.refresh()

        if callable(self.callback) and not silent:
            self.callback()

    def refresh(self):
        self.clear()
        for query in self.history:
            self.addItem(query)


class CheckListLayout(QGroupBox):
    def __init__(self, title, master, attr, items, cols=1, callback=None):
        super().__init__(title=title)
        self.master = master
        self.attr = attr
        self.items = items
        self.callback = callback

        self.current_values = getattr(self.master, self.attr)

        layout = QGridLayout()
        self.setLayout(layout)

        nrows = len(items) // cols + bool(len(items) % cols)

        self.boxes = []
        for i, value in enumerate(self.items):
            box = QCheckBox(value)
            box.setChecked(value in self.current_values)
            box.stateChanged.connect(self.synchronize)
            self.boxes.append(box)
            layout.addWidget(box, i % nrows, i // nrows)

    def synchronize(self):
        values = []
        for item, check_box in zip(self.items, self.boxes):
            if check_box.isChecked():
                values.append(item)

        setattr(self.master, self.attr, values)

        if self.callback:
            self.callback()


class ComboBox(QComboBox):
    def __init__(self, master, attr, items):
        super().__init__()
        self.attr = attr
        self.master = master

        if not isinstance(items[0], tuple):
            self.items = [(str(item), item) for item in items]
        else:
            self.items = items

        for i, (key, value) in enumerate(self.items):
            self.addItem(key)
            if value == getattr(master, attr, None):
                self.setCurrentIndex(i)

        self.currentIndexChanged[int].connect(self.synchronize)

    def synchronize(self, i):
        setattr(self.master, self.attr, self.items[i][1])


class DatePicker(QDateEdit):
    QT_DATE_FORMAT = 'yyyy-MM-dd'
    PY_DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, widget, master, attribute, label, margin=(0, 0, 0, 0),
                 display_format=QT_DATE_FORMAT, min_date=None, max_date=None, calendar_popup=True):
        super().__init__()
        self.master = master
        self.attribute = attribute

        hb = hBox(widget)
        hb.layout().setContentsMargins(*margin)
        hb.layout().addWidget(QLabel(label))
        hb.layout().addWidget(self)

        self.setCalendarPopup(calendar_popup)
        self.setDisplayFormat(display_format)
        self.setDate(self.to_qdate(getattr(master, attribute)))
        if min_date:
            self.setMinimumDate(self.to_qdate(min_date))
        if max_date:
            self.setMaximumDate(self.to_qdate(max_date))
        self.dateChanged.connect(self.synchronize)

    @classmethod
    def to_qdate(cls, date):
        return QDate.fromString(date.strftime(cls.PY_DATE_FORMAT),
                                       cls.QT_DATE_FORMAT)

    def synchronize(self):
        setattr(self.master, self.attribute, self.date().toPyDate())


class DatePickerInterval(QWidget):
    def __init__(self, widget, master, attribute_from, attribute_to, min_date=None, max_date=None,
                 label_from='From:', label_to='To:', margin=(0, 0, 0, 0)):
        super().__init__()
        self.setParent(widget)

        hb = hBox(widget)
        self.picker_from = DatePicker(hb, master, attribute_from, label_from,
                                      min_date=min_date, max_date=max_date, margin=margin)
        self.picker_to = DatePicker(hb, master, attribute_to, label_to,
                                    min_date=min_date, max_date=max_date, margin=margin)
        self.picker_from.dateChanged.connect(self.synchronize)
        self.picker_to.dateChanged.connect(self.synchronize)
        self.synchronize()

    def synchronize(self):
        self.picker_from.setMaximumDate(self.picker_to.date())
        self.picker_to.setMinimumDate(self.picker_from.date())


class FileWidget(QWidget):
    on_open = pyqtSignal(str)

    # TODO consider removing directory_aliases since it is not used any more
    def __init__(self, dialog_title='', dialog_format='',
                 start_dir=os.path.expanduser('~/'),
                 icon_size=(12, 20), minimal_width=200,
                 browse_label='Browse', on_open=None,
                 reload_button=True, reload_label='Reload',
                 recent_files=None, directory_aliases=None,
                 allow_empty=True, empty_file_label='(none)'):
        """ Creates a widget with a button for file loading and
        an optional combo box for recent files and reload buttons.

        Args:
            dialog_title (str): The title of the dialog.
            dialog_format (str): Formats for the dialog.
            start_dir (str): A directory to start from.
            icon_size (int, int): The size of buttons' icons.
            on_open (callable): A callback function that accepts filepath as the only argument.
            reload_button (bool): Whether to show reload button.
            reload_label (str): The text displayed on the reload button.
            recent_files (List[str]): List of recent files.
            directory_aliases (dict): An {alias: dir} dictionary for fast directories' access.
            allow_empty (bool): Whether empty path is allowed.
        """
        super().__init__()
        self.dialog_title = dialog_title
        self.dialog_format = dialog_format
        self.start_dir = start_dir

        # Recent files should also contain `empty_file_label` so
        # when (none) is selected this is stored in settings.
        self.recent_files = recent_files if recent_files is not None else []
        self.directory_aliases = directory_aliases or {}
        self.allow_empty = allow_empty
        self.empty_file_label = empty_file_label
        if self.empty_file_label not in self.recent_files \
                and (self.allow_empty or not self.recent_files):
            self.recent_files.append(self.empty_file_label)

        self.check_existence()
        self.on_open.connect(on_open)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if recent_files is not None:
            self.file_combo = QComboBox()
            self.file_combo.setMinimumWidth(minimal_width)
            self.file_combo.activated[int].connect(self.select)
            self.update_combo()
            layout.addWidget(self.file_combo)

        self.browse_button = QPushButton(browse_label)
        self.browse_button.setFocusPolicy(Qt.NoFocus)
        self.browse_button.clicked.connect(self.browse)
        self.browse_button.setIcon(self.style()
                                   .standardIcon(QStyle.SP_DirOpenIcon))
        self.browse_button.setIconSize(QSize(*icon_size))
        self.browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.browse_button)

        if reload_button:
            self.reload_button = QPushButton(reload_label)
            self.reload_button.setFocusPolicy(Qt.NoFocus)
            self.reload_button.clicked.connect(self.reload)
            self.reload_button.setIcon(self.style()
                                       .standardIcon(QStyle.SP_BrowserReload))
            self.reload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.reload_button.setIconSize(QSize(*icon_size))
            layout.addWidget(self.reload_button)

    def browse(self, start_dir=None):
        start_dir = start_dir or self.start_dir
        path, _ = QFileDialog().getOpenFileName(self, self.dialog_title,
                                                start_dir, self.dialog_format)

        if path and self.recent_files is not None:
            if path in self.recent_files:
                self.recent_files.remove(path)
            self.recent_files.insert(0, path)
            self.update_combo()

        if path:
            self.open_file(path)

    def select(self, n):
        name = self.file_combo.currentText()
        if name == self.empty_file_label:
            del self.recent_files[n]
            self.recent_files.insert(0, self.empty_file_label)
            self.update_combo()
            self.open_file(self.empty_file_label)
        elif name in self.directory_aliases:
            self.browse(self.directory_aliases[name])
        elif n < len(self.recent_files):
            name = self.recent_files[n]
            del self.recent_files[n]
            self.recent_files.insert(0, name)
            self.update_combo()
            self.open_file(self.recent_files[0])

    def update_combo(self):
        """ Sync combo values to the changes in self.recent_files. """
        if self.recent_files is not None:
            self.file_combo.clear()
            for i, file in enumerate(self.recent_files):
                # remove (none) when we have some files and allow_empty=False
                if file == self.empty_file_label and \
                        not self.allow_empty and len(self.recent_files) > 1:
                    del self.recent_files[i]
                else:
                    self.file_combo.addItem(os.path.split(file)[1])

            for alias in self.directory_aliases.keys():
                self.file_combo.addItem(alias)

    def reload(self):
        if self.recent_files:
            self.select(0)

    def check_existence(self):
        if self.recent_files:
            to_remove = []
            for file in self.recent_files:
                doc_path = os.path.join(get_sample_corpora_dir(), file)
                exists = any(os.path.exists(f) for f in [file, doc_path])
                if file != self.empty_file_label and not exists:
                    to_remove.append(file)
            for file in to_remove:
                self.recent_files.remove(file)

    def open_file(self, path):
        self.on_open.emit(path if path != self.empty_file_label else '')

    def get_selected_filename(self):
        if self.recent_files:
            return self.recent_files[0]
        else:
            return self.empty_file_label


class ValidatedLineEdit(QLineEdit):
    invalid_input_signal = pyqtSignal(str)

    def __init__(self, master, attr, validator, *args):
        super().__init__(*args)
        self.master = master
        self.attr = attr
        self.validator = validator

        self.setText(getattr(master, attr))
        self.on_change()
        self.textChanged.connect(self.on_change)

    def on_change(self):
        if self.validator(self.text()):
            self.setStyleSheet("QLineEdit { border : 1px solid gray;}")
            self.synchronize()
        else:
            self.setStyleSheet("QLineEdit { border : 2px solid red;}")
            self.invalid_input_signal.emit("Invalid '{}' value.".format(self.attr))

    def synchronize(self):
        setattr(self.master, self.attr, self.text())


class AbsoluteRelativeSpinBox(QWidget):
    editingFinished = pyqtSignal()
    valueChanged = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        layout = QStackedLayout(self)

        self.double_spin = QDoubleSpinBox()
        self.double_spin.valueChanged.connect(self.double_value_changed)
        self.double_spin.editingFinished.connect(self.double_editing_finished)
        layout.addWidget(self.double_spin)

        self.int_spin = QSpinBox()
        self.int_spin.setMaximum(10 ** 4)
        self.int_spin.valueChanged.connect(self.int_value_changed)
        self.int_spin.editingFinished.connect(self.int_editing_finished)
        layout.addWidget(self.int_spin)

        self.setValue(kwargs.get('value', 0.))

    def double_value_changed(self):
        if self.double_spin.value() > 1:
            self.layout().setCurrentIndex(1)
            self.int_spin.setValue(self.double_spin.value())

        self.valueChanged.emit()

    def double_editing_finished(self):
        if self.double_spin.value() <= 1.:
            self.editingFinished.emit()

    def int_value_changed(self):
        if self.int_spin.value() == 0:
            self.layout().setCurrentIndex(0)
            self.double_spin.setValue(1. - self.double_spin.singleStep())
            # There is no need to emit valueChanged signal.

    def int_editing_finished(self):
        if self.int_spin.value() > 0:
            self.editingFinished.emit()

    def value(self):
        return self.int_spin.value() or self.double_spin.value()

    def setValue(self, value):
        if isinstance(value, int):
            self.layout().setCurrentIndex(1)
            self.int_spin.setValue(value)
        else:
            self.layout().setCurrentIndex(0)
            self.double_spin.setValue(value)

    def setSingleStep(self, step):
        if isinstance(step, float):
            self.double_spin.setSingleStep(step)
        else:
            self.int_spin.setSingleStep(step)


class RangeWidget(QWidget):
    valueChanged = pyqtSignal()
    editingFinished = pyqtSignal()

    def __init__(self, widget, master, attribute, minimum=0., maximum=1., step=.05,
                 min_label=None, max_label=None, allow_absolute=False, dtype=float,
                 callback=None, *args):
        super().__init__(*args)
        if widget:
            widget.layout().addWidget(self)
        self.allow_absolute_values = allow_absolute
        self.master = master
        self.attribute = attribute
        self.min = minimum
        self.max = maximum
        self.step = step

        self.min_label = min_label
        self.max_label = max_label
        a, b = self.master_value()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if self.allow_absolute_values:
            SpinBox = AbsoluteRelativeSpinBox
        else:
            if dtype == float:
                SpinBox = QDoubleSpinBox
            else:
                SpinBox = QSpinBox

        if self.min_label:
            layout.addWidget(QLabel(self.min_label))

        self.min_spin = SpinBox(value=a)
        self.min_spin.setSingleStep(self.step)
        layout.addWidget(self.min_spin)

        if self.max_label:
            layout.addWidget(QLabel(self.max_label))

        self.max_spin = SpinBox(value=b)
        self.max_spin.setSingleStep(self.step)
        layout.addWidget(self.max_spin)

        self.set_range()
        self.min_spin.editingFinished.connect(self._editing_finished)
        self.max_spin.editingFinished.connect(self._editing_finished)
        if callback:
            self.valueChanged.connect(callback)

    def synchronize(self):
        a, b = self.value()
        if isinstance(self.attribute, str):
            setattr(self.master, self.attribute, (a, b))
        else:
            setattr(self.master, self.attribute[0], a)
            setattr(self.master, self.attribute[1], b)
        self.set_range()

    def _editing_finished(self):
        value_before = self.master_value()
        self.synchronize()
        if value_before != self.master_value():
            self.editingFinished.emit()

    def master_value(self):
        if isinstance(self.attribute, str):
            return getattr(self.master, self.attribute)
        return (getattr(self.master, self.attribute[0]),
                getattr(self.master, self.attribute[1]))

    def value(self):
        return self.min_spin.value(), self.max_spin.value()

    def set_range(self):
        if not self.allow_absolute_values:
            a, b = self.value()
            self.min_spin.setRange(self.min, b)
            self.max_spin.setRange(a, self.max)


class ResourceLoader(QWidget, OWComponent):
    valueChanged = pyqtSignal(str, str)

    recent_files = settings.Setting([])
    recent_provider = settings.Setting([])
    resource_path = settings.Setting('')

    def __init__(self, widget, model_format, provider_format,
                 model_button_label='Model', provider_button_label='Provider'):
        QWidget.__init__(self)
        OWComponent.__init__(self, widget)

        self.model_path = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.model_widget = FileWidget(recent_files=self.recent_files, dialog_title='Load model',
                                       dialog_format=model_format, start_dir=None,
                                       on_open=self.load_model, allow_empty=False,
                                       reload_button=False, browse_label=model_button_label)
        self.model_path = self.recent_files[0] if self.recent_files else None

        layout.addWidget(self.model_widget)

        self.provider_widget = FileWidget(recent_files=self.recent_provider, dialog_title='Load provider',
                                          dialog_format=provider_format, start_dir=None,
                                          on_open=self.load_provider, allow_empty=False,
                                          reload_button=False, browse_label=provider_button_label)
        layout.addWidget(self.provider_widget)

    def load_model(self, path_to_file):
        self.model_path = path_to_file
        self.valueChanged.emit(self.model_path, self.resource_path)

    def load_provider(self, path_to_file):
        self.resource_path = path_to_file
        self.valueChanged.emit(self.model_path, self.resource_path)
