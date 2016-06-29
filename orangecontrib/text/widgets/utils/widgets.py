import os

from PyQt4.QtGui import (QComboBox, QWidget, QHBoxLayout, QPushButton, QStyle,
                         QSizePolicy, QFileDialog, QLineEdit, QDoubleSpinBox,
                         QSpinBox, QTextEdit)
from PyQt4 import QtCore, QtGui


class ListEdit(QTextEdit):

    def __init__(self, master=None, attr=None, *args):
        super().__init__(*args)
        self.master = master
        self.attr = attr

        if self.master and self.attr:
            self.setText('\n'.join(getattr(self.master, self.attr, [])))

        self.textChanged.connect(self.synchronize)
        self.setToolTip('Separate queries with a newline.')

    def synchronize(self):
        if self.master and self.attr:
            setattr(self.master, self.attr, self.toPlainText().split('\n'))


class CheckListLayout(QtGui.QGroupBox):
    def __init__(self, title, master, attr, items, cols=1):
        super().__init__(title=title)
        self.master = master
        self.attr = attr
        self.items = items

        self.current_values = getattr(self.master, self.attr)

        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        nrows = len(items) // cols + bool(len(items) % cols)

        self.boxes = []
        for i, (text, value) in enumerate(self.items):
            box = QtGui.QCheckBox(text)
            box.setChecked(value in self.current_values)
            box.stateChanged.connect(self.synchronize)
            self.boxes.append(box)
            layout.addWidget(box, i % nrows, i // nrows)

    def synchronize(self):
        values = []
        for item, check_box in zip(self.items, self.boxes):
            if check_box.isChecked():
                values.append(item[1])

        setattr(self.master, self.attr, values)


class ComboBox(QComboBox):
    def __init__(self, master, attr, items):
        super().__init__()
        self.attr = attr
        self.master = master

        if not isinstance(items[0], tuple):
            self.items = [(str(item).capitalize(), item) for item in items]
        else:
            self.items = items

        for i, (key, value) in enumerate(self.items):
            self.addItem(key)
            if value == getattr(master, attr, None):
                self.setCurrentIndex(i)

        self.currentIndexChanged[int].connect(self.synchronize)

    def synchronize(self, i):
        setattr(self.master, self.attr, self.items[i][1])


class DateInterval(QWidget):
    QT_DATE_FORMAT = 'yyyy-MM-dd'
    PY_DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, master, attribute, display_format=QT_DATE_FORMAT,
                 min_date=None, max_date=None, from_label='from:', to_label='to:'):
        super().__init__()
        self.attribute = attribute
        self.master = master
        self.min_date = min_date
        self.max_date = max_date

        layout = QtGui.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QtGui.QLabel(from_label))
        self.from_widget = QtGui.QDateEdit(displayFormat=display_format)
        layout.addWidget(self.from_widget)

        layout.addWidget(QtGui.QLabel(to_label))
        self.to_widget = QtGui.QDateEdit(displayFormat=display_format)
        layout.addWidget(self.to_widget)

        if min_date:
            self.from_widget.setMinimumDate(self.to_qdate(min_date))
            self.to_widget.setMinimumDate(self.to_qdate(min_date))
        if max_date:
            self.from_widget.setMaximumDate(self.to_qdate(max_date))
            self.to_widget.setMaximumDate(self.to_qdate(max_date))

        self.from_widget.dateChanged.connect(self.synchronize)
        self.to_widget.dateChanged.connect(self.synchronize)

        self.value = getattr(master, attribute)

    @classmethod
    def to_qdate(cls, date):
        return QtCore.QDate.fromString(date.strftime(cls.PY_DATE_FORMAT),
                                       cls.QT_DATE_FORMAT)

    @property
    def value(self):
        return self.from_widget.date().toPyDate(), self.to_widget.date().toPyDate()

    @value.setter
    def value(self, value):
        if value:
            self.from_widget.setDate(self.to_qdate(value[0]))
            self.to_widget.setDate(self.to_qdate(value[1]))

    def synchronize(self):
        setattr(self.master, self.attribute, self.value)
        self.from_widget.setMaximumDate(self.to_widget.date())
        self.to_widget.setMinimumDate(self.from_widget.date())


class FileWidget(QWidget):
    default_size = (12, 20)
    empty_file = '(none)'
    start_dir = os.path.expanduser('~/')
    loading_error_signal = QtCore.pyqtSignal(str)
    pathChanged = QtCore.pyqtSignal(str)

    def __init__(self, recent=None, icon_size=None,
                 dialog_title='', dialog_format='',
                 callback=None, directory_aliases=None,
                 allow_empty=True, show_labels=False):
        """ Creates a widget with combo box for recent files and Browse and Reload buttons.

        Args:
            recent (List[str]): List of recent files.
            icon_size (int, int): The size of buttons' icons.
            dialog_title (str): The title of an open file dialog.
            dialog_format (str): Formats for file dialog.
            callback (callable): A function that accepts filepath as the only argument.
            directory_aliases (dict): An {alias: dir} dictionary for fast directories' access.
            allow_empty (bool): Whether empty path is allowed.
            show_labels (bool): Whether to show text along with the buttons' icons.
        """
        super().__init__()
        self.dialog_title = dialog_title
        self.dialog_format = dialog_format
        self.directory_aliases = directory_aliases or {}
        self.allow_empty = allow_empty
        self.recent_files = recent if recent is not None else []
        self._check_files_exist()
        self.callback = callback

        icon_size = icon_size or self.default_size

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(200)
        self.file_combo.activated[int].connect(self.select)
        self.update_combo()
        layout.addWidget(self.file_combo)

        self.browse_button = QPushButton('Browse' if show_labels else '')
        self.browse_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.browse_button.clicked.connect(self.browse)
        self.browse_button.setIcon(self.style()
                                   .standardIcon(QStyle.SP_DirOpenIcon))
        self.browse_button.setIconSize(QtCore.QSize(*icon_size))
        self.browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout.addWidget(self.browse_button)

        self.reload_button = QPushButton('Reload' if show_labels else '')
        self.reload_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.reload_button.clicked.connect(self.reload)
        self.reload_button.setIcon(self.style()
                                   .standardIcon(QStyle.SP_BrowserReload))
        self.reload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.reload_button.setIconSize(QtCore.QSize(*icon_size))
        layout.addWidget(self.reload_button)

    def update_combo(self):
        self.file_combo.clear()
        for file in self.recent_files:
            self.file_combo.addItem(os.path.split(file)[1])

        if self.allow_empty or not self.recent_files:
            self.file_combo.addItem(self.empty_file)

        for alias in self.directory_aliases.keys():
            self.file_combo.addItem(alias)

    def select(self, n):
        name = self.file_combo.currentText()
        if n < len(self.recent_files):
            name = self.recent_files[n]
            del self.recent_files[n]
            self.recent_files.insert(0, name)
        elif name == self.empty_file:
            self.open_file(None)
        elif name in self.directory_aliases:
            self.browse(self.directory_aliases[name])

        if len(self.recent_files) > 0:
            self.update_combo()
            self.open_file(self.recent_files[0])

    def browse(self, start_dir=None):
        start_dir = start_dir if start_dir else self.start_dir
        path = QFileDialog().getOpenFileName(self, self.dialog_title,
                                             start_dir, self.dialog_format)
        if not path:
            return

        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.update_combo()
        self.open_file(path)

    def reload(self):
        if self.recent_files:
            self.select(0)

    def open_file(self, path=None):
        self.pathChanged.emit(path)

        try:
            if self.callback:
                self.callback(path if path != self.empty_file else None)
        except (OSError, IOError):
            self.loading_error_signal.emit('Could not open "{}".'
                                           .format(path))

    def _check_files_exist(self):
        to_remove = [
            file for file in self.recent_files
            if not os.path.exists(file)
        ]
        for file in to_remove:
            self.recent_files.remove(file)


class ValidatedLineEdit(QLineEdit):

    invalid_input_signal = QtCore.pyqtSignal(str)

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

    editingFinished = QtCore.pyqtSignal()
    valueChanged = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        layout = QtGui.QStackedLayout(self)

        self.double_spin = QDoubleSpinBox()
        self.double_spin.valueChanged.connect(self.double_value_changed)
        self.double_spin.editingFinished.connect(self.double_editing_finished)
        layout.addWidget(self.double_spin)

        self.int_spin = QSpinBox()
        self.int_spin.setMaximum(10**4)
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

    valueChanged = QtCore.pyqtSignal()
    editingFinished = QtCore.pyqtSignal()

    def __init__(self, master, attribute, minimum=0., maximum=1., step=.05,
                 min_label=None, max_label=None, allow_absolute=False, *args):
        super().__init__(*args)
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
            SpinBox = QDoubleSpinBox

        self.min_spin = SpinBox(value=a)
        self.min_spin.setSingleStep(self.step)
        layout.addWidget(self.min_spin)

        self.max_spin = SpinBox(value=b)
        self.max_spin.setSingleStep(self.step)
        layout.addWidget(self.max_spin)

        self.min_spin.valueChanged.connect(self.valueChanged)
        self.min_spin.editingFinished.connect(self.synchronize)
        self.max_spin.valueChanged.connect(self.valueChanged)
        self.max_spin.editingFinished.connect(self.synchronize)

    def synchronize(self):
        a, b = self.value()
        if isinstance(self.attribute, str):
            setattr(self.master, self.attribute, (a, b))
        else:
            setattr(self.master, self.attribute[0], a)
            setattr(self.master, self.attribute[1], b)
        self.set_range()
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
            a, b = self.value
            self.min_dspin.setRange(self.min, b)
            self.max_spin.setRange(a, self.max)
