from PyQt4.QtGui import QTextEdit, QComboBox, QWidget
from PyQt4 import QtGui, QtCore
from datetime import date


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
