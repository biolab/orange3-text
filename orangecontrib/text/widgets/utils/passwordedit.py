from AnyQt.QtCore import QTimer
from AnyQt.QtWidgets import QLineEdit, QAction

from orangewidget.utils import load_styled_icon


class PasswordEdit(QLineEdit):
    """Password entry widget with reveal/conceal action."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setEchoMode(PasswordEdit.EchoMode.PasswordEchoOnEdit)
        self.timer = QTimer(singleShot=True, interval=5000)
        self.timer.timeout.connect(self.concealText)
        self._icons = [
            load_styled_icon(__package__, "../icons/show-password.svg"),
            load_styled_icon(__package__, "../icons/hide-password.svg"),
        ]
        self._action = ac = QAction("Show password", self)
        ac.setIcon(self._icons[0])
        ac.triggered.connect(self.toggleEchoMode)
        self.addAction(ac, QLineEdit.TrailingPosition)

    def toggleEchoMode(self):
        """Toggle echo mode"""
        if self.echoMode() == PasswordEdit.EchoMode.Normal:
            self.concealText()
        else:
            self.revealText()

    def revealText(self):
        """Temporarily reveal password."""
        self._action.setIcon(self._icons[1])
        self.setEchoMode(PasswordEdit.EchoMode.Normal)
        self.timer.start()

    def concealText(self):
        """Conceal the password."""
        self._action.setIcon(self._icons[0])
        self.setEchoMode(PasswordEdit.EchoMode.PasswordEchoOnEdit)
        self.timer.stop()
