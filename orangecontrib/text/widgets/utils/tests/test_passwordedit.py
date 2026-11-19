from orangewidget.tests.base import GuiTest
from orangecontrib.text.widgets.utils.passwordedit import PasswordEdit


class TestPasswordEdit(GuiTest):
    """Tests for the PasswordEdit class."""

    def setUp(self):
        super().setUp()
        self.edit = PasswordEdit()

    def tearDown(self) -> None:
        del self.edit
        super().tearDown()

    def test_initial_echo_mode(self):
        self.assertEqual(self.edit.echoMode(), PasswordEdit.EchoMode.PasswordEchoOnEdit)

    def test_set_and_get_text(self):
        self.edit.setText("secret-key-123")
        self.assertEqual(self.edit.text(), "secret-key-123")

    def test_toggle_echo_mode(self):
        self.edit.toggleEchoMode()
        self.assertEqual(self.edit.echoMode(), PasswordEdit.EchoMode.Normal)
        self.edit.toggleEchoMode()
        self.assertEqual(self.edit.echoMode(), PasswordEdit.EchoMode.PasswordEchoOnEdit)

    def test_reveal_conceal_text(self):
        """revealText should set Normal echo mode."""
        self.edit.revealText()
        self.assertEqual(self.edit.echoMode(), PasswordEdit.EchoMode.Normal)
        self.edit.concealText()
        self.assertEqual(self.edit.echoMode(), PasswordEdit.EchoMode.PasswordEchoOnEdit)

    def test_toggleEchoMode_normal_to_password(self):
        """toggleEchoMode should switch from Normal to PasswordEchoOnEdit."""
        self.edit.revealText()
        self.edit.toggleEchoMode()
        self.assertEqual(self.edit.echoMode(), PasswordEdit.EchoMode.PasswordEchoOnEdit)
