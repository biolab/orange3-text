import unittest
from unittest.mock import patch

from AnyQt.QtWidgets import QLineEdit, QApplication, QWidget
from AnyQt.QtCore import QEvent, Qt
from AnyQt.QtGui import QStandardItemModel, QFocusEvent
from AnyQt.QtTest import QTest, QSignalSpy

from Orange.widgets.tests.base import GuiTest

from orangecontrib.text.widgets.utils.llmmodelwidget import (
    LLMModelWidget,
    TextCombo,
)


class TestTextCombo(GuiTest):
    """Tests for the TextCombo class."""

    def setUp(self):
        super().setUp()
        self.combo = TextCombo()
        self.model = QStandardItemModel()
        self.combo.setModel(self.model)

    def tearDown(self) -> None:
        del self.combo
        del self.model
        super().tearDown()

    def test_add_item(self):
        self.combo.addItem("first item", "user-data-1")
        self.assertEqual(self.combo.count(), 1)
        self.assertEqual(self.combo.currentText(), "first item")

    def test_insert_item_child(self):
        self.combo.addItem("root item")
        self.assertEqual(self.combo.count(), 1)
        self.assertEqual(self.combo.currentText(), "root item")
        self.combo.setRootModelIndex(self.model.index(0, 0))
        self.combo.addItem("child item")
        self.assertEqual(self.combo.count(), 1)
        self.assertEqual(self.combo.currentText(), "child item")
        self.combo.insertItem(0, "child item 1")
        self.assertEqual(self.combo.count(), 2)
        self.assertEqual(self.combo.currentText(), "child item")
        self.assertEqual(self.combo.currentIndex(), 1)
        self.assertEqual(self.model.rowCount(), 1)

    def test_insert_item_empty_text(self):
        """Empty text should not insert."""
        initial_count = self.combo.count()
        self.combo.insertItem(0, "", "data")
        self.assertEqual(self.combo.count(), initial_count)

    def test_insert_item_whitespace_only(self):
        """Whitespace-only text should not insert."""
        initial_count = self.combo.count()
        self.combo.insertItem(0, "   ", "data")
        self.assertEqual(self.combo.count(), initial_count)


def enter_text(widget: QLineEdit, text: str, enter: bool=True):
    widget.selectAll()
    QTest.keyClick(widget, Qt.Key.Key_Delete)
    QTest.keyClicks(widget, text)
    if enter:
        QTest.keyClick(widget, Qt.Key.Key_Return)


def send_focus_out(widget: QWidget, reason =Qt.TabFocusReason):
    event = QFocusEvent(QEvent.FocusOut, reason)
    QApplication.sendEvent(widget, event)


class TestLLMModelWidget(GuiTest):
    """Tests for the LLMModelWidget widget."""

    def setUp(self):
        super().setUp()
        self.widget = LLMModelWidget(keyringServiceName="test-service")

    def tearDown(self) -> None:
        del self.widget
        super().tearDown()

    def test_initial_state(self):
        """Test widget initial state."""
        self.assertEqual(self.widget.baseUrl(), "")
        self.assertEqual(self.widget.apiKey(), "")
        self.assertEqual(self.widget.modelId(), "")

    def test_edit(self):
        """Test user simulated editing."""
        spy = QSignalSpy(self.widget.editingFinished)
        le = self.widget.base_url_cb.lineEdit()
        enter_text(le, "http://localhost", enter=False)
        self.assertEqual(self.widget.baseUrl(), "http://localhost")
        self.assertEqual(len(spy), 0)
        enter_text(self.widget.model_cb.lineEdit(), "model", enter=False)
        self.assertEqual(self.widget.modelId(), "model")
        self.assertEqual(len(spy), 0)
        # move focus out of the widget
        QApplication.sendEvent(
            self.widget.model_cb, QFocusEvent(QEvent.FocusOut, Qt.TabFocusReason)
        )
        self.assertEqual(len(spy), 1)

    @patch("keyring.get_password", return_value=None)
    def test_set_base_url(self, mock_get):
        """Test setting base URL after history is set."""
        items = [
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="llmmodel",
                has_key=False,
            )
        ]
        self.widget.setHistory(items)
        spy = QSignalSpy(self.widget.changed)
        self.widget.setBaseUrl("https://api.example.com")
        self.assertEqual(self.widget.baseUrl(), "https://api.example.com")
        self.assertEqual(len(spy), 0)

    @patch("keyring.set_password")
    @patch("keyring.get_password", return_value=None)
    def test_api_key_persisted_to_keyring(self, mock_get, mock_set):
        """Test that API key is stored in keyring."""
        items = [
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="llmmodel",
                has_key=False,
            )
        ]
        self.widget.setHistory(items)
        self.widget.setBaseUrl("https://api.example.com")
        spy = QSignalSpy(self.widget.changed)
        self.widget.setApiKey("sk-secret-key")
        mock_set.assert_called_once_with("test-service", "https://api.example.com", "sk-secret-key")
        self.assertEqual(self.widget.apiKey(), "sk-secret-key")
        mock_set.reset_mock()
        enter_text(self.widget.api_key_le, "sk-key")
        mock_set.assert_called_once_with("test-service", "https://api.example.com", "sk-key")
        self.assertEqual(len(spy), 1)

    @patch("keyring.get_password", return_value=None)
    def test_set_model_id(self, mock_get):
        """Test setting model ID after history is set."""
        items = [
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="llmmodel",
                has_key=False,
            )
        ]
        self.widget.setHistory(items)
        self.widget.setModelId("llmmodel")
        self.assertEqual(self.widget.modelId(), "llmmodel")
        # get_password must not be called when has_key is False
        mock_get.assert_not_called()

    @patch("keyring.get_password", return_value="sk-stored-key")
    def test_set_history_with_api_key(self, mock_get):
        """Test setting history where an API key is stored in keyring."""
        items = [
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="llmmodel",
                has_key=True,
            )
        ]
        self.widget.setHistory(items)
        self.assertEqual(self.widget.apiKey(), "sk-stored-key")
        mock_get.assert_called_once_with("test-service", "https://api.example.com")

    @patch("keyring.get_password", return_value=None)
    def test_set_history_multiple_urls(self, mock_get):
        """Test setting history with multiple different base URLs."""
        items = [
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="llmmodel",
                has_key=False,
            ),
            LLMModelWidget.Item(
                base_url="https://api.foo.com",
                model="llmmodel-2",
                has_key=False,
            ),
        ]
        self.widget.setHistory(items)
        history = self.widget.history()
        self.assertEqual(len(history), 2)

        urls = [h["base_url"] for h in history]
        self.assertEqual(urls[0], "https://api.example.com")
        self.assertEqual(urls[1], "https://api.foo.com")

    @patch("keyring.get_password")
    def test_changed_signal_on_api_key_edit(self, mock_get):
        """Test that changed signal is emitted when API key is edited via line edit."""
        mock_get.return_value = None
        items = [
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="llmmodel",
                has_key=False,
            )
        ]
        self.widget.setHistory(items)
        self.widget.setBaseUrl("https://api.example.com")
        spy = QSignalSpy(self.widget.changed)
        # Simulate editing the API key line edit
        enter_text(self.widget.api_key_le, "sk-new-key")
        self.assertEqual(len(spy), 1)

    @patch("keyring.get_password")
    def test_changed_signal_on_model_id_change(self, mock_get):
        """Test that changed signal is emitted when model ID changes via line edit."""
        mock_get.return_value = None
        items = [
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="llmmodel",
                has_key=False,
            ),
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="gpt-3.5-turbo",
                has_key=False,
            ),
        ]
        self.widget.setHistory(items)

        spy = QSignalSpy(self.widget.changed)
        # Simulate editing the model combobox directly
        self.widget.model_cb.setCurrentIndex(0)

        # QTest.keyClick(..., Qt.KeyEnter ) in enter_text doubles returnPressed
        # emit, this does not happen in normal event dispatch. Manually
        # send focus out event to compensate.
        enter_text(self.widget.model_cb.lineEdit(), "llmmodel-2", enter=False)
        send_focus_out(self.widget.model_cb)
        self.assertEqual(self.widget.modelId(), "llmmodel-2", )
        self.assertEqual(len(spy), 1)

    @patch("keyring.set_password")
    @patch("keyring.get_password", return_value=None)
    def test_api_key_not_stored_without_base_url(self, mock_get, mock_set):
        """Test that API key is not stored if there's no base URL."""
        self.widget.setApiKey("sk-secret-key")
        mock_set.assert_not_called()

    @patch("keyring.get_password", return_value=None)
    def test_combobox_populated(self, mock_get):
        """Test that base URL and model combobox are populated after setHistory."""
        items = [
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="llmmodel",
                has_key=False,
            ),
            LLMModelWidget.Item(
                base_url="https://api.example.com",
                model="llmmodel-1",
                has_key=False,
            ),
            LLMModelWidget.Item(
                base_url="https://api.foo.com",
                model="llmmodel-3",
                has_key=False,
            ),
            LLMModelWidget.Item(
                base_url="https://api.foo.com",
                model="llmmodel-4",
                has_key=False,
            ),
        ]
        self.widget.setHistory(items)
        # The base URL combobox should have 2 entries
        self.assertEqual(self.widget.base_url_cb.count(), 2)
        self.assertEqual(self.widget.baseUrl(), "https://api.example.com")
        # Model combobox must have 2 entries
        self.assertEqual(self.widget.model_cb.count(), 2)
        self.assertEqual(self.widget.modelId(), "llmmodel")
        self.assertEqual(self.widget.model_cb.itemText(1), "llmmodel-1")


if __name__ == "__main__":
    unittest.main()