import logging

from typing import Mapping, Any, TypedDict, Iterable

import keyring
from more_itertools import unique_everseen

from AnyQt.QtCore import (
    Qt, QEvent, QModelIndex, QAbstractItemModel, QObject, Signal,
)
from AnyQt.QtGui import QStandardItemModel, QStandardItem
from AnyQt.QtWidgets import QWidget, QFormLayout, QApplication, QLineEdit

from orangecanvas.utils import group_by_all
from orangecontrib.text.widgets.utils.passwordedit import PasswordEdit

from Orange.widgets.utils.combobox import TextEditCombo

ApiKeyRole = Qt.UserRole + 41
#: Flag indicating if a user already entered api key for a provider.
#: Used to avoid premature calls to `keyring.get_password`.
HasApiKeyRole = Qt.UserRole + 42
#: Stores model id
ModelRole = Qt.UserRole + 43

log = logging.getLogger(__name__)


class TextCombo(TextEditCombo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__insertPolicy = self.insertPolicy()
        self.setLineEdit(QLineEdit(self))

    def setLineEdit(self, edit: QLineEdit) -> None:
        if edit == self.lineEdit():
            return
        try:
            old = self.lineEdit()
            old.returnPressed.disconnect(self.__le_rp_before)
            old.returnPressed.disconnect(self.__le_rp_after)
        except TypeError:
            pass
        edit.returnPressed.connect(self.__le_rp_before)
        super().setLineEdit(edit)
        edit.returnPressed.connect(self.__le_rp_after)

    def __le_rp_before(self):
        # Disable insertion before ComboBox can process returnPressed signal
        # from the line edit
        self.__insertPolicy = self.insertPolicy()
        self.setInsertPolicy(TextCombo.NoInsert)

    def __le_rp_after(self):
        # Re-enable insertion
        self.setInsertPolicy(self.__insertPolicy)
        # Move focus to trigger TextEditCombo.__on_editingFinished
        self.focusNextChild()

    def insertItem(self, index: int, text: str, userData = None):
        """Reimplemented.

        QComboBox does not properly insert under rootModelIndex when model
        is a QStandardItemModel.
        This only works if called from `TextEditCombo.__on_editingFinished`
        """
        model = self.model()
        if model is None or not text.strip():
            return
        root = self.rootModelIndex()
        if isinstance(model, QStandardItemModel):
            item = Item({Qt.DisplayRole: text, Qt.UserRole: userData})
            ritem = model.itemFromIndex(root)
            if ritem is None:
                ritem = model.invisibleRootItem()
            count = ritem.rowCount()
            ritem.insertRow(index, item)
            if count == 0:  # Need to update current state if count was 0 before
                self.setCurrentIndex(0)
        else:
            super().insertItem(index, text, userData)

    def addItem(self, text, userData = None):
        self.insertItem(self.count(), text, userData)


class Item(QStandardItem):
    def __init__(self, data: Mapping[int, Any]):
        super().__init__()
        for role, value in data.items():
            self.setData(value,  role)


def move_up_helper(model: QStandardItemModel, parent: QModelIndex, index: int):
    """Move the `index` row in model to first position."""
    if index < 1:
        return
    root = model.itemFromIndex(parent)
    if root is None:
        root = model.invisibleRootItem()
    if 0 <= index < root.rowCount():
        row = root.takeRow(index)
        root.insertRow(0, row)


class LLMModelWidget(QWidget):
    """
    A widget form for entering llm provider endpoint url with api key and
    model selection. The api key is stored using `keyring`

    Parameters:
        keyringServiceName:
            The keyring service name under which the entered api key is stored.
    """
    #: Signal emitted when the data entered by the user changes.
    changed = Signal()
    #: Signal emitted when widget loses focus or Enter/Return is pressed.
    editingFinished = Signal()

    class Item(TypedDict):
        base_url: str
        model: str
        has_key: bool

    def __init__(self, *args, keyringServiceName: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.__edited: bool = False
        self.keyringServiceName = keyringServiceName
        self.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout(
            formAlignment=Qt.AlignLeft,
            labelAlignment=Qt.AlignLeft,
            fieldGrowthPolicy=QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow,
        )
        form.setContentsMargins(0, 0, 0, 0)
        self.base_url_cb = TextCombo(
            insertPolicy=TextCombo.InsertAtTop
        )
        self._model = QStandardItemModel()
        self.base_url_cb.setModel(self._model)

        self.base_url_cb.editingFinished.connect(self._on_base_url_edited)
        self.base_url_cb.editTextChanged.connect(self._mark_edited)
        self.base_url_cb.currentIndexChanged.connect(self._on_base_url_index_change)
        self.api_key_le = PasswordEdit(
            echoMode=PasswordEdit.EchoMode.PasswordEchoOnEdit,
            placeholderText="sk-...",
        )

        self.api_key_le.editingFinished.connect(self._on_api_key_edit)
        self.api_key_le.returnPressed.connect(self.api_key_le.focusNextChild)
        self.api_key_le.textEdited.connect(self._mark_edited)
        self.model_cb = TextCombo(
            placeholderText="model",
            insertPolicy=TextCombo.InsertAtTop
        )
        self.model_cb.setModel(self._model)
        self.model_cb.editingFinished.connect(self._on_model_id_edit)
        self.model_cb.editTextChanged.connect(self._mark_edited)

        form.addRow("API Base Url:", self.base_url_cb)
        form.addRow("Api Key:", self.api_key_le)
        form.addRow("Model:", self.model_cb)
        self.base_url_cb.installEventFilter(self)
        self.api_key_le.installEventFilter(self)
        self.model_cb.installEventFilter(self)
        self.setLayout(form)

    def setHistory(self, history: list[Item]):
        """
        Set the history.
        """
        model = self._model
        model.clear()
        items = unique_everseen(history, key=lambda item: (item["base_url"], item["model"]))
        items_by_url = group_by_all(items, key=lambda item: item["base_url"])
        for i, (base_url, items) in enumerate(items_by_url):
            item = Item({
                Qt.DisplayRole: base_url,
                HasApiKeyRole: items[0].get("has_key"),
            })
            item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            model.appendRow(item)
            for j, itm in enumerate(items):
                item.appendRow(Item({
                    Qt.DisplayRole: itm["model"],
                    ModelRole: itm["model"],
                }))

        if model.rowCount():
            self.model_cb.setRootModelIndex(self._model.index(0, 0))
            self.model_cb.setCurrentIndex(0)

        self._on_base_url_change()

    def history(self) -> list[Item]:
        """Return the history"""
        model = self.base_url_cb.model()
        items = []
        roles = {Qt.DisplayRole: "base_url", ModelRole: "model", HasApiKeyRole: "has_key"}

        def values(model: QAbstractItemModel, midx: QModelIndex, roles: Iterable[int]) -> dict:
            return {role: model.data(midx, role) for role in roles}

        for i in range(model.rowCount()):
            midx = model.index(i, 0)
            endp = values(model, midx, roles.keys())
            for j in range(model.rowCount(midx)):
                vals = {
                    **endp,
                    **{ModelRole: model.index(j, 0, midx).data(Qt.DisplayRole)},
                }
                vals = {roles[r]: vals[r] for r in roles}
                items.append(vals)
        return items

    def baseUrl(self) -> str:
        """Return the base api url."""
        return self.base_url_cb.currentText()

    def setBaseUrl(self, url: str) -> None:
        """Set the base api url."""
        current = self.base_url_cb.currentText()
        if current == url:
            return
        self.base_url_cb.setText(url)
        self._on_base_url_change()

    def _on_base_url_change(self):
        model = self.base_url_cb.model()
        index = self.base_url_cb.currentIndex()
        if index != 0:
            move_up_helper(model, QModelIndex(), index)
            self.base_url_cb.setCurrentIndex(0)

        url = self.base_url_cb.currentText()
        midx = model.index(self.base_url_cb.currentIndex(), 0)
        self.model_cb.setRootModelIndex(midx)
        if self.model_cb.currentIndex() == -1 and self.model_cb.count():
            self.model_cb.setCurrentIndex(0)
        api_key = ""
        if self.base_url_cb.currentData(HasApiKeyRole):
            api_key = self._get_secret(url) or ""
        self.__set_api_key(api_key, store=False)

    def _on_base_url_edited(self):
        self._on_base_url_change()
        self.__emit_changed()

    def _on_base_url_index_change(self, index):
        model = self.base_url_cb.model()
        self.model_cb.setRootModelIndex(model.index(index, 0))
        self.model_cb.setCurrentIndex(0)

    def _mark_edited(self):
        self.__edited = True

    def apiKey(self) -> str:
        """Return the api key."""
        return self.api_key_le.text()

    def setApiKey(self, key: str) -> None:
        """Set the api key."""
        if key == self.apiKey():
            return
        self.__set_api_key(key)

    def __set_api_key(self, key, store=True):
        base_url = self.baseUrl()
        if key and base_url and store:
            self._store_secret(base_url, key)
        self.api_key_le.setText(key)
        index = self.base_url_cb.currentIndex()
        self.base_url_cb.setItemData(index, key, ApiKeyRole)
        self.base_url_cb.setItemData(index, bool(key), HasApiKeyRole)

    def _on_api_key_edit(self):
        # coming from QLineEdit.editingFinished which triggers on focus out
        # after setText(text) even when the text is not modified.
        modified = self.api_key_le.isModified()
        self.__set_api_key(self.apiKey())
        if modified:
            self.__emit_changed()

    def modelId(self) -> str:
        """Return the model id"""
        return self.model_cb.currentText()

    def setModelId(self, modelId: str) -> None:
        """Set model id"""
        if modelId == self.model_cb.currentText():
            return
        self.model_cb.setText(modelId)
        self._on_model_id_change()

    def _on_model_id_change(self):
        # move the model to first index
        index = self.model_cb.currentIndex()
        model = self.model_cb.model()
        rootmidx = self.model_cb.rootModelIndex()
        if index != 0:
            move_up_helper(model, rootmidx, index)
            self.model_cb.setCurrentIndex(0)

    def _on_model_id_edit(self) -> None:
        self._on_model_id_change()
        self.__emit_changed()

    def eventFilter(self, recv: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.FocusOut and event.reason() == Qt.FocusReason.TabFocusReason:
            newfocus = QApplication.focusWidget()
            if self.__edited and not self.isAncestorOf(newfocus):
                self.__emit_editingFinished()
        return super().eventFilter(recv, event)

    def __emit_changed(self):
        self.__edited = True
        self.changed.emit()

    def __emit_editingFinished(self):
        self.__edited = False
        self.editingFinished.emit()

    def _get_secret(self, service):
        try:
            return keyring.get_password(self.keyringServiceName, service)
        except Exception:
            log.exception("Failed to get secret for '%r'.", service)
            return None

    def _store_secret(self, service, password):
        try:
            keyring.set_password(self.keyringServiceName, service, password)
        except Exception:
            log.exception("Failed to set secret for '%s'.", service)