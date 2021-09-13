"""
Import Documents
-------------

Import documents 'into' canvas session from a local file system
"""
import sys
import os
import enum
import warnings
import logging
import traceback
from urllib.parse import urlparse

from types import SimpleNamespace as namespace
from concurrent.futures._base import TimeoutError
from typing import List, Optional

from AnyQt.QtCore import Qt, QEvent, QFileInfo, QThread
from AnyQt.QtCore import pyqtSlot as Slot
from AnyQt.QtGui import QStandardItem, QDropEvent

from AnyQt.QtWidgets import (
    QAction, QPushButton, QComboBox, QApplication, QStyle, QFileDialog,
    QFileIconProvider, QStackedWidget, QProgressBar, QWidget, QHBoxLayout,
    QVBoxLayout, QLabel, QGridLayout, QSizePolicy, QCompleter
)
from numpy import array

from orangewidget.utils.itemmodels import PyListModel

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import widget, gui, settings
from Orange.widgets.data.owfile import LineEditSelectOnFocus
from Orange.widgets.utils.filedialogs import RecentPath
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)
from Orange.widgets.widget import Output

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.import_documents import ImportDocuments, \
    NoDocumentsException

try:
    from orangecanvas.preview.previewbrowser import TextLabel
except ImportError:
    from Orange.canvas.preview.previewbrowser import TextLabel

# domain for skipped images output
SKIPPED_DOMAIN = Domain([], metas=[
    StringVariable("name"),
    StringVariable("path")
])


def prettifypath(path):
    home = os.path.expanduser("~/")
    if path.startswith(home):  # case sensitivity!
        path = os.path.join("~", os.path.relpath(path, home))
    return path


log = logging.getLogger(__name__)


class RuntimeEvent(QEvent):
    Init = QEvent.registerEventType()


def RecentPath_asqstandarditem(pathitem):
    icon_provider = QFileIconProvider()
    # basename of a normalized name (strip right path component separators)
    basename = os.path.basename(os.path.normpath(pathitem.abspath))
    item = QStandardItem(
        icon_provider.icon(QFileInfo(pathitem.abspath)),
        basename
    )
    item.setToolTip(pathitem.abspath)
    item.setData(pathitem, Qt.UserRole)
    return item


class State(enum.IntEnum):
    NoState, Processing, Done, Cancelled, Error = range(5)


class OWImportDocuments(widget.OWWidget):
    name = "Import Documents"
    description = "Import text documents from folders."
    icon = "icons/ImportDocuments.svg"
    priority = 110

    class Outputs:
        data = Output("Corpus", Corpus, default=True)
        skipped_documents = Output("Skipped documents", Table)

    LOCAL_FILE, URL = range(2)
    source = settings.Setting(LOCAL_FILE)
    #: list of recent paths
    recent_paths: List[RecentPath] = settings.Setting([])
    currentPath: Optional[str] = settings.Setting(None)
    recent_urls: List[str] = settings.Setting([])
    lemma_cb = settings.Setting(True)
    pos_cb = settings.Setting(False)
    ner_cb = settings.Setting(False)

    want_main_area = False
    resizing_enabled = False

    Modality = Qt.ApplicationModal
    MaxRecentItems = 20

    class Warning(widget.OWWidget.Warning):
        read_error = widget.Msg("{} couldn't be read.")

    def __init__(self):
        super().__init__()
        #: widget's runtime state
        self.__state = State.NoState
        self.base_corpus = None
        self.corpus = None
        self.n_text_categories = 0
        self.n_text_data = 0
        self.skipped_documents = []
        self.is_conllu = False
        self.tokens = None
        self.pos = None
        self.ner = None

        self.__invalidated = False
        self.__pendingTask = None

        layout = QGridLayout()
        layout.setSpacing(4)
        gui.widgetBox(self.controlArea, orientation=layout, box='Source')
        source_box = gui.radioButtons(None, self, "source", box=True,
                                      callback=self.start, addToLayout=False)
        rb_button = gui.appendRadioButton(source_box, "Folder:",
                                          addToLayout=False)
        layout.addWidget(rb_button, 0, 0, Qt.AlignVCenter)

        box = gui.hBox(None, addToLayout=False, margin=0)
        box.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        self.recent_cb = QComboBox(
            sizeAdjustPolicy=QComboBox.AdjustToMinimumContentsLengthWithIcon,
            minimumContentsLength=16,
            acceptDrops=True
        )
        self.recent_cb.installEventFilter(self)
        self.recent_cb.activated[int].connect(self.__onRecentActivated)

        browseaction = QAction(
            "Open/Load Documents", self,
            iconText="\N{HORIZONTAL ELLIPSIS}",
            icon=self.style().standardIcon(QStyle.SP_DirOpenIcon),
            toolTip="Select a folder from which to load the documents"
        )
        browseaction.triggered.connect(self.__runOpenDialog)
        reloadaction = QAction(
            "Reload", self,
            icon=self.style().standardIcon(QStyle.SP_BrowserReload),
            toolTip="Reload current document set"
        )
        reloadaction.triggered.connect(self.reload)
        self.__actions = namespace(
            browse=browseaction,
            reload=reloadaction,
        )

        browsebutton = QPushButton(
            browseaction.iconText(),
            icon=browseaction.icon(),
            toolTip=browseaction.toolTip(),
            clicked=browseaction.trigger,
            default=False,
            autoDefault=False,
        )
        reloadbutton = QPushButton(
            reloadaction.iconText(),
            icon=reloadaction.icon(),
            clicked=reloadaction.trigger,
            default=False,
            autoDefault=False,
        )
        box.layout().addWidget(self.recent_cb)
        layout.addWidget(box, 0, 1)
        layout.addWidget(browsebutton, 0, 2)
        layout.addWidget(reloadbutton, 0, 3)

        rb_button = gui.appendRadioButton(source_box, "URL:", addToLayout=False)
        layout.addWidget(rb_button, 3, 0, Qt.AlignVCenter)

        self.url_combo = url_combo = QComboBox()
        url_model = PyListModel()
        url_model.wrap(self.recent_urls)
        url_combo.setLineEdit(LineEditSelectOnFocus())
        url_combo.setModel(url_model)
        url_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        url_combo.setEditable(True)
        url_combo.setInsertPolicy(url_combo.InsertAtTop)
        url_edit = url_combo.lineEdit()
        l, t, r, b = url_edit.getTextMargins()
        url_edit.setTextMargins(l + 5, t, r, b)
        layout.addWidget(url_combo, 3, 1, 1, 3)
        url_combo.activated.connect(self._url_set)
        # whit completer we set that combo box is case sensitive when
        # matching the history
        completer = QCompleter()
        completer.setCaseSensitivity(Qt.CaseSensitive)
        url_combo.setCompleter(completer)

        self.addActions([browseaction, reloadaction])

        reloadaction.changed.connect(
            lambda: reloadbutton.setEnabled(reloadaction.isEnabled())
        )

        box = gui.hBox(self.controlArea, "Conllu import options")
        gui.checkBox(box, self, "lemma_cb", "Lemma",
                     callback=self.commit)
        gui.checkBox(box, self, "pos_cb", "POS tags",
                     callback=self.commit)
        gui.checkBox(box, self, "ner_cb", "NER",
                     callback=self.commit)
        self.controlArea.layout().addWidget(box)

        box = gui.vBox(self.controlArea, "Info")
        self.infostack = QStackedWidget()

        self.info_area = QLabel(
            text="No document set selected",
            wordWrap=True
        )
        self.progress_widget = QProgressBar(
            minimum=0, maximum=100
        )
        self.cancel_button = QPushButton(
            "Cancel",
            icon=self.style().standardIcon(QStyle.SP_DialogCancelButton),
            default=False,
            autoDefault=False,
        )
        self.cancel_button.clicked.connect(self.cancel)

        w = QWidget()
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)

        hlayout.addWidget(self.progress_widget)
        hlayout.addWidget(self.cancel_button)
        vlayout.addLayout(hlayout)

        self.pathlabel = TextLabel()
        self.pathlabel.setTextElideMode(Qt.ElideMiddle)
        self.pathlabel.setAttribute(Qt.WA_MacSmallSize)

        vlayout.addWidget(self.pathlabel)
        w.setLayout(vlayout)

        self.infostack.addWidget(self.info_area)
        self.infostack.addWidget(w)

        box.layout().addWidget(self.infostack)

        self.__initRecentItemsModel()
        self.__invalidated = True
        self.__executor = ThreadExecutor(self)

        QApplication.postEvent(self, QEvent(RuntimeEvent.Init))

    def _url_set(self):
        url = self.url_combo.currentText()
        pos = self.recent_urls.index(url)
        url = url.strip()
        if not urlparse(url).scheme:
            url = "http://" + url
            self.url_combo.setItemText(pos, url)
            self.recent_urls[pos] = url
        self.source = self.URL
        self.start()

    def __initRecentItemsModel(self):
        if self.currentPath is not None and \
                not os.path.isdir(self.currentPath):
            self.currentPath = None

        recent_paths = []
        for item in self.recent_paths:
            if os.path.isdir(item.abspath):
                recent_paths.append(item)
        recent_paths = recent_paths[:OWImportDocuments.MaxRecentItems]
        recent_model = self.recent_cb.model()
        for pathitem in recent_paths:
            item = RecentPath_asqstandarditem(pathitem)
            recent_model.appendRow(item)

        self.recent_paths = recent_paths

        if self.currentPath is not None and \
                os.path.isdir(self.currentPath) and self.recent_paths and \
                os.path.samefile(self.currentPath,
                                 self.recent_paths[0].abspath):
            self.recent_cb.setCurrentIndex(0)
        else:
            self.currentPath = None
            self.recent_cb.setCurrentIndex(-1)
        self.__actions.reload.setEnabled(self.currentPath is not None)

    def customEvent(self, event):
        """Reimplemented."""
        if event.type() == RuntimeEvent.Init:
            if self.__invalidated:
                try:
                    self.start()
                finally:
                    self.__invalidated = False

        super().customEvent(event)

    def __runOpenDialog(self):
        startdir = os.path.expanduser("~/")
        if self.recent_paths:
            startdir = os.path.dirname(self.recent_paths[0].abspath)

        caption = "Select Top Level Folder"
        if OWImportDocuments.Modality == Qt.WindowModal:
            dlg = QFileDialog(
                self, caption, startdir,
                acceptMode=QFileDialog.AcceptOpen,
                modal=True,
            )
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly)
            dlg.setDirectory(startdir)
            dlg.setAttribute(Qt.WA_DeleteOnClose)

            @dlg.accepted.connect
            def on_accepted():
                dirpath = dlg.selectedFiles()
                if dirpath:
                    self.setCurrentPath(dirpath[0])
                    self.start()
            dlg.open()
        else:
            dirpath = QFileDialog.getExistingDirectory(
                self, caption, startdir
            )
            if dirpath:
                self.setCurrentPath(dirpath)
                self.start()

    def __onRecentActivated(self, index):
        item = self.recent_cb.itemData(index)
        if item is None:
            return
        assert isinstance(item, RecentPath)
        self.setCurrentPath(item.abspath)
        self.start()

    def __updateInfo(self):
        if self.__state == State.NoState:
            text = "No document set selected"
        elif self.__state == State.Processing:
            text = "Processing"
        elif self.__state == State.Done:
            nvalid = self.n_text_data
            ncategories = self.n_text_categories
            n_skipped = len(self.skipped_documents)
            if ncategories < 2:
                text = "{} document{}".format(nvalid,
                                              "s" if nvalid != 1 else "")
            else:
                text = "{} documents / {} categories".format(nvalid,
                                                             ncategories)
            if n_skipped > 0:
                text = text + ", {} skipped".format(n_skipped)
        elif self.__state == State.Cancelled:
            text = "Cancelled"
        elif self.__state == State.Error:
            text = "Error state"
        else:
            assert False

        self.info_area.setText(text)

        if self.__state == State.Processing:
            self.infostack.setCurrentIndex(1)
        else:
            self.infostack.setCurrentIndex(0)

    def setCurrentPath(self, path):
        """
        Set the current root text path to path

        If the path does not exists or is not a directory the current path
        is left unchanged

        Parameters
        ----------
        path : str
            New root import path.

        Returns
        -------
        status : bool
            True if the current root import path was successfully
            changed to path.
        """
        if self.currentPath is not None and path is not None and \
                os.path.isdir(self.currentPath) and os.path.isdir(path) and \
                os.path.samefile(self.currentPath, path) and \
                self.source == self.LOCAL_FILE:
            return True

        success = True
        error = None
        if path is not None:
            if not os.path.exists(path):
                error = "'{}' does not exist".format(path)
                path = None
                success = False
            elif not os.path.isdir(path):
                error = "'{}' is not a folder".format(path)
                path = None
                success = False

        if error is not None:
            self.error(error)
            warnings.warn(error, UserWarning, stacklevel=3)
        else:
            self.error()

        if path is not None:
            newindex = self.addRecentPath(path)
            self.recent_cb.setCurrentIndex(newindex)
            if newindex >= 0:
                self.currentPath = path
            else:
                self.currentPath = None
        else:
            self.currentPath = None
        self.__actions.reload.setEnabled(self.currentPath is not None)

        if self.__state == State.Processing:
            self.cancel()
        self.source = self.LOCAL_FILE
        return success

    def addRecentPath(self, path):
        """
        Prepend a path entry to the list of recent paths

        If an entry with the same path already exists in the recent path
        list it is moved to the first place

        Parameters
        ----------
        path : str
        """
        existing = None
        for pathitem in self.recent_paths:
            try:
                if os.path.samefile(pathitem.abspath, path):
                    existing = pathitem
                    break
            except FileNotFoundError:
                # file not found if the `pathitem.abspath` no longer exists
                pass

        model = self.recent_cb.model()

        if existing is not None:
            selected_index = self.recent_paths.index(existing)
            assert model.item(selected_index).data(Qt.UserRole) is existing
            self.recent_paths.remove(existing)
            row = model.takeRow(selected_index)
            self.recent_paths.insert(0, existing)
            model.insertRow(0, row)
        else:
            item = RecentPath(path, None, None)
            self.recent_paths.insert(0, item)
            model.insertRow(0, RecentPath_asqstandarditem(item))
        return 0

    def __setRuntimeState(self, state):
        assert state in State
        self.setBlocking(state == State.Processing)
        message = ""
        if state == State.Processing:
            assert self.__state in [State.Done,
                                    State.NoState,
                                    State.Error,
                                    State.Cancelled]
            message = "Processing"
        elif state == State.Done:
            assert self.__state == State.Processing
        elif state == State.Cancelled:
            assert self.__state == State.Processing
            message = "Cancelled"
        elif state == State.Error:
            message = "Error during processing"
        elif state == State.NoState:
            message = ""
        else:
            assert False

        self.__state = state

        if self.__state == State.Processing:
            self.infostack.setCurrentIndex(1)
        else:
            self.infostack.setCurrentIndex(0)

        self.setStatusMessage(message)
        self.__updateInfo()

    def reload(self):
        """
        Restart the text scan task
        """
        if self.__state == State.Processing:
            self.cancel()
        self.source = self.LOCAL_FILE
        self.corpus = None
        self.start()

    def start(self):
        """
        Start/execute the text indexing operation
        """
        self.error()
        self.Warning.clear()
        self.progress_widget.setValue(0)

        self.__invalidated = False
        startdir = self.currentPath if self.source == self.LOCAL_FILE \
            else self.url_combo.currentText().strip()
        if not startdir:
            return

        if self.__state == State.Processing:
            assert self.__pendingTask is not None
            log.info("Starting a new task while one is in progress. "
                     "Cancel the existing task (dir:'{}')"
                     .format(self.__pendingTask.startdir))
            self.cancel()

        self.__setRuntimeState(State.Processing)

        report_progress = methodinvoke(
            self, "__onReportProgress", (object,))

        task = ImportDocuments(startdir, self.source == self.URL,
                               report_progress=report_progress)

        # collect the task state in one convenient place
        self.__pendingTask = taskstate = namespace(
            task=task,
            startdir=startdir,
            future=None,
            watcher=None,
            cancelled=False,
            cancel=None,
        )

        def cancel():
            # Cancel the task and disconnect
            if taskstate.future.cancel():
                pass
            else:
                taskstate.task.cancelled = True
                taskstate.cancelled = True
                try:
                    taskstate.future.result(timeout=0)
                except UserInterruptError:
                    pass
                except TimeoutError:
                    log.info("The task did not stop in in a timely manner")
            taskstate.watcher.finished.disconnect(self.__onRunFinished)

        taskstate.cancel = cancel

        def run_text_scan_task_interupt():
            try:
                return task.run()
            except UserInterruptError:
                # Suppress interrupt errors, so they are not logged
                return

        taskstate.future = self.__executor.submit(run_text_scan_task_interupt)
        taskstate.watcher = FutureWatcher(taskstate.future)
        taskstate.watcher.finished.connect(self.__onRunFinished)

    @Slot()
    def __onRunFinished(self):
        assert QThread.currentThread() is self.thread()
        assert self.__state == State.Processing
        assert self.__pendingTask is not None
        assert self.sender() is self.__pendingTask.watcher
        assert self.__pendingTask.future.done()
        task = self.__pendingTask
        self.__pendingTask = None

        corpus, errors, lemmas, pos, ner, is_conllu = None, [], None, None, \
                                                      None, False
        try:
            corpus, errors, lemmas, pos, ner, is_conllu = task.future.result()
        except NoDocumentsException:
            state = State.Error
            self.error("Folder contains no readable files.")
        except Exception:
            sys.excepthook(*sys.exc_info())
            state = State.Error
            self.error(traceback.format_exc())
        else:
            state = State.Done
            self.error()

        if corpus:
            self.n_text_data = len(corpus)
            self.n_text_categories = len(corpus.domain.class_var.values) \
                if corpus.domain.class_var else 0

        self.base_corpus = self.corpus = corpus
        self.is_conllu = is_conllu
        self.tokens = lemmas
        self.pos = pos
        self.ner = ner
        if self.corpus:
            self.corpus.name = "Documents"
        self.skipped_documents = errors

        if len(errors):
            self.Warning.read_error(
                "Some files" if len(errors) > 1 else "One file"
            )

        self.__setRuntimeState(state)
        self.commit()

    def cancel(self):
        """
        Cancel current pending task (if any).
        """
        if self.__state == State.Processing:
            assert self.__pendingTask is not None
            self.__pendingTask.cancel()
            self.__pendingTask = None
            self.__setRuntimeState(State.Cancelled)

    @Slot(object)
    def __onReportProgress(self, arg):
        # report on scan progress from a worker thread
        # arg must be a namespace(count: int, lastpath: str)
        assert QThread.currentThread() is self.thread()
        if self.__state == State.Processing:
            self.pathlabel.setText(prettifypath(arg.lastpath))
            self.progress_widget.setValue(int(100 * arg.progress))

    def add_features(self):
        lemma, pos, ner = self.lemma_cb, self.pos_cb, self.ner_cb
        if self.corpus is None:
            return
        self.corpus = self.base_corpus.copy()
        if lemma:
            self.corpus.store_tokens(self.tokens)
        if pos:
            tags = array(self.pos, dtype=object)
            self.corpus.pos_tags = tags
        if ner:
            var = StringVariable("named entities")
            self.corpus = self.corpus.add_column(var, self.ner)

    def commit(self):
        """
        Create and commit a Corpus from the collected text meta data.
        """
        if self.is_conllu:
            self.add_features()
        self.Outputs.data.send(self.corpus)
        if self.skipped_documents:
            skipped_table = (
                Table.from_list(
                    SKIPPED_DOMAIN,
                    [[x, os.path.join(self.currentPath, x)]
                     for x in self.skipped_documents]
                )
            )
            skipped_table.name = "Skipped documents"
        else:
            skipped_table = None
        self.Outputs.skipped_documents.send(skipped_table)

    def onDeleteWidget(self):
        self.cancel()
        self.__executor.shutdown(wait=True)
        self.__invalidated = False

    def eventFilter(self, receiver, event):
        # re-implemented from QWidget
        # intercept and process drag drop events on the recent directory
        # selection combo box
        def dirpath(event):
            # type: (QDropEvent) -> Optional[str]
            """Return the directory from a QDropEvent."""
            data = event.mimeData()
            urls = data.urls()
            if len(urls) == 1:
                url = urls[0]
                path = url.toLocalFile()
                if os.path.isdir(path):
                    return path
            return None

        if receiver is self.recent_cb and \
                event.type() in {QEvent.DragEnter, QEvent.DragMove,
                                 QEvent.Drop}:
            assert isinstance(event, QDropEvent)
            path = dirpath(event)
            if path is not None and event.possibleActions() & Qt.LinkAction:
                event.setDropAction(Qt.LinkAction)
                event.accept()
                if event.type() == QEvent.Drop:
                    self.setCurrentPath(path)
                    self.start()
            else:
                event.ignore()
            return True

        return super().eventFilter(receiver, event)

    def send_report(self):
        if not self.currentPath:
            return
        items = [('Path', self.currentPath),
                 ('Number of documents', self.n_text_data)]
        if self.n_text_categories:
            items += [('Categories', self.n_text_categories)]
        if self.skipped_documents:
            items += [('Number of skipped', len(self.skipped_documents))]
        self.report_items(items, )


class UserInterruptError(BaseException):
    """
    A BaseException subclass used for cooperative task/thread cancellation
    """
    pass


def main(argv=sys.argv):
    app = QApplication(list(argv))
    argv = app.arguments()
    if len(argv) > 1:
        path = argv[1]
    else:
        path = None
    w = OWImportDocuments()
    w.show()
    w.raise_()

    if path is not None:
        w.setCurrentPath(path)

    app.exec_()
    w.saveSettings()
    w.onDeleteWidget()
    return 0


if __name__ == "__main__":
    sys.exit(main())
