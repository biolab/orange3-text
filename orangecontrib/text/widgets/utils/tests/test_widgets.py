import os
import unittest
from unittest.mock import patch, ANY

from orangecontrib.text.corpus import get_sample_corpora_dir
from orangecontrib.text.widgets.utils import FileWidget, QFileDialog
from orangewidget.tests.base import GuiTest


class TestFileWidget(GuiTest):
    @patch.object(QFileDialog, "getOpenFileName", return_value=("path", ""))
    def test_start_dir(self, mock):
        file_widget = FileWidget(
            recent_files=[],
            icon_size=(16, 16),
            dialog_title="Open Orange Document Corpus",
            reload_label="Reload",
            browse_label="Browse",
            allow_empty=False,
            minimal_width=250,
        )
        file_widget.browse()
        mock.assert_called_with(ANY, ANY, "~/", ANY)

        file_widget.recent_files = ["book-excerpts.tab"]
        file_widget.browse()
        mock.assert_called_with(ANY, ANY, get_sample_corpora_dir(), ANY)

        cur_dir = os.path.dirname(__file__)
        file_widget.recent_files.insert(0, os.path.join(cur_dir, "file.tab"))
        file_widget.browse()
        mock.assert_called_with(ANY, ANY, cur_dir, ANY)

        # dir doesn't exit case
        file_widget.recent_files.insert(0, "/non/exiting/dir/file.tab")
        file_widget.browse()
        mock.assert_called_with(ANY, ANY, "~/", ANY)

        # if browse have start_dir argument use this path
        file_widget.browse("/sample/path")
        mock.assert_called_with(ANY, ANY, "/sample/path", ANY)


if __name__ == "__main__":
    unittest.main()
