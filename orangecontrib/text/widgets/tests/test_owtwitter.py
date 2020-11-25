from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.widgets.owtwitter import OWTwitter


class TestTwitterWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWTwitter)

    def test_no_error(self):
        self.widget.search()
        self.assertFalse(self.widget.Error.empty_authors.is_shown())

    def test_empty_author_list(self):
        self.widget.mode = 1
        self.widget.mode_toggle()
        self.widget.search()
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.empty_authors.is_shown())
