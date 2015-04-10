from Orange.widgets.widget import OWWidget
from Orange.widgets import gui


class MyWidget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Example Widget"
    icon = "icons/example.svg"

    def __init__(self):
        super().__init__()

        gui.label(self, self, "test")
