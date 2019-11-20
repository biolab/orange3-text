# coding: utf-8
import os
import re
from collections import defaultdict
from itertools import chain
from urllib.parse import urljoin
from urllib.request import pathname2url

import numpy as np
from AnyQt.QtCore import Qt, QTimer, pyqtSlot, QUrl, QObject
from AnyQt.QtWidgets import QApplication, QSizePolicy

from Orange.data import Table
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import Input, Output
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.country_codes import \
    CC_EUROPE, INV_CC_EUROPE, SET_CC_EUROPE, \
    CC_WORLD, INV_CC_WORLD, \
    CC_USA, INV_CC_USA, SET_CC_USA

CC_NAMES = re.compile('[\w\s\.\-]+')


class Map:
    WORLD = 'world_mill_en'
    EUROPE = 'europe_mill_en'
    USA = 'us_aea_en'
    all = (('World',  WORLD),
           ('Europe', EUROPE),
           ('USA',    USA))


class OWDocMap(widget.OWWidget):
    name = "Document Map"
    priority = 530
    icon = "icons/DocMap.svg"
    replaces = ["orangecontrib.text.widgets.owgeomap.OWGeoMap"]
    keywords = ["GeoMap"]

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        corpus = Output("Corpus", Corpus)

    want_main_area = False

    selected_attr = settings.Setting('')
    selected_map = settings.Setting(0)
    regions = settings.Setting([])

    def __init__(self):
        super().__init__()
        self.data = None
        self._create_layout()

    def region_selected(self, regions):
        """Called from JavaScript"""
        if not regions:
            self.regions = []
        if not regions or self.data is None:
            return self.Outputs.corpus.send(None)
        self.regions = regions.split(',')
        attr = self.data.domain[self.selected_attr]
        if attr.is_discrete: return  # TODO, FIXME: make this work for discrete attrs also
        from Orange.data.filter import FilterRegex
        filter = FilterRegex(attr, r'\b{}\b'.format(r'\b|\b'.join(self.regions)), re.IGNORECASE)
        self.Outputs.corpus.send(self.data._filter_values(filter))

    def _create_layout(self):
        box = gui.widgetBox(self.controlArea,
                            orientation='horizontal')
        self.varmodel = VariableListModel(parent=self)
        self.attr_combo = gui.comboBox(box, self, 'selected_attr',
                                       orientation=Qt.Horizontal,
                                       label='Region attribute:',
                                       callback=self.on_attr_change,
                                       sendSelectedValue=True)
        self.attr_combo.setModel(self.varmodel)
        self.map_combo = gui.comboBox(box, self, 'selected_map',
                                      orientation=Qt.Horizontal,
                                      label='Map type:',
                                      callback=self.on_map_change,
                                      items=Map.all)
        hexpand = QSizePolicy(QSizePolicy.Expanding,
                              QSizePolicy.Fixed)
        self.attr_combo.setSizePolicy(hexpand)
        self.map_combo.setSizePolicy(hexpand)

        url = urljoin('file:',
                      pathname2url(os.path.join(
                          os.path.dirname(__file__),
                          'resources',
                          'owdocmap.html')))

        class Bridge(QObject):
            @pyqtSlot(str)
            def region_selected(_, regions):
                return self.region_selected(regions)

        self.webview = gui.WebviewWidget(self.controlArea, Bridge(), url=QUrl(url), debug=False)
        self.controlArea.layout().addWidget(self.webview)

        QTimer.singleShot(
            0, lambda: self.webview.evalJS('REGIONS = {};'.format({Map.WORLD: CC_WORLD,
                                                                   Map.EUROPE: CC_EUROPE,
                                                                   Map.USA: CC_USA})))

    def _repopulate_attr_combo(self, data):
        vars = [a for a in chain(data.domain.metas,
                                 data.domain.attributes,
                                 data.domain.class_vars)
                if a.is_string] if data else []
        self.varmodel.wrap(vars)
        # Select default attribute
        self.selected_attr = next((var.name
                                   for var in vars
                                   if var.name.lower().startswith(('country', 'location', 'region'))),
                                  vars[0].name if vars else '')

    @Inputs.data
    def on_data(self, data):
        if data and not isinstance(data, Corpus):
            data = Corpus.from_table(data.domain, data)
        self.data = data
        self._repopulate_attr_combo(data)
        if not data:
            self.region_selected('')
            QTimer.singleShot(0, lambda: self.webview.evalJS('DATA = {}; renderMap();'))
        else:
            QTimer.singleShot(0, self.on_attr_change)

    def on_map_change(self, map_code=''):
        if map_code:
            self.map_combo.setCurrentIndex(self.map_combo.findData(map_code))
        else:
            map_code = self.map_combo.itemData(self.selected_map)

        inv_cc_map, cc_map = {Map.USA: (INV_CC_USA, CC_USA),
                              Map.WORLD: (INV_CC_WORLD, CC_WORLD),
                              Map.EUROPE: (INV_CC_EUROPE, CC_EUROPE)}[map_code]
        # Set country counts for JS
        data = defaultdict(int)
        for locations in self._iter_locations():
            keys = set(inv_cc_map.get(loc, loc) for loc in locations)
            for key in keys:
                if key in cc_map:
                    data[key] += 1
        # Draw the new map
        self.webview.evalJS('DATA = {};'
                            'MAP_CODE = "{}";'
                            'SELECTED_REGIONS = {};'
                            'renderMap();'.format(dict(data),
                                                  map_code,
                                                  self.regions))

    def on_attr_change(self):
        if not self.selected_attr:
            return
        values = set(chain.from_iterable(self._iter_locations()))
        # Auto-select region map
        if 0 == len(values - SET_CC_USA):
            map_code = Map.USA
        elif 0 == len(values - SET_CC_EUROPE):
            map_code = Map.EUROPE
        else:
            map_code = Map.WORLD
        self.on_map_change(map_code)

    def _iter_locations(self):
        """ Iterator that yields an iterable per documents with all its's
        locations. """
        if self.data is not None:
            attr = self.data.domain[self.selected_attr]
            for i in self.data.get_column_view(self.data.domain.index(attr))[0]:
                # If string attr is None instead of empty string skip it.
                # This happens on data sets from WB Indicators.
                if i is not None:
                    if len(i) > 3:
                        yield map(lambda x: x.strip(), CC_NAMES.findall(i.lower()))
                    else:
                        yield (i, )


def main():
    from Orange.data import Table, Domain, StringVariable

    words = np.column_stack([
        'Slovenia Slovenia SVN USA Iraq Iraq Iraq Iraq France FR'.split(),
        'Slovenia Slovenia SVN France FR Austria NL GB GB GB'.split(),
        'Alabama AL Texas TX TX TX MS Montana US-MT MT'.split(),
    ])
    metas = [
        StringVariable('World'),
        StringVariable('Europe'),
        StringVariable('USA'),
    ]
    domain = Domain([], metas=metas)
    table = Table.from_numpy(domain,
                             X=np.zeros((len(words), 0)),
                             metas=words)
    app = QApplication([''])
    w = OWDocMap()
    w.on_data(table)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
