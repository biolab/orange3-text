# coding: utf-8
from urllib.parse import urljoin
from urllib.request import pathname2url
from itertools import chain

from Orange.widgets.utils.itemmodels import VariableListModel
from collections import defaultdict, Counter
from os import path
from math import pi as PI
import re

import numpy as np

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt, QTimer

from Orange.widgets import widget, gui, settings
from Orange.data import Table
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


class OWGeoMap(widget.OWWidget):
    name = "GeoMap"
    priority = 20000
    icon = "icons/GeoMap.svg"
    inputs = [("Data", Table, "on_data")]
    outputs = [('Corpus', Corpus)]

    want_main_area = False

    selected_attr = settings.Setting('')
    selected_map = settings.Setting(0)
    regions = settings.Setting([])

    def __init__(self):
        super().__init__()
        self.data = None
        self._create_layout()

    @QtCore.pyqtSlot(str, result=str)
    def region_selected(self, regions):
        """Called from JavaScript"""
        if not regions:
            self.regions = []
        if not regions or self.data is None:
            return self.send('Corpus', None)
        self.regions = regions.split(',')
        attr = self.data.domain[self.selected_attr]
        if attr.is_discrete: return  # TODO, FIXME: make this work for discrete attrs also
        from Orange.data.filter import FilterRegex
        filter = FilterRegex(attr, r'\b{}\b'.format(r'\b|\b'.join(self.regions)), re.IGNORECASE)
        self.send('Corpus', self.data._filter_values(filter))

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
        hexpand = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding,
                                    QtGui.QSizePolicy.Fixed)
        self.attr_combo.setSizePolicy(hexpand)
        self.map_combo.setSizePolicy(hexpand)
        html = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<base href="{}/"/>
<style>
html, body, #map {{margin:0px;padding:0px;width:100%;height:100%;}}
</style>
<link href="resources/jquery-jvectormap-2.0.2.css" rel="stylesheet">
<script src="resources/jquery-2.1.4.min.js"></script>
<script src="resources/jquery-jvectormap-2.0.2.min.js"></script>
<script src="resources/jquery-jvectormap-world-mill-en.js"></script>
<script src="resources/jquery-jvectormap-europe-mill-en.js"></script>
<script src="resources/jquery-jvectormap-us-aea-en.js"></script>
<script src="resources/geomap-script.js"></script>
</head>
<body>
<div id="map"></div>
</body>
</html>'''.format(urljoin('file:', pathname2url(path.abspath(path.dirname(__file__)))))
        self.webview = gui.WebviewWidget(self.controlArea, self, debug=False)
        self.controlArea.layout().addWidget(self.webview)
        self.webview.setHtml(html)
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
                              Map.EUROPE: (INV_CC_EUROPE, CC_EUROPE)} [map_code]
        # Set country counts in JS
        data = defaultdict(int)
        for cc in getattr(self, 'cc_counts', ()):
            key = inv_cc_map.get(cc, cc)
            if key in cc_map:
                data[key] += self.cc_counts[cc]
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
        attr = self.data.domain[self.selected_attr]
        self.cc_counts = Counter(chain.from_iterable(
            set(name.strip() for name in CC_NAMES.findall(i.lower())) if len(i) > 3 else (i,)
            for i in self.data.get_column_view(self.data.domain.index(attr))[0]))
        # Auto-select region map
        values = set(self.cc_counts)
        if 0 == len(values - SET_CC_USA):
            map_code = Map.USA
        elif 0 == len(values - SET_CC_EUROPE):
            map_code = Map.EUROPE
        else:
            map_code = Map.WORLD
        self.on_map_change(map_code)


def main():
    from Orange.data import Table, Domain, ContinuousVariable, StringVariable

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
    app = QtGui.QApplication([''])
    w = OWGeoMap()
    w.on_data(table)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
