# coding: utf-8
from collections import OrderedDict
from os import path
from math import pi as PI

import numpy as np

from PyQt4 import QtCore, QtGui

from Orange.widgets import widget, gui, settings
from Orange.data import Table
from orangecontrib.text.country_codes import \
    CC_EUROPE, INV_CC_EUROPE, SET_CC_EUROPE, \
    CC_WORLD, INV_CC_WORLD, \
    CC_USA, INV_CC_USA, SET_CC_USA


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
    outputs = [('Data', Table)]

    want_main_area = False

    selected_attr = settings.Setting(0)
    selected_map = settings.Setting(0)

    def __init__(self):
        super().__init__()
        self._create_layout()

    @QtCore.pyqtSlot(str, result=str)
    def region_selected(self, regions):
        """Called from JavaScript"""
        if not regions:
            return self.send('Data', None)
        regions = regions.split(',')
        from Orange.data.filter import FilterStringList
        # TODO, FIXME: make this work for discrete attrs also
        filter = FilterStringList(self.metas[self.selected_attr], regions)
        self.send('Data', self.data._filter_values(filter))

    def _create_layout(self):
        box = gui.widgetBox(self.controlArea,
                            orientation='horizontal')
        self.attr_combo = gui.comboBox(box, self, 'selected_attr',
                                       orientation='horizontal',
                                       label='Region attribute:',
                                       callback=self.on_attr_change)
        self.map_combo = gui.comboBox(box, self, 'selected_map',
                                      orientation='horizontal',
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
<style>
html, body, #map {margin:0px;padding:0px;width:100%;height:100%;}
</style>
<link  href="resources/jquery-jvectormap-2.0.2.css" rel="stylesheet">
</head>
<body>
<div id="map"></div>
</body>
</html>'''
        self.webview = gui.WebviewWidget(self.controlArea, self, debug=True)
        self.webview.setHtml(html, 'file://{}/'.format(path.dirname(__file__)))
        for script in ('jquery-2.1.4.min.js',
                       'jquery-jvectormap-2.0.2.min.js',
                       'jquery-jvectormap-world-mill-en.js',
                       'jquery-jvectormap-europe-mill-en.js',
                       'jquery-jvectormap-us-aea-en.js',
                       'geomap-script.js'):
            self.webview.evalJS(open(path.join(path.dirname(__file__), 'resources', script), encoding='utf-8').read())
        self.webview.evalJS('REGIONS = {};'.format({Map.WORLD: CC_WORLD,
                                                    Map.EUROPE: CC_EUROPE,
                                                    Map.USA: CC_USA}))

    def _repopulate_attr_combo(self, data):
        from itertools import chain
        self.metas = [a for a in chain(data.domain.metas,
                                       data.domain.attributes,
                                       data.domain.class_vars)
                      # Filter string variables
                      if (a.is_discrete and a.values and isinstance(a.values[0], str) and not a.ordered or
                          a.is_string)] if data else []
        self.attr_combo.clear()
        for i, var in enumerate(self.metas):
            self.attr_combo.addItem(gui.attributeIconDict[var], var.name)
            # Select default attribute
            if var.name.lower() == 'country':
                self.selected_attr = i
        self.attr_combo.setCurrentIndex(self.attr_combo.findText(self.metas[self.selected_attr].name))

    def on_data(self, data):
        self.data = data
        self._repopulate_attr_combo(data)
        self.on_attr_change()

    def on_map_change(self, map_code=''):
        if map_code:
            self.map_combo.setCurrentIndex(self.map_combo.findData(map_code))
        else:
            map_code = map_code or self.map_combo.itemData(self.selected_map)
        self.webview.evalJS('MAP_CODE = "{}";'.format(map_code))
        self.webview.evalJS('renderMap();')

    def on_attr_change(self):
        attr = self.metas[self.selected_attr]
        from collections import Counter, defaultdict
        counts = Counter(self.data.get_column_view(self.data.domain.index(attr))[0])
        # Auto-select region map
        values = set(counts)
        if 0 == len(values - SET_CC_USA):
            cc_map, inv_cc_map, map_code = CC_USA, INV_CC_USA, Map.USA
        elif 0 == len(values - SET_CC_EUROPE):
            cc_map, inv_cc_map, map_code = CC_EUROPE, INV_CC_EUROPE, Map.EUROPE
        else:
            cc_map, inv_cc_map, map_code = CC_WORLD, INV_CC_WORLD, Map.WORLD
        data = defaultdict(int)
        for cc in counts:
            data[inv_cc_map.get(cc, cc)] += counts[cc]
        self.webview.evalJS('DATA = {};'.format(dict(data)))
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
