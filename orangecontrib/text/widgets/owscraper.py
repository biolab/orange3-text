from newspaper import Article, ArticleException
from datetime import datetime
import numpy as np
import csv
import os
import warnings

from PyQt4 import QtCore, QtGui

from Orange.canvas.utils import environ
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget
from Orange.data import Table, StringVariable, Domain
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.scraper import ARTICLE_TEXT_FIELDS, _date_to_str, _get_info


class OWScraper(OWWidget):
	# Basic widget info
    name = "Article Scraper"
    description = "gets article, title, author, date etc. from the URL"
    icon = "icons/Scraper.svg"
    priority = 100
    category="text"

    # Input/output
    inputs = [("Data", Table, "set_data")]
    outputs = [("Corpus", Corpus)]
  
    want_main_area = False
   
    # Settings
    selected_url = Setting(0)
    
    # Output includes checkboxes.
    includes_article = Setting(True)
    includes_title = Setting(True)
    includes_author = Setting(True)
    includes_date = Setting(True)
    includes_web_url = Setting(True)


    def __init__(self):
        super().__init__()


        self.output_corpus=None
        self.url_list=[]
        self.metas=[]

        # To hold all the controls. Makes access easier.
        self.scraper_controls = []
        box = gui.widgetBox(self.controlArea,
                            orientation='horizontal')
        self.url_combo = gui.comboBox(box, self, 'selected_url',
                                       orientation='horizontal',
                                       label='URL attribute:',
                                       callback=self.on_url_change)
        # Text includes box.
        self.text_includes_box = gui.widgetBox(self.controlArea,
                                               "Output", addSpace=True)
        self.article_chbox = gui.checkBox(self.text_includes_box, self,
                                           "includes_article", "Article")
        self.author_chbox = gui.checkBox(self.text_includes_box, self,
                                         "includes_author", "Author")
        self.date_chbox = gui.checkBox(self.text_includes_box, self,
                                          "includes_date", "Date")
        self.title_chbox = gui.checkBox(self.text_includes_box, self,
                                           "includes_title", "Title")
        self.web_url_chbox = gui.checkBox(self.text_includes_box, self,
                                          "includes_web_url", "URL")

        self.scraper_controls.append(self.article_chbox)
        self.scraper_controls.append(self.author_chbox)
        self.scraper_controls.append(self.date_chbox)
        self.scraper_controls.append(self.title_chbox)
        self.scraper_controls.append(self.web_url_chbox)
        self.run_query_button = gui.button(self.controlArea, self, 'Run Query',callback=self.apply,
                                           tooltip="Run the chosen article Query")
        
        self.scraper_controls.append(self.run_query_button)


    def on_url_change(self):
        """
        callback function
        :return:
        """
        url_attr = self.metas[self.selected_url]
        if not url_attr.is_string :
            return self.warning(0, 'Only String attributes supported.')
        else:
            self.url_list=self.data.get_column_view(self.data.domain.index(url_attr))[0]

    def repopulate_url_combo(self, data):
        """
        change in attribute selected
        :param data:
        :return: attribute name
        """
        from itertools import chain
        self.metas = [a for a in chain(data.domain.metas,
                                       data.domain.attributes,
                                       data.domain.class_vars)
                      # Filter string variables
                      if (a.is_discrete and a.values and isinstance(a.values[0], str) and not a.ordered or
                          a.is_string)] if data else []
        self.url_combo.clear()
        self.selected_url = 0
        for i, var in enumerate(self.metas):
            self.url_combo.addItem(gui.attributeIconDict[var], var.name)
            # Select default url attribute
            if var.name == 'url':
                self.selected_url = i
        if self.metas:
            self.url_combo.setCurrentIndex(self.url_combo.findText(self.metas[self.selected_url].name))

    def generate_corpus(self, url_list):
        """
        generate new corpus with values requested by user
        :param url_list:
        :return: corpus
        """
        new_table=None
        text_includes_params = [self.includes_article, self.includes_author, self.includes_date, 
                                 self.includes_title, self.includes_web_url]
        if True not in text_includes_params:
            self.warning(1, "You must select at least one text field.")
            return
        required_text_fields = [incl_field for yes, incl_field in zip(text_includes_params, ARTICLE_TEXT_FIELDS) if yes]
        meta_vars = [StringVariable.make(field) for field in required_text_fields]
        metadata=[]
        for url in url_list:
            info, is_cached =_get_info(url)
            final_fields = [incl_field for yes, incl_field in zip(text_includes_params, info) if yes]
            metadata.append(final_fields)
        metadata = np.array(metadata, dtype=object)
        metas=metadata
        domain = Domain([], class_vars=None, metas=(meta_vars))
        new_table = Corpus(None, None, metadata, domain, meta_vars)
        self.output_corpus=new_table
        self.send("Corpus",self.output_corpus)

    def apply(self):
        self.on_url_change()
        self.generate_corpus(self.url_list)

    def set_data(self, data):
        """
        gets the input data
        generates a new corpus with the requested values
        :param : data
        """
        self.data = data
        self.repopulate_url_combo(data)


def main():
    app = QtGui.QApplication([''])
    widget = OWScraper()
    widget.show()
    Data=Table("../datasets/url")
    widget.set_data(Data)
    app.exec()        


if __name__ == "__main__":
    main()
