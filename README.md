Text Mining Add-on for Orange3
==============================

This is a text mining add-on for [Orange3](http://orange.biolab.si). It extends Orange in scripting and GUI
part.

Installation
------------

To install the add-on with pip use

    pip install Orange3-Text

To install the add-on from source, run

    python setup.py install

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    python setup.py develop

Usage
-----

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    python3 -m Orange.canvas

Text Mining widgets are in the toolbox bar under Text Mining section.