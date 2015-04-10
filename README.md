Orange3 Example Add-on
======================

This is an example add-on for [Orange3](http://orange.biolab.si). Add-on can extend Orange either 
in scripting or GUI part, or in both. We here focus on GUI part. We here implement a simple (empty) widget,
register it with Orange and add a new workflow with this widget to example tutorials.

Installation
------------

To install the add-on, run

    python setup.py install

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    python setup.py develop

Usage
-----

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    python -m Orange.canvas

The new widget should appear in the toolbox bar under the section Example.
