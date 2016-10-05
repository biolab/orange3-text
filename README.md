Orange3 Text 
============

[![Build Status](https://travis-ci.org/biolab/orange3-text.svg?branch=master)](https://travis-ci.org/biolab/orange3-text)
[![codecov](https://codecov.io/gh/biolab/orange3-text/branch/master/graph/badge.svg)](https://codecov.io/gh/biolab/orange3-text)
[![Documentation Status](https://readthedocs.org/projects/orange3-text/badge/?version=latest)](http://orange3-text.readthedocs.org/en/latest/?badge=latest)

Orange3 Text extends [Orange3](http://orange.biolab.si), a data mining software
package, with common functionality for text mining. It provides access
to publicly available data, like NY Times, Twitter, Wikipedia and PubMed. Furthermore,
it provides tools for preprocessing, constructing vector spaces (like
bag-of-words, topic modeling, and similarity hashing) and visualizations like word cloud
end geo map. All features can be combined with powerful data mining techniques
from the Orange data mining framework.

Anaconda installation
---------------------

The easiest way to install Orange3-Text is with Anaconda distribution. Download [Anaconda](https://www.continuum.io/downloads) 
for your OS (Python version 3.5). In your Anaconda Prompt first add conda-forge to your channels:

    conda config --add channels conda-forge

Then install Orange3

    conda install orange3

This will install the latest release of Orange. Install biopython to access PubMed widget (optional!)

    conda install biopython

Biopython is an optional dependency and not installing it will simply result in a missing PubMed widget. Then

    pip install Orange3-Text

Run

   python -m Orange.canvas

to open Orange and check if everything is installed properly.

Installation from source
------------------------

To install the add-on from source

    # Clone the repository and move into it
    git clone https://github.com/biolab/orange3-text.git
    cd orange3-text

    # Install the dependencies:
    pip install requirements.txt
    pip install requirements-opt.txt # Optional dependecy for PubMed, requires compiler.

    # Finally install Orange3-Text in editable/development mode.
    pip install -e .

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    python setup.py develop

Windows setup for biopython library
-----------------------------------

If you're not using Anaconda distribution, you can manually install biopython library before installing the add-on.
First, download the compiler and then run the setup with

Windows + [GCC](https://gcc.gnu.org/releases.html):

    python setup.py build_ext --inplace --compiler=mingw32 install

Windows + [Visual Studio](http://landinghub.visualstudio.com/visual-cpp-build-tools):

    python setup.py build_ext --inplace --compiler=msvc install

Usage
-----

After the installation, the widgets from this add-on are registered with Orange. To run Orange from the terminal,
use

    python3 -m Orange.canvas

or

	orange-canvas

The new widgets are in the toolbox bar under Text Mining section.
