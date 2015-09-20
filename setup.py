#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages

NAME = 'Orange3-Text'

VERSION = '0.1.4'

DESCRIPTION = 'Orange3 TextMining add-on.'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
LONG_DESCRIPTION = open(README_FILE).read()
AUTHOR = 'Bioinformatics Laboratory, FRI UL'
AUTHOR_EMAIL = 'contact@orange.biolab.si'
URL = "https://github.com/biolab/orange3-text"
DOWNLOAD_URL = "https://github.com/biolab/orange3-text/tarball/{}".format(VERSION)

ENTRY_POINTS = {
    'orange3.addon': (
        'text = orangecontrib.text',
    ),
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        'exampletutorials = orangecontrib.text.tutorials',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/text/widgets/__init__.py
        'Text Mining = orangecontrib.text.widgets',
    ),
}

KEYWORDS = [
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3-text',
    'data mining',
    'orange3 add-on',
]

INSTALL_REQUIRES = (
    'Orange >= 3.1',
    'nltk',
    'scikit-learn',
    'numpy',
    'gensim',
)

PACKAGE_DATA = {
    'orangecontrib.text': ['tutorials/*.ows'],
    'orangecontrib.text.widgets': ['icons/*'],
    'orangecontrib.text.tests': ['nyt-cache.txt'],
}

if 'test' in sys.argv:
    extra_setuptools_args = dict(
        test_suite='orangecontrib.text.tests',
    )
else:
    extra_setuptools_args = dict()

if __name__ == '__main__':
    setup(
        name=NAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        packages=find_packages(),
        package_data=PACKAGE_DATA,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
        **extra_setuptools_args
    )
