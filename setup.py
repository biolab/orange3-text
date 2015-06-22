#!/usr/bin/env python

from setuptools import setup, find_packages

VERSION = '0.1.1'

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

KEYWORDS = (
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3-text',
    'data mining',
    'orange3 add-on',
)

if __name__ == '__main__':
    setup(
        name="Orange3-Text",
        description="Orange TextMining add-on.",
        version=VERSION,
        author='Bioinformatics Laboratory, FRI UL',
        author_email='contact@orange.biolab.si',
        url="https://github.com/nikicc/orange3-text",
        download_url="https://github.com/nikicc/orange3-text/tarball/{}".format(VERSION),
        packages=find_packages(),
        package_data={
            'orangecontrib.text': ['tutorials/*.ows'],
            'orangecontrib.text.widgets': ['icons/*'],
        },
        install_requires=['Orange', 'nltk', 'scikit-learn', 'numpy', 'gensim', 'appdirs'],
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
    )
