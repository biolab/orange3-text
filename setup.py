#!/usr/bin/env python

from setuptools import setup

ENTRY_POINTS = {
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
)

if __name__ == '__main__':
    setup(
        name="Orange3-Text",
        packages=['orangecontrib',
                  'orangecontrib.text',
                  'orangecontrib.text.tutorials',
                  'orangecontrib.text.widgets'],
        package_data={
            'orangecontrib.text': ['tutorials/*.ows'],
            'orangecontrib.text.widgets': ['icons/*'],
        },
        install_requires=['Orange'],
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
    )
