#!/usr/bin/env python

import os

from setuptools import setup

ENTRY_POINTS = {
    'orange.widgets.tutorials': (
        'exampletutorials = orangecontrib.example.exampletutorials',
    ),
    'orange.widgets': (
        'Example = orangecontrib.example.widgets',
    ),
}

if __name__ == '__main__':
    setup(
        name="Orange3 Example Add-on",
        packages=['orangecontrib',
                  'orangecontrib.example',
                  'orangecontrib.example.exampletutorials',
                  'orangecontrib.example.widgets'],
        package_data={
          'orangecontrib.example': ['exampletutorials/*.ows'],
          'orangecontrib.example.widgets': ['icons/*'],
        },
        install_requires=['Orange'],
        entry_points=ENTRY_POINTS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
    )
