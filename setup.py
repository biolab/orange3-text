#!/usr/bin/env python

import os

from setuptools import setup

ENTRY_POINTS = {
    'orange.widgets.tutorials': (
        'exampletutorials = orangecontrib.astaric.exampletutorials',
    ),
}

NAMESPACE_PACAKGES = ["orangecontrib", "orangecontrib.astaric"]

if __name__ == '__main__':
    setup(
        name="Custom Orange Tutorial",
        packages=['orangecontrib.astaric.exampletutorials'],
        package_data={'orangecontrib.astaric': ['exampletutorials/*.ows']},
        install_requires=['Orange'],
        entry_points=ENTRY_POINTS,
        namespace_packages=['orangecontrib', 'orangecontrib.astaric'],
        include_package_data=True,
        zip_safe=False,
    )
