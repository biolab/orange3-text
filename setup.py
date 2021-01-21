#!/usr/bin/env python

import os
import subprocess
import warnings
from unittest import TestSuite

from setuptools import setup, find_packages

try:
    # need recommonmark for build_htmlhelp command
    import recommonmark
except ImportError:
    pass

NAME = 'Orange3-Text'

MAJOR = 1
MINOR = 3
MICRO = 0
IS_RELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
FULL_VERSION = VERSION

DESCRIPTION = 'Orange3 TextMining add-on.'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.pypi')
LONG_DESCRIPTION = open(README_FILE).read()
AUTHOR = 'Bioinformatics Laboratory, FRI UL'
AUTHOR_EMAIL = 'info@biolab.si'
URL = "https://github.com/biolab/orange3-text"
DOWNLOAD_URL = "https://github.com/biolab/orange3-text/tarball/{}".format(VERSION)


KEYWORDS = [
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3-text',
    'data mining',
    'orange3 add-on',
]

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

    # Register widget help
    "orange.canvas.help": (
        'html-index = orangecontrib.text.widgets:WIDGET_HELP_PATH',),
}


def git_version():
    """ Return the git revision as a string. Copied from numpy setup.py. """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    """ Copied from numpy setup.py. """
    global FULL_VERSION
    FULL_VERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('orangecontrib/text/version.py'):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load orangecontrib/text/__init__.py
        from importlib.machinery import SourceFileLoader
        version = SourceFileLoader('orangecontrib.text.version',
                                   'orangecontrib/text/version.py').load_module()
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not IS_RELEASED:
        FULL_VERSION += '.dev0+' + GIT_REVISION[:7]

    return FULL_VERSION, GIT_REVISION


def write_version_py(filename='orangecontrib/text/version.py'):
    """ Copied from numpy setup.py. """
    cnt = """
# THIS FILE IS GENERATED FROM ORANGE3-TEXT SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
    short_version += ".dev"
"""
    FULL_VERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULL_VERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(IS_RELEASED)})
    finally:
        a.close()

INSTALL_REQUIRES = sorted(set(
    line.partition('#')[0].strip()
    for line in open(os.path.join(os.path.dirname(__file__), 'requirements.txt'))
) - {''})


def temp_test_suite():
    warnings.warn(
        "The package does not support testing with this command. Please use"
        "python -m unittest discover", FutureWarning)
    return TestSuite([])


if __name__ == '__main__':
    write_version_py()
    setup(
        name=NAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        version=FULL_VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        packages=find_packages(),
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        zip_safe=False,
        test_suite="setup.temp_test_suite"
    )
