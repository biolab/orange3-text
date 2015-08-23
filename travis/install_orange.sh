#!/usr/bin/env bash
git clone https://github.com/biolab/orange3
cd orange3
pip install -r requirements.txt
python setup.py develop
cd $TRAVIS_BUILD_DIR
