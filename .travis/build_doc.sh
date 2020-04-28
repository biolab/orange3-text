#!/bin/bash

set -o errexit

pip install sphinx recommonmark

# -W treats warnings as errors to make sure all went ok
sphinx-build -W doc doc/_build

XVFBARGS="-screen 0 1280x1024x24"

# check if the widget catalog in the repository (for orange-hugo is up to date)
cd $TRAVIS_BUILD_DIR/doc/
wget_command="wget -N https://raw.githubusercontent.com/biolab/orange-hugo/master/scripts/create_widget_catalog.py"
run_command="python create_widget_catalog.py --categories \"Text Mining\" --doc ."
eval "$wget_command"
eval "catchsegv xvfb-run -a -s "\"$XVFBARGS"\" $run_command"
diff=$(git diff -- widgets.json)
if [ ! -z "$diff" ]
then
  echo "Widget catalog is stale. Rebuild it with:"
  echo "cd doc"
  echo "$wget_command"
  echo "$run_command"
  echo
  echo "$diff"
  exit 1
fi
