#!/bin/bash

set -o errexit

# -W treats warnings as errors to make sure all went ok
sphinx-build -W doc doc/_build
