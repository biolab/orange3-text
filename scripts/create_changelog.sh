#!/bin/bash
set -u -e -o pipefail

function get_version {
    echo $(
        (grep '^MAJOR' setup.py | sed -E "s/.* = //" | tr '\r\n' '.');
        (grep '^MINOR' setup.py | sed -E "s/.* = //" | tr '\r\n' '.');
        (grep '^MICRO' setup.py | sed -E "s/.* = //" | tr -d '\r\n');
        )
}


VERSION=$(get_version)
RELEASE_DATE=$(date "+%Y-%m-%d")

# compile a list of already reported PRs
reported="$(mktemp)"

# TODO: {1,4} - change when more than 10000 PR/issues
grep -E "#[0-9]{1,4}" -o CHANGELOG.md | sort > "$reported"

echo "[$VERSION] - $RELEASE_DATE"
echo "--------------------"
echo '##### Enhancements'
git log stable..master --first-parent --format='%s %b' |
    sed -E 's/.*#([0-9]+).*\[ENH\] *(.*)/\* \2 ([#\1](\.\.\/\.\.\/pull\/\1))/' |
    { grep -E '^\*' || true; } | { grep -v -F -f "$reported" || true; }
echo
echo "##### Bugfixes"
git log stable..master --first-parent --format='%s %b' |
    sed -E 's/.*#([0-9]+).*\[FIX\] *(.*)/\* \2 ([#\1](\.\.\/\.\.\/pull\/\1))/' |
    { grep -E '^\*' || true; } | { grep -v -F -f "$reported" || true; }
