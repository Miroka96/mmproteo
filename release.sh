#!/bin/bash -e

# before that, tag your latest commit with the new version number:
# e.g.: git tag 0.4.0
# the project version should be instantly updated
# remove a tag using: git tag -d 0.4.0

python3 -m pip install --user --upgrade setuptools wheel twine
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
