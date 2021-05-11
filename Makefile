# https://gist.github.com/prwhite/8168133

.PHONY: help test install-requirements dev-install pip-install release build-docs

help:                                 ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

test:                                 ## run tests
	cd tests && pytest

install-requirements:                 ## install requirements
	pip3 install -r requirements.txt

dev-install: install-requirements     ## install mmproteo in development mode
	pip3 install -e .

pip-install:                          ## install mmproteo from PyPi
	pip3 install mmproteo

release: clean                        ## build and release mmproteo
	@! (mmproteo --version | egrep "dirty|\+" && echo "current version is dirty or untagged")
	pip3 install --upgrade setuptools wheel twine
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*

build-docs:                           ## build the docs
	cd docs && sphinx-apidoc -f --separate -o src/ ../src/
	make -C docs html

clean:                                ## clean up
	-rm -r docs/_build
	-rm -r docs/src
	-rm -r dist
	-rm -r build
