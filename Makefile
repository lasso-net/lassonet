docs:
	cd sphinx_docs && $(MAKE) html
	- rm -rf docs/api
	mkdir docs/api
	cp -r sphinx_docs/_build/html/* docs/api

.PHONY: docs