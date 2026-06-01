PYTHON ?= python

.PHONY: syntax lint format clean

syntax:
	$(PYTHON) tests/check_syntax.py

lint:
	$(PYTHON) -m flake8 apcyc api data evaluation models router scripts trainer utils *.py

format:
	$(PYTHON) -m isort apcyc api data evaluation models router scripts trainer utils *.py
	$(PYTHON) -m yapf -ir apcyc api data evaluation models router scripts trainer utils *.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name '.DS_Store' -delete
