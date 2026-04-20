#################### PACKAGE ACTIONS ###################

install_package:
	@pip uninstall -y my_shelves || :
	@pip install -e .

pylint:
	find . -iname "*.py" -not -path "./tests/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes
