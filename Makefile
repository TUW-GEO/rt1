#######################################
# This file is part of RT1
#######################################

# XXX: machine specific paths
# pep8, ignoring some errors like e.g. indention errors or linelength error
PEP = pep8 --ignore=E501
TDIR = ./tmp
VERSION = 0.1-dev
TESTDIRS = tests


clean :
	find . -name "*.pyc" -exec rm -rf {} \;
	find . -name "*.so" -exec rm -rf {} \;
	find . -name "data_warnings.log" -exec rm -rf {} \;
	rm -rf build
	rm -rf MANIFEST
	rm -rf cover
	rm -rf tmp
	rm -rf dist
	rm -rf geoval.egg-info


dist : clean
	python setup.py sdist

#upload_docs:
#	python setup.py upload_sphinx

upload_pip: dist
	# ensure that pip version has always counterpart on github
	git push origin master
	# note that this requres .pypirc file beeing in home directory
	python setup.py sdist 
	twine upload dist/*




