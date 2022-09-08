from setuptools import setup, find_packages
import sys
import os

setup(name='funman',
      version='0.1',
      description='Functional Model Analysis Tool',
      url='',
      author='Dan Bryce',
      author_email='dbryce@sift.net',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'':'src'},
      install_requires=["gromet2smtlib"],
      tests_require=["unittest"],
      zip_safe=False
      )