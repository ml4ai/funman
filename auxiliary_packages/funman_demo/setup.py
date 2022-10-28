from setuptools import setup, find_packages
import os

with open(os.path.join("src", "funman_demo", '_version.py')) as version_file:
    version = version_file.readlines()[-1].split()[-1].strip("\"'")

setup(name='funman_demo',
      version=version,
      description='Functional Model Analysis Tool - Demo Tooling',
      url='',
      author='Dan Bryce',
      author_email='dbryce@sift.net',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'':'src'},
      install_requires=[
        "funman",
        "matplotlib"
      ],
      tests_require=["unittest"],
      zip_safe=False
      )