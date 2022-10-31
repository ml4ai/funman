from pathlib import Path
from setuptools import setup, find_packages
import os

with open(os.path.join("src", "funman_demo", '_version.py')) as version_file:
    version = version_file.readlines()[-1].split()[-1].strip("\"'")

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='funman_demo',
      version=version,
      description='Functional Model Analysis Tool - Demo Package',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='',
      author='Dan Bryce',
      author_email='dbryce@sift.net',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'':'src'},
      install_requires=[
        "funman",
        "matplotlib",
        "jupyter"
      ],
      tests_require=["unittest"],
      zip_safe=False
      )