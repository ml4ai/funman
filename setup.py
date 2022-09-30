from setuptools import setup, find_packages
import os

with open(os.path.join("src", "funman", '_version.py')) as version_file:
    version = version_file.readlines()[-1].split()[-1].strip("\"'")

setup(name='funman',
      version=version,
      description='Functional Model Analysis Tool',
      url='',
      author='Dan Bryce',
      author_email='dbryce@sift.net',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'':'src'},
      install_requires=[
        "model2smtlib",
        # TODO imported these do keep things turning over in reorg.
        #      Come back to determine if they can be factored out.
        "numpy",
        "matplotlib",
        "jupyter"
      ],
      tests_require=["unittest"],
      zip_safe=False
      )