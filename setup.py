from setuptools import setup, find_packages

setup(name='funman',
      version='0.1',
      description='Functional Model Analysis Tool',
      url='',
      author='Dan Bryce',
      author_email='dbryce@sift.net',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'':'src'},
      install_requires=[
        "gromet2smtlib",
        # TODO imported these do keep things turning over in reorg.
        #      Come back to determine if they can be factored out.
        "numpy",
        "matplotlib",
        "jupyter"
      ],
      tests_require=["unittest"],
      zip_safe=False
      )