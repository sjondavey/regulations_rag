from setuptools import setup, find_packages

setup(
    name='regulations_rag',
    version='0.1',
    package_dir={'': 'src'},  # Specifies that packages are under src directory
    packages=find_packages(where='src'),  # Tells setuptools to find packages under src
)