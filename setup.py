from setuptools import setup, find_packages

def parse_requirements(filename):
    """Parse a requirements file into a list of dependencies."""
    with open(filename, 'r') as file:
        requirements = file.readlines()
    requirements = [r.strip() for r in requirements]
    requirements = [r for r in requirements if r and not r.startswith('#')]
    return requirements

requirements = parse_requirements('requirements.txt')

setup(
    name='regulations_rag',
    version='0.8.4.0',
    packages=find_packages(),  # Automatically find and include packages
    install_requires=requirements,
    include_package_data=True,  # Ensure package data files are included
    zip_safe=False  # Set to False if you are unsure about zip compatibility
)