from setuptools import setup, find_packages

def parse_requirements(filename):
    """Parse a requirements file into a list of dependencies."""
    with open(filename, 'r') as file:
        requirements = file.readlines()
    requirements = [r.strip() for r in requirements]
    # Exclude specific lines such as comments or empty lines
    requirements = [r for r in requirements if r and not r.startswith('#')]
    # Here you can also add logic to handle package versioning more gracefully
    return requirements

# Usage
requirements = parse_requirements('requirements.txt')

setup(
    name='regulations_rag',
    version='0.6.1.9',
    install_requires=requirements,
)