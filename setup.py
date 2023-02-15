from setuptools import setup

setup(
    name='pv_tandem',
    version='0.1.0',
    author='Peter Tillmann',
    author_email='Peter.tillmann@helmholtz-berlin.de',
    packages=['pv_tandem'],
    url='https://github.com/nano-sippe/pv_tandem',
    license='LICENSE.txt',
    description='Package for tandem solar cell simulations',
    long_description=open('README.md').read(),
    install_requires=[
    "numpy",
    "pandas",
    "pvlib"
    ],
)
