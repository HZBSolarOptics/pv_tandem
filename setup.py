#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

INSTALL_REQUIRES = ['pvlib >= 0.7.0']

TESTS_REQUIRE = ['pytest>=3',]
EXTRAS_REQUIRE = {
    'optional': ['netcdf4',],
    'doc': ['ipython', 'matplotlib', 'sphinx == 4.5.0',
            'pydata-sphinx-theme == 0.8.1', 'sphinx-gallery',
            'docutils == 0.15.2', 'seaborn'],
    'test': TESTS_REQUIRE
}
EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))



setup(
    author="Peter Tillmann",
    author_email='Peter.tillmann@helmholtz-berlin.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python toolbox for simulation and energy yield calculations of tandem solar cells.",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=TESTS_REQUIRE,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pv_tandem',
    name='pv_tandem',
    packages=find_packages(include=['pv_tandem', 'pv_tandem.*']),
    test_suite='tests',
    url='https://github.com/P-tillmann/pv_tandem',
    version='0.1.0',
    zip_safe=False,
)
