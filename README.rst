.. image:: logo.png
  :width: 600
  :alt: logo

.. image:: https://img.shields.io/pypi/v/pv_tandem.svg
        :target: https://pypi.python.org/pypi/pv_tandem

.. image:: https://img.shields.io/github/actions/workflow/status/nano-sippe/pv_tandem/pytest.yml
        :target: https://github.com/nano-sippe/pv_tandem/actions/workflows/pytest.yml/badge.svg

.. image:: https://readthedocs.org/projects/pv-tandem/badge/?version=latest
        :target: https://pv-tandem.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




pv_tandem is a python toolbox for simulation and energy yield calculations of single junction and tandem solar cells with classical (monofacial) or bifacial illumination.

* Free software: MIT license
* Documentation: https://pv-tandem.readthedocs.io.

Features
--------

* Helping tools to process irradiance data for solar cell simulations
* Electrical modeling with 1-diode model for single junction or tandem solar cells
* Modeling of illumination in a large PV power plant for the front- and backside of the modules

Where to get it
---------------

pv_tandem can easily be installed with pip:

.. code-block:: bash

    pip install pv_tandem

If you want to be able to directly run the examples you can install pv_tandems together with all nessesary libraries with

.. code-block:: bash

    pip install pv_tandem[doc]

The source code is hosted at GitHub at: https://github.com/HZBSolarOptics/pv_tandem

If you want to install from source clone the project from github:

.. code-block:: bash

    git clone https://github.com/nano-sippe/pv_tandem

Then change into the pv_tandem folder and install with pip

.. code-block:: bash

    pip install .

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
