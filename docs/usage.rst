===============
Getting started
===============

To get started with pv_tandem you will need two (three) things:

#. Install pv_tandem into your python enviroment.
#. Define the one-diode parameters.
#. (Optinal) Irradiance data for a specific location.

The third step can be omitted if the specific device should only be modeled for standart test conditions (STC). To learn more about the modeling approach of pv_tandem see the :doc:`modeling basics <model_basics>` section


Installation
____________

To install pv_tandem, run this command in your terminal:

.. code-block:: console

    $ pip install pv_tandem


To use pv_tandem in a project::

    import pv_tandem
