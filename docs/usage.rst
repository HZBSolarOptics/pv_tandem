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

Basic Usage
___________

The basic requirement to simulate any solar cell with pv_tandem is to define the one-diode model with corespodning parameters, the absorbed photocurrent (Jsc) and the cell temperture:

.. plot:: pyplots/one-diode.py
   :include-source:

You can find many examples in the :ref:`gallery` that show how to simulate different types of solar cells and tandem with bifacial or monofacial illumination.