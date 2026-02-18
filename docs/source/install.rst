Installation
============

The package uses a standard ``src/`` layout and can be installed in editable mode for
local development.

Base install
------------

.. code-block:: bash

   python -m pip install -e .

Install with documentation extras
---------------------------------

.. code-block:: bash

   python -m pip install -e ".[docs]"

Optional extras
---------------

For Parquet support outside the default dependency set:

.. code-block:: bash

   python -m pip install -e ".[parquet]"

For development and tests:

.. code-block:: bash

   python -m pip install -e ".[dev]"
