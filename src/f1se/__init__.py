"""F1 Strategy Engine (f1se).

A pit-strategy recommendation engine: given a race situation, recommend when to
pit and which tyre compounds to fit, with quantified uncertainty.

The package is decoupled by design — modelling logic lives in plain functions
under ``models`` / ``sim``; ``api`` and the Streamlit app are thin layers on top.
"""

__version__ = "0.1.0"
