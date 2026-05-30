"""
Python model 'sales_growth.py'
Translated using PySD
"""

from pathlib import Path

from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 24,
    "time_step": lambda: 1,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="INITIAL TIME", units="month", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="FINAL TIME", units="month", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="TIME STEP", units="month", comp_type="Constant", comp_subtype="Normal"
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


@component.add(
    name="SAVEPER",
    units="month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The save time step for the simulation.
    """
    return __data["time"].saveper()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(name="ad_spend", comp_type="Constant", comp_subtype="Normal")
def ad_spend():
    return 100


@component.add(name="conversion", comp_type="Constant", comp_subtype="Normal")
def conversion():
    return 0.5


@component.add(name="churn_rate", comp_type="Constant", comp_subtype="Normal")
def churn_rate():
    return 0.05


@component.add(
    name="acquisition",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"ad_spend": 1, "conversion": 1},
)
def acquisition():
    return ad_spend() * conversion()


@component.add(
    name="churn_flow",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"sales": 1, "churn_rate": 1},
)
def churn_flow():
    return sales() * churn_rate()


@component.add(
    name="Sales",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_sales": 1},
    other_deps={
        "_integ_sales": {"initial": {}, "step": {"acquisition": 1, "churn_flow": 1}}
    },
)
def sales():
    return _integ_sales()


_integ_sales = Integ(lambda: acquisition() - churn_flow(), lambda: 1000, "_integ_sales")
