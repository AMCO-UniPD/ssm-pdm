"""
Python module with some configuration variables
"""

import os
import sys
ceruleo_path = os.path.join(os.path.dirname(__file__), "ceruleo")
sys.path.append(ceruleo_path)
from ceruleo.dataset.catalog.PHMDataset2018 import FailureType
from pathlib import Path

PHM_PATH_ACQ4 = Path(Path.home() / "datasets")
PHM_PATH = Path("/mnt/disk1/davide_frizzo/datasets")

CMAPSS_MODELS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004",
]

PHM_FEATURES = [
    "IONGAUGEPRESSURE",
    "ETCHBEAMVOLTAGE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE",
    "ROTATIONSPEED",
    "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION",
    "ETCHSOURCEUSAGE",
    "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
]

#NOTE: These names are without the _ because are needed to
# be passed as arguments to PHMDataset2018 which accepts only
# names without the underscore

PHM_TOOLS = [
    "01M01",
    "01M02",
    "02M01",
    "02M02",
    "03M01",
    "03M02",
    "04M01",
    "04M02",
    "05M01",
    "05M02",
    "06M01",
    "06M02",
    "07M01",
    "07M02",
    "08M01",
    "08M02",
    "09M01",
    "09M02",
    "10M01",
    "10M02",
]

PHM_TEST_TOOLS = [
    "01M02",
    "02M02",
    "03M01",
    "04M01",
    "06M01",
]

PHM_FAILURES = {
    "flow_low": FailureType.FlowCoolPressureDroppedBelowLimit,
    "flow_high_pump": FailureType.FlowcoolPressureTooHighCheckFlowcoolPump,
    "flow_leak": FailureType.FlowcoolLeak,
    "flow_high_pump_no_wafer": FailureType.FlowcoolPressureTooHighCheckFlowcoolPumpNoWaferID
}

PHM_FAIL_TYPES = list(PHM_FAILURES.keys())

APPROACHES = ["padding", "windowed"]
