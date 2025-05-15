"""
Note: this file can only download the data from our Kadi4Mat server if
you have access and permissions to do so. The files in question are also
publicly accessible via https://doi.org/10.5281/zenodo.15407849. Piece
them together and put them into a "data" folder within this folder.
"""

import importlib.util
import json
import numpy as np
import os
import sys

from contextlib import redirect_stdout
from io import StringIO
from kadi_apy.lib.core import KadiManager, Record

from ep_bolfi.kadi_tools.read_measurement_from_parquet import (
    read_measurement_from_parquet
)
from ep_bolfi.utility.dataset_formatting import Cycling_Information

manager = KadiManager()

try:
    os.mkdir("data")
except FileExistsError:
    ...

data = {
    type: {
        electrode: {}
        for electrode in ["negative", "positive"]
    }
    for type in [
        "protocol", "parameters", "soc", "ocv", "voltage", "overpotential"
    ]
}

for id, filename, type, electrode, direction in [
    [
        1873,
        "18650-LG-3500-MJ1-Anode GITT_ch ohne EIS(2340).json",
        "protocol",
        "negative",
        "lithiation"
    ],
    [
        1873,
        "18650-LG-3500-MJ1-Anode GITT_dch ohne EIS(2320).json",
        "protocol",
        "negative",
        "delithiation"
    ],
    [
        5729,
        "18650-LG-3500-MJ1-Cathode GITT_ch ohne EIS(2327).json",
        "protocol",
        "positive",
        "lithiation"
    ],
    [
        5729,
        "18650-LG-3500-MJ1-Cathode GITT_dch ohne EIS(2321).json",
        "protocol",
        "positive",
        "delithiation"
    ],
    [
        1873,
        "18650-LG-3500-MJ1-Anode GITT_ch ohne EIS(2340).parquet",
        "voltage",
        "negative",
        "lithiation"
    ],
    [
        1873,
        "18650-LG-3500-MJ1-Anode GITT_dch ohne EIS(2320).parquet",
        "voltage",
        "negative",
        "delithiation"
    ],
    [
        5729,
        "18650-LG-3500-MJ1-Cathode GITT_ch ohne EIS(2327).parquet",
        "voltage",
        "positive",
        "lithiation"
    ],
    [
        5729,
        "18650-LG-3500-MJ1-Cathode GITT_dch ohne EIS(2321).parquet",
        "voltage",
        "positive",
        "delithiation"
    ],
    [
        1933,
        "18650-LG-3500-MJ1-Anode GITT_ch ohne EIS(2340)_overpotential.parquet",
        "overpotential",
        "negative",
        "lithiation"
    ],
    [
        1933,
        "18650-LG-3500-MJ1-Anode GITT_dch ohne EIS(2320)_overpotential"
        ".parquet",
        "overpotential",
        "negative",
        "delithiation"
    ],
    [
        8996,
        "18650-LG-3500-MJ1-Cathode GITT_ch ohne EIS(2327)_overpotential"
        ".parquet",
        "overpotential",
        "positive",
        "lithiation"
    ],
    [
        8996,
        "18650-LG-3500-MJ1-Cathode GITT_dch ohne EIS(2321)_overpotential."
        "parquet",
        "overpotential",
        "positive",
        "delithiation"
    ],
    [
        1663,
        "lg_mj1_parameterize_negative_diffusivity_lithiation.py",
        "parameters",
        "negative",
        "lithiation",
    ],
    [
        1663,
        "lg_mj1_parameterize_negative_diffusivity_delithiation.py",
        "parameters",
        "negative",
        "delithiation",
    ],
    [
        1663,
        "lg_mj1_parameterize_positive_diffusivity_lithiation.py",
        "parameters",
        "positive",
        "lithiation",
    ],
    [
        1663,
        "lg_mj1_parameterize_positive_diffusivity_delithiation.py",
        "parameters",
        "positive",
        "delithiation",
    ],
    [
        1505,
        "18650-LG-3500-MJ1-Anode GITT_ch ohne EIS(2340)_extraction_"
        "with_negative_soc.json",
        "ocv",
        "negative",
        "lithiation"
    ],
    [
        1505,
        "18650-LG-3500-MJ1-Anode GITT_dch ohne EIS(2320)_extraction_"
        "with_negative_soc.json",
        "ocv",
        "negative",
        "delithiation"
    ],
    [
        5733,
        "18650-LG-3500-MJ1-Cathode GITT_ch ohne EIS(2327)_extraction"
        "_with_positive_soc.json",
        "ocv",
        "positive",
        "lithiation"
    ],
    [
        5733,
        "18650-LG-3500-MJ1-Cathode GITT_dch ohne EIS(2321)_extraction"
        "_with_positive_soc.json",
        "ocv",
        "positive",
        "delithiation"
    ]
]:
    filepath = "data/" + filename
    if type in ["protocol", "parameters", "ocv"]:
        record = Record(manager, id, create=False)
        record.download_file(record.get_file_id(filename), filepath)
        if type == "parameters":
            name = filename[:-3]
            spec = importlib.util.spec_from_file_location(name, filepath)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            data[type][electrode][direction] = mod.parameters
        else:
            with open(filepath, 'r') as f:
                data[type][electrode][direction] = json.load(f)
    else:
        with StringIO() as json_string, redirect_stdout(json_string):
            try:
                os.replace(filepath, filename)
            except PermissionError:
                ...
            read_measurement_from_parquet([
                '-r', id, '-n', filename, '-t', "cycling"
            ], standalone_mode=False)
            data[type][electrode][direction] = Cycling_Information.from_json(
                json_string.getvalue()
            )
            try:
                os.replace(filename, filepath)
            except PermissionError:
                ...

diffusivities = {
    electrode: {
        direction: {
            model: {}
            for model in ["SPM", "SPMe", "DFN"]
        }
        for direction in ["lithiation", "delithiation"]
    }
    for electrode in ["negative", "positive"]
}
for id, filename, electrode, direction, model in [
    [
        1660,
        "negative-electrode-diffusivity-m2s-1_lithiation_from_SPM.json",
        "negative",
        "lithiation",
        "SPM"
    ],
    [
        1660,
        "negative-electrode-diffusivity-m2s-1_lithiation_from_SPMe.json",
        "negative",
        "lithiation",
        "SPMe"
    ],
    [
        1660,
        "negative-electrode-diffusivity-m2s-1_lithiation_from_DFN.json",
        "negative",
        "lithiation",
        "DFN"
    ],
    [
        1660,
        "negative-electrode-diffusivity-m2s-1_delithiation_from_SPM.json",
        "negative",
        "delithiation",
        "SPM"
    ],
    [
        1660,
        "negative-electrode-diffusivity-m2s-1_delithiation_from_SPMe.json",
        "negative",
        "delithiation",
        "SPMe"
    ],
    [
        1660,
        "negative-electrode-diffusivity-m2s-1_delithiation_from_DFN.json",
        "negative",
        "delithiation",
        "DFN"
    ],
    [
        9001,
        "positive-electrode-diffusivity-m2s-1_lithiation_from_SPM.json",
        "positive",
        "lithiation",
        "SPM"
    ],
    [
        9001,
        "positive-electrode-diffusivity-m2s-1_lithiation_from_SPMe.json",
        "positive",
        "lithiation",
        "SPMe"
    ],
    [
        9001,
        "positive-electrode-diffusivity-m2s-1_lithiation_from_DFN.json",
        "positive",
        "lithiation",
        "DFN"
    ],
    [
        9001,
        "positive-electrode-diffusivity-m2s-1_delithiation_from_SPM.json",
        "positive",
        "delithiation",
        "SPM"
    ],
    [
        9001,
        "positive-electrode-diffusivity-m2s-1_delithiation_from_SPMe.json",
        "positive",
        "delithiation",
        "SPMe"
    ],
    [
        9001,
        "positive-electrode-diffusivity-m2s-1_delithiation_from_DFN.json",
        "positive",
        "delithiation",
        "DFN"
    ]
]:
    filepath = "data/" + filename
    record = Record(manager, id, create=False)
    record.download_file(record.get_file_id(filename), filepath)
    with open(filepath, 'r') as f:
        diffusivity_file = json.load(f)
    diffusivities[electrode][direction][model]["mode"] = np.asarray(
        diffusivity_file[
            electrode.capitalize() + " electrode diffusivity [m2.s-1]"
        ]
    )
    diffusivities[electrode][direction][model]["lower"] = np.asarray([
        eb[0] for eb in diffusivity_file[
            "error bounds of "
            + electrode.capitalize()
            + " electrode diffusivity [m2.s-1]"
        ]
    ])
    diffusivities[electrode][direction][model]["upper"] = np.asarray([
        eb[1] for eb in diffusivity_file[
            "error bounds of "
            + electrode.capitalize()
            + " electrode diffusivity [m2.s-1]"
        ]
    ])
