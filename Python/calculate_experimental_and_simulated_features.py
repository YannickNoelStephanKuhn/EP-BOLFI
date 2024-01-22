# Copyright (c): German Aerospace Center (DLR)
import json
from runpy import run_module

from parameters.estimation.gitt import (
    calculate_experimental_and_simulated_features
)

with open('../GITT estimation results/estimation_results.json', 'r') as f:
    estimation_results = json.load(f)

simulators = []
experimental_datasets = []
parameter_sets = []

for i in range(len(estimation_results)):
    data = run_module(
        'parameters.estimation.gitt_basf',
        init_globals={
            'optimize_simulation_speed': True,
            'soc_dependent_estimation': True,
            'white_noise': False,
            'parameter_noise': False,
            'pulse_number': i,
        }
    )
    simulators.append(data['simulator'])
    experimental_datasets.append(data['experimental_dataset'])
    parameter_sets.append(data['parameters'])

evaluations, sensitivities, experimental_features = (
    calculate_experimental_and_simulated_features(
        simulators,
        experimental_datasets,
        parameter_sets,
        estimation_results,
        data['free_parameters_names'],
        data['transform_parameters'],
        bounds_in_standard_deviations=1
    )
)

print("evaluations:")
print(evaluations)
print("sensitivities:")
print(sensitivities)
with open("../GITT estimation results/simulated_features_at_each_pulse.json",
          "w") as f:
    json.dump(evaluations, f)
with open("../GITT estimation results/boundaries_of_simulated_features.json",
          "w") as f:
    json.dump(sensitivities, f)

print("experimental_features:")
print(experimental_features)
with open("../GITT estimation results/"
          "experimental_features_at_each_pulse.json",
          "w") as f:
    json.dump(experimental_features, f)
