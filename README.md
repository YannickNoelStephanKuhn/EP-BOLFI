# EP-BOLFI

EP-BOLFI (Expectation Propagation with Bayesian Optimization for Likelihood-Free Inference) extends the simulator optimizer BOLFI with the data featurization of Expectation Propagation. EP-BOLFI inherits the advantages of both: high stability to measurement noise and considerable reduction of computational effort. The performance is one to two orders of magnitude better than Markov-Chain Monte Carlo, counted in the number of simulator evaluations required.

## Documentation

[ep_bolfi](ep_bolfi/) contains the EP-BOLFI algorithm, a few example models, and utility functions that cover fitting functions, datafile imports, dataset processing, and visualization. For further details, please refer to [documentation](documentation/).

## Examples

The examples currently comprise of the code used in the [EP-BOLFI publication](https://doi.org/10.48550/arXiv.2208.03289). Apart from the SPMe benchmark, they analyze a GITT dataset provided by BASF. You can find the dataset at the DOI [10.5281/zenodo.7478267](https://doi.org/10.5281/zenodo.7478267). If you wish to re-run the parameterization yourself, copy the contents of GITT_data_and_parameterization_info.zip into the top folder first.
 - To perform the SPMe benchmarks from Aitio et al., please use spme_benchmark_multimodal.py and spme_benchmark_unimodal.py. To calculate the tabulated results presented in the article, use evaluate_spme_benchmark_confidence.py.
 - To preprocess the OCV curves, please use ocv_from_cccv_and_gitt.py.
 - To view the GITT data, please use measurement_plot.py.
 - To estimate parameters from the GITT data from BASF, use run_estimation.py, and after that collect_gitt_estimation_results.py. This may take more than a week to run.
 - To get the tabulated results for the GITT parameterization, please use evaluate_gitt_estimation_confidence.py.
 - To perform the verification of the GITT data parameterization, please use verify_estimation.py. This may take more than a week to run.
 - To analyze the correlation matrix for the 7-parameter estimation, please use correlation_visualization.py.
 - To plot the results from the GITT estimation procedure, please use gitt_visualization.py.
 - To analyze the precision and reliability of the GITT estimation procedure, please use calculate_experimental_and_simulated_features.py. To plot this analysis, please use analytic_vs_epbolfi_results.py and sensitivity_visualization.py.
 - To plot the joint resistance of the two exchange-current densities, please use joint_resistance.py.

## Using EP-BOLFI to process your measurements with your model

Please have a look at [the setup example](Python/parameters/estimation/). [gitt_basf.py](Python/parameters/estimation/gitt_basf.py) contains the preprocessing of the GITT dataset and a GITT simulator. [gitt.py](Python/parameters/estimation/gitt.py) contains one possible definition of features in a GITT measurement. For performing the optimization, please have a look at run_estimation.py. If you wish to re-use an optimization result as starting value, use .Q, .r, .Q_features and .r_features and pass them to the initialization of the EP-BOLFI object.

## Installation

EP-BOLFI requires [Python 3.9](https://www.python.org/downloads/release/python-3913/). Just download the newest Release .whl file and [requirements.txt](requirements.txt). Install the dependencies with [requirements.txt](requirements.txt) via [pip](https://pypi.org/project/pip/), then install the .whl file. In case you want to build the package from source, please refer to [CONTRIBUTING.md](CONTRIBUTING.md#building-from-source).

### Using pip

Create a virtual environment and activate it. On Linux and Mac:
```bash
python3.9 -m venv ep_bolfi
source ep_bolfi/bin/activate
```
On Windows:
```powershell
py -3.9 -m venv ep_bolfi
. .\ep_bolfi\Scripts\activate
```

Then install the dependencies and package:
```bash
pip install -r requirements.txt
pip install ep_bolfi-${VERSION}-py3-none-any.whl
```

## Contributing to EP-BOLFI

Please refer to the [contributing guidelines](CONTRIBUTING.md) and adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

GPLv3, see LICENSE file.
