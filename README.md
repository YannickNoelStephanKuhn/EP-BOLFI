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

## Using the Kadi4Mat tools

The files in (ep_bolfi/kadi_tools) are command-line tools. These command-line tools are designed to interface with the database software [Kadi4Mat](https://kadi.iam-cms.kit.edu/), of which we have an internal instance running at (https://kadi-dlr.hiu-batteries.de/).

In order to use these command-line tools in the workflow toolchain of Kadi4Mat, they are implemented with the library [xmlhelpy](https://gitlab.com/iam-cms/workflows/xmlhelpy) which extends the [Click](https://github.com/pallets/click) library for command-line tools with the option to generate machine-readable representations of command-line tools. You can find these representations in `ep_bolfi/kadi_tools/xml_representations`, but only when installing a Release. If they are missing in your installation, please refer to the manual instructions in [CONTRIBUTING.md](CONTRIBUTING.md).

[KadiStudio](https://bwsyncandshare.kit.edu/s/cJSZrE6fDTR6cLQ) can import the .py files, or on a Kadi4Mat instance with the online workflow editor enabled, the .xml files can be imported by uploading them to any Record on the respective Kadi4Mat instance. The command-line tools are then available as building blocks for workflows.

For executing workflows that contain one of these tools, the command `python` has to launch a Python environment with this library installed. Either bake the activation of said environment into the convenience scripts in the following section, or make your command line automatically activate it by following these steps.

On Linux:
```bash
nano ~/.profile
```
Then add the following line to the bottom of the file:
```
source ~/ep_bolfi/bin/activate
```

On Windows:
```powershell
notepad $((Split-Path $profile -Parent) + "\profile.ps1")
```
Then add the following line to the bottom of the text file:
```
. .\ep_bolfi\Scripts\activate
```

In the case where spurious lines like "warning in ...: failed to import cython module: falling back to numpy" show up and break workflow scripts, these are due to an unfortunate design decision in GPy. You need to install GPy from source like so to improve performance as well:

```bash
git clone https://github.com/SheffieldML/GPy.git
cd GPy
pip install .
```

## Installation

EP-BOLFI requires [Python 3.9](https://www.python.org/downloads/release/python-3913/). Then, install EP-BOLFI and its dependencies via pip:
```bash
pip install ep-bolfi
```
In case you want to build the package from source, please refer to [CONTRIBUTING.md](CONTRIBUTING.md#building-from-source).

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
pip install ep_bolfi-3.0py3-none-any.whl
```

## Contributing to EP-BOLFI

Please refer to the [contributing guidelines](CONTRIBUTING.md) and adhere to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

GPLv3, see LICENSE file.
