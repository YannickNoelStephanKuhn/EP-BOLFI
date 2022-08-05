# EP-BOLFI

EP-BOLFI (Expectation Propagation with Bayesian Optimization for Likelihood-Free Inference) extends the simulator optimizer BOLFI with the data featurization of Expectation Propagation. EP-BOLFI inherits the advantages of both: high stability to measurement noise and considerable reduction of computational effort. The performance is one to two orders of magnitude better than Markov-Chain Monte Carlo, counted in the number of simulator evaluations required.

This is the accompanying repository to the article "EP-BOLFI: Measurement-Noise-Aware Parameterization of Continuum Battery Models from Electrochemical Measurements Applied to Full-Cell GITT Measurements", where EP-BOLFI is used to parameterize GITT measurements of a full battery cell.

## Reproducing the data analysis in the article

First, please download the SPMe benchmark results, GITT data and GITT parameterization results and put them into the root folder of your download of this repository.
 - To perform the SPMe benchmarks from Aitio et al., please use spme_benchmark.py. To calculate the tabulated results presented in the article, use evaluate_spme_benchmark_confidence.py.
 - To preprocess the OCV curves, please use ocv_from_cccv_and_gitt.py.
 - To view the GITT data, please use measurement_plot.py.
 - To estimate parameters from the GITT data from BASF, use run_estimation.py, and after that collect_gitt_estimation_results.py. This may take more than a week to run.
 - To get the tabulated results for the GITT parameterization, please use evaluate_gitt_estimation_confidence.py.
 - To perform the verification of the GITT data parameterization, please use verify_estimation.py. This may take more than a week to run.
 - To analyze the correlation matrix for the 7-parameter estimation, please use correlation_visualization.py.
 - To plot the results from the GITT estimation procedure, please use gitt_visualization.py.
 - To analyze the precision and reliability of the GITT estimation procedure, please use calculate_experimental_and_simulated_features.py. To plot this analysis, please use analytic_vs_epbolfi_results.py and sensitivity_visualization.py.
 - To plot the joint resistance of the two exchange-current densities, please use joint_resistance.py.

## Using EP-BOLFI to process your GITT measurements with your model

Please have a look at parameters/estimation/gitt_basf.py, which contains all the code that is necessary to preprocess your simulator and data for use with EP-BOLFI. Both experimental and simulated output have to have the following shape: a list of lists, where each list refers to a charge pulse or a rest phase, starting with a charge pulse. Always close with a rest phase. For performing the optimization, please have a look at run_estimation.py. The variable names and procedures are explained in the article. If you wish to re-use an optimization result as starting value, use .Q, .r, .Q_features and .r_features and pass them to the keyword arguments Q, r, Q_features and r_features of perform_gitt_estimation.

## Installation

EP-BOLFI requires Python 3.9(.2). Just copy the repository to a location on your computer and install the dependencies either via [pip](https://pypi.org/project/pip/) or [conda](https://anaconda.org/). We recommend using conda on Windows systems.

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda install -c conda-forge --file requirements.txt
```

## Citing EP-BOLFI

If you use EP-BOLFI in your work, please cite our paper

> Kuhn, Y., Wolf, H., Horstmann, B., & Latz, A. (2022). EP-BOLFI: Measurement-Noise-Aware Parameterization of Continuum Battery Models from Electrochemical Measurements Applied to Full-Cell GITT Measurements. _arXiv_.

You can use the bibtex

```
@article{Kuhn2022,
  archivePrefix = {arXiv},
  arxivId = {},
  author = {Kuhn, Yannick and Wolf, Hannes and Horstmann, Birger and Latz, Arnulf},
  pages = {1--18},
  title = {{EP-BOLFI: Measurement-Noise-Aware Parameterization of Continuum Battery Models from Electrochemical Measurements Applied to Full-Cell GITT Measurements}},
  year = {2022},
}
```

We would be grateful if you could also cite [ELFI](https://github.com/elfi-dev/elfi), which EP-BOLFI relies upon for BOLFI.

## Contributing to EP-BOLFI

We will open up EP-BOLFI to contributions at a later date. Please use the "Watch" feature on the [GitHub repository](https://github.com/YannickNoelStephanKuhn/EP-BOLFI) to get notified when EP-BOLFI goes Open Source.

## Licence

Shield: [![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg
