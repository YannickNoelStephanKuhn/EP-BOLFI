import sys
import os

# -- Project information -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #project-information

project = 'EP-BOLFI'
copyright = '2022-%Y, Yannick Kuhn'
author = 'Yannick Kuhn'
version = release = '${VERSION}'

# -- General configuration ---------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #general-configuration

exclude_patterns = ['*__pycache__*',]
extensions = ['myst_parser', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary']
templates_path = ['_templates']
language = 'en'
root_doc = 'index'
pygments_style = 'sphinx'
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

# -- Options for HTML output -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #options-for-html-output

html_theme = 'alabaster'

# -- Options for LaTeX output ------------------------------------------
latex_engine = "lualatex"

# -- Point to the code -------------------------------------------------

sys.path.insert(0, os.path.abspath("ep_bolfi/"))
