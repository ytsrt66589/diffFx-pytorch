import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'diffFx-pytorch'
copyright = '2025, Yen-Tung (Arthur) Yeh'
author = 'Yen-Tung (Arthur) Yeh'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Automatically include docstrings
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx_autodoc_typehints' # Better type hint support
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Templates and patterns
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# Change from alabaster to Read the Docs theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

# autodoc configurations
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Type hints configurations
set_type_checking_flag = True
typehints_fully_qualified = False
always_document_param_types = True