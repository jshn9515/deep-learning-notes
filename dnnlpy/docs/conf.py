"""Sphinx configuration for the dnnlpy API reference."""

import tomllib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

project = 'dnnlpy'
author = 'Yunjie Lin'

with (PACKAGE_ROOT / 'pyproject.toml').open('rb') as project_file:
    release = tomllib.load(project_file)['project']['version']

extensions = [
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]
exclude_patterns = [
    '_build',
    '.DS_Store',
    '.venv',
]

# Sphinx
add_module_names = False
python_use_unqualified_type_names = True

# AutoAPI
autoapi_dirs = ['../src']
autoapi_root = 'reference'
autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_member_order = 'bysource'
autoapi_python_class_content = 'class'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]

# Docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# HTML
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']


def skip_attributes(app, what, name, obj, skip, options):
    if what == 'attribute':
        return True
    return skip


def setup(app):
    app.connect('autoapi-skip-member', skip_attributes)
