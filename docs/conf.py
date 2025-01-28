# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
from subprocess import run as subprun

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ARIS_lite'
copyright = '2024, Jan Haacker'
author = 'Jan Haacker'
version = subprun("git tag --sort=taggerdate | grep -e \"^v\" | head -1", shell=True, check=True, capture_output=True).stdout.decode().strip()
print(version)
git_hash = subprun(f"git rev-list -n 1 {version}", shell=True, check=True, capture_output=True).stdout.decode().strip()
print(git_hash)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.githubpages',
              'myst_parser',
              'sphinx.ext.linkcode',
              # 'sphinx.ext.napoleon',
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

sys.path.insert(0, str(Path('..').resolve()))
sys.path.insert(0, str(Path('..', '..').resolve()))


def get_func_line_number(module, func_name) -> int:
    with open(str(Path('..', module+".py").resolve())) as f:
        for num, line in enumerate(f, 1):
            if f"def {func_name}" in line:
                return num


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = "/".join(info['module'].split(".")[1:])
    if "fullname" in info:
        anchor = "#L%d" % get_func_line_number(filename, info["fullname"])
    else:
        anchor = ""
    return "https://github.com/j-haacker/ARIS_lite/blob/%s/%s.py%s" % (git_hash, filename, anchor)
