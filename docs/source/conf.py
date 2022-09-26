# -*- coding: utf-8 -*-
#
# landlab documentation build configuration file, created by
# sphinx-quickstart on Tue Apr 23 17:33:31 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import sys
import tomli
import pathlib
from datetime import date

import landlab

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                                 os.pardir)))
sys.path.insert(0, os.path.abspath("../.."))
docs_dir = os.path.dirname(__file__)


# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_inline_tabs",
    "sphinxcontrib.towncrier",
    "sphinx_jinja",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# Regex for links that we know work in browser, but do not work in sphinx/CI (BE VERY CAREFUL ADDING LINKS TO THIS LIST)
if os.getenv("GITHUB_ACTIONS"):
    linkcheck_ignore = [
        r"https://pubs.geoscienceworld.org/gsa/geology.*",  # Added by KRB Dec 2019, at this point two links match this pattern
        r"https://doi.org/10.1130/*",  # Added by KRB Jan 2019. Four links match this pattern
        r"https://dx.doi.org/10.1029/2011jf002181",  # Added by EWHH April 2020
        r"https://doi.org/10.1029/2019JB018596",  # Added by EWHH April 2020
        r"https://doi.org/10.3133/pp294B",  # Added by EWHH September 2021
        #     r"https://yaml.org/start.html",  # Added by EWHH September 2021
    ]
    linkcheck_retries = 5

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "landlab"
copyright = str(date.today().year) + ", The Landlab Team"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = landlab.__version__
# The full version, including alpha/beta/rc tags.
release = landlab.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The reST default role (used for this markup: `text`) to use for all documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True
add_module_names = False

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# selects what content will be inserted into the main body of an autoclass directive: 'class'(default), 'both', or 'init'
autoclass_content = "both"


# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = "alabaster"
html_theme = "furo"
html_title = "landlab"
html_logo = "_static/landlab_logo.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "announcement": "<em>Landlab 2.5 released!</em>",
    "source_repository": "https://github.com/landlab/landlab/",
    "source_branch": "master",
    "source_directory": "docs/source",
    "sidebar_hide_name": True,
    "footer_icons": [
        {
            "name": "power",
            "url": "https://csdms.colorado.edu",
            "html": """
               <svg stroke="currentColor" fill="currentColor" stroke-width="0" version="1.1" viewBox="0 0 16 16" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M6 0l-6 8h6l-4 8 14-10h-8l6-6z"></path></svg>
               <b><i>Powered by CSDMS</i></b>
            """,
            "class": "",
        },
    ],
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "images/landlab_logo.jpg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}


# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "landlabdoc"


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    #  The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',
    #  The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',
    #  Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", "landlab.tex", "landlab Documentation", "Author", "manual"),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", "landlab", "landlab Documentation", ["Author"], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "landlab",
        "landlab Documentation",
        "Author",
        "landlab",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False


# -- Options for Epub output ---------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = "landlab"
epub_author = "Author"
epub_publisher = "Author"
epub_copyright = "2013, Author"

# The language of the text. It defaults to the language option
# or en if the language is not set.
# epub_language = ''

# The scheme of the identifier. Typical schemes are ISBN or URL.
# epub_scheme = ''

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
# epub_identifier = ''

# A unique identification for the text.
# epub_uid = ''

# A tuple containing the cover image and cover page html template filenames.
# epub_cover = ()

# A sequence of (type, uri, title) tuples for the guide element of content.opf.
# epub_guide = ()

# HTML files that should be inserted before the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_pre_files = []

# HTML files shat should be inserted after the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_post_files = []

# A list of files that should not be packed into the epub file.
# epub_exclude_files = []

# The depth of the table of contents in toc.ncx.
# epub_tocdepth = 3

# Allow duplicate toc entries.
# epub_tocdup = True

# Fix unsupported image types using the PIL.
# epub_fix_images = False

# Scale large images.
# epub_max_image_width = 0

# If 'no', URL addresses will not be shown.
# epub_show_urls = 'inline'

# If false, no index is generated.
# epub_use_index = True

todo_include_todos = True
# latex_elements = dict(preamble='\\usepackage{amsmath}')

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True

# -- Options for towncrier_draft extension --------------------------------------------

towncrier_draft_autoversion_mode = "draft"  # or: 'sphinx-release', 'sphinx-version'
towncrier_draft_include_empty = True
# towncrier_draft_working_directory = pathlib.Path(docs_dir).parent.parent
towncrier_draft_working_directory = pathlib.Path(docs_dir).parent / "towncrier"


with open("../index.toml", "rb") as fp:
    cats = tomli.load(fp)
cats["grids"].pop("ModelGrid")

jinja_contexts = {"llcats": cats}
