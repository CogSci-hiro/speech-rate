"""Sphinx configuration for the rate documentation."""

import importlib.util
import inspect
import os
import sys
from importlib import import_module, metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
else:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_project_metadata() -> dict:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return {}
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return payload.get("project", {})


def _infer_package_name(project_name: str) -> str:
    candidate = project_name.replace("-", "_")
    if (SRC_DIR / candidate / "__init__.py").exists():
        return candidate

    if SRC_DIR.exists():
        for package_dir in sorted(SRC_DIR.iterdir()):
            if package_dir.is_dir() and (package_dir / "__init__.py").exists():
                return package_dir.name

    return candidate


PROJECT_METADATA = _load_project_metadata()
project = str(PROJECT_METADATA.get("name", "rate"))
PACKAGE_NAME = _infer_package_name(project)

author_entries = PROJECT_METADATA.get("authors", [])
if isinstance(author_entries, list):
    author_names = [entry.get("name", "") for entry in author_entries if isinstance(entry, dict)]
    author_names = [name for name in author_names if name]
else:
    author_names = []
author = ", ".join(author_names) if author_names else f"{project} developers"

try:
    release = str(PROJECT_METADATA.get("version") or metadata.version(PACKAGE_NAME))
except metadata.PackageNotFoundError:
    release = str(PROJECT_METADATA.get("version", "0+unknown"))
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
]

OPTIONAL_EXTENSIONS = {
    "numpydoc": "numpydoc",
    "sphinx_gallery.gen_gallery": "sphinx_gallery",
    "sphinxcontrib.bibtex": "sphinxcontrib.bibtex",
    "sphinx_copybutton": "sphinx_copybutton",
    "sphinx_design": "sphinx_design",
}
for extension_name, module_name in OPTIONAL_EXTENSIONS.items():
    if importlib.util.find_spec(module_name) is not None:
        extensions.append(extension_name)

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_inherit_docstrings = True
numpydoc_show_class_members = False
default_role = "py:obj"

html_theme = "pydata_sphinx_theme" if importlib.util.find_spec("pydata_sphinx_theme") else "alabaster"
html_title = f"{project} documentation"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

if "sphinx_gallery.gen_gallery" in extensions:
    from sphinx_gallery.sorting import FileNameSortKey

    sphinx_gallery_conf = {
        "examples_dirs": ["auto_examples"],
        "gallery_dirs": ["auto_examples"],
        "filename_pattern": r"plot_",
        "within_subsection_order": FileNameSortKey,
        "download_all_examples": False,
    }

if "sphinxcontrib.bibtex" in extensions:
    bibtex_bibfiles = ["references.bib"]

todo_include_todos = False
if "sphinx_copybutton" in extensions:
    copybutton_prompt_text = r">>> |\.\.\. |\$ "
    copybutton_prompt_is_regexp = True

_GITHUB_DOCS_URL = os.environ.get("DOCS_GITHUB_URL", "").strip()


def linkcode_resolve(domain: str, info: dict) -> str | None:
    """Resolve links for documented Python objects."""
    if domain != "py" or not _GITHUB_DOCS_URL:
        return None

    module_name = info.get("module")
    if not module_name:
        return None

    try:
        module = import_module(module_name)
    except Exception:
        return None

    obj = module
    fullname = info.get("fullname", "")
    for attr in fullname.split("."):
        if not attr:
            continue
        obj = getattr(obj, attr, None)
        if obj is None:
            return None

    try:
        source_file = inspect.getsourcefile(obj)
        source_lines, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    if not source_file:
        return None

    try:
        relative_path = Path(source_file).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    end_lineno = lineno + len(source_lines) - 1
    base = _GITHUB_DOCS_URL.rstrip("/")
    return f"{base}/{relative_path.as_posix()}#L{lineno}-L{end_lineno}"
