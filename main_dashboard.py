"""Single-entry Streamlit launcher for the integrated portfolio dashboard.

This explicitly loads the tab integration before executing the existing app.py,
so Allocation Mix and Debt appear as tabs in the same dashboard.
"""

import runpy

# Import explicitly rather than relying on Python's automatic sitecustomize hook.
# That automatic hook is not reliable when Streamlit launches an app script.
import sitecustomize  # noqa: F401

runpy.run_path("app.py", run_name="__main__")
