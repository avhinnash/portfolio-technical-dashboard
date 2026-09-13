"""Single-entry Streamlit launcher for the integrated portfolio dashboard.

This launcher patches Streamlit's tab creation directly before executing app.py,
so Allocation Mix and Debt always appear inside the same dashboard.
"""

import runpy

import streamlit as st
from extra_tabs import render_allocation_tab, render_debt_tab

_original_tabs = st.tabs
_integrated_once = False


def _integrated_tabs(labels, *args, **kwargs):
    global _integrated_once
    label_list = list(labels)
    target = [
        "📈 Technical Charts",
        "🧭 Portfolio Risk",
        "⚙️ Portfolio Optimizer",
        "🛤️ Transition Plan",
    ]

    if not _integrated_once and label_list == target:
        _integrated_once = True
        tabs = _original_tabs(
            target + ["🥧 Allocation Mix", "💳 Debt"],
            *args,
            **kwargs,
        )

        # Render the two added sections directly into their tab containers.
        with tabs[4]:
            render_allocation_tab()
        with tabs[5]:
            render_debt_tab()

        # app.py expects four return values, so hand back the original four tabs.
        return tabs[:4]

    return _original_tabs(labels, *args, **kwargs)


st.tabs = _integrated_tabs

runpy.run_path("app.py", run_name="__main__")
