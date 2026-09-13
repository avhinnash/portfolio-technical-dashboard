"""Inject Allocation Mix and Debt into the main Streamlit tab bar.

Python imports sitecustomize automatically at interpreter startup when it is
available on sys.path. This lets the existing app.py keep its four-tab layout
code while the rendered dashboard presents six tabs in one app.
"""

try:
    import streamlit as st
    from extra_tabs import render_allocation_tab, render_debt_tab

    _original_tabs = st.tabs
    _injected = False

    def _integrated_tabs(labels, *args, **kwargs):
        global _injected
        label_list = list(labels)
        target = ["📈 Technical Charts", "🧭 Portfolio Risk", "⚙️ Portfolio Optimizer", "🛤️ Transition Plan"]
        if not _injected and label_list == target:
            _injected = True
            all_tabs = _original_tabs(label_list + ["🥧 Allocation Mix", "💳 Debt"], *args, **kwargs)
            with all_tabs[4]:
                render_allocation_tab()
            with all_tabs[5]:
                render_debt_tab()
            return all_tabs[:4]
        return _original_tabs(labels, *args, **kwargs)

    st.tabs = _integrated_tabs
except Exception:
    # Never prevent the main dashboard from loading if optional integration fails.
    pass
