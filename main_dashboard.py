"""Single-entry Streamlit launcher for the integrated portfolio dashboard.

This avoids monkey-patching Streamlit. It executes the existing app source after
replacing its four-tab declaration with a six-tab declaration, then renders the
Allocation Mix and Debt content into tabs 5 and 6.
"""

from pathlib import Path

# Read the existing dashboard source exactly as maintained in app.py.
source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")

old_tabs = 'tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical Charts", "🧭 Portfolio Risk", "⚙️ Portfolio Optimizer", "🛤️ Transition Plan"])'
new_tabs = 'tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Technical Charts", "🧭 Portfolio Risk", "⚙️ Portfolio Optimizer", "🛤️ Transition Plan", "🥧 Allocation Mix", "💳 Debt"])'

if old_tabs not in source:
    raise RuntimeError("Could not find the expected four-tab declaration in app.py. The dashboard layout may have changed.")

source = source.replace(old_tabs, new_tabs, 1)
source += '''\n\n# Integrated tabs added by main_dashboard.py\nfrom extra_tabs import render_allocation_tab, render_debt_tab\n\nwith tab5:\n    render_allocation_tab()\n\nwith tab6:\n    render_debt_tab()\n'''

# Execute as one Streamlit script so all six tabs share the same app/session.
exec(compile(source, str(Path(__file__).with_name("app.py")), "exec"), globals(), globals())
