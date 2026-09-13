"""Single Streamlit entrypoint for the portfolio dashboard.

The core dashboard and the Allocation/Debt sections render into one six-tab app.
Run with: python -m streamlit run app.py
"""

from pathlib import Path

source_path = Path(__file__).with_name("portfolio_core.py")
source = source_path.read_text(encoding="utf-8")

old_tabs = 'tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical Charts", "🧭 Portfolio Risk", "⚙️ Portfolio Optimizer", "🛤️ Transition Plan"])'
new_tabs = 'tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Technical Charts", "🧭 Portfolio Risk", "⚙️ Portfolio Optimizer", "🛤️ Transition Plan", "🥧 Allocation Mix", "💳 Debt"])'

if old_tabs not in source:
    raise RuntimeError("Could not find the expected tab declaration in portfolio_core.py")

source = source.replace(old_tabs, new_tabs, 1)
source += '''\n\nfrom extra_tabs import render_allocation_tab, render_debt_tab\n\nwith tab5:\n    render_allocation_tab()\n\nwith tab6:\n    render_debt_tab()\n'''

exec(compile(source, str(source_path), "exec"), globals(), globals())
