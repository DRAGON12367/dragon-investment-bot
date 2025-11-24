"""
Streamlit Cloud entry point - Main dashboard
This file should be set as the main file path in Streamlit Cloud
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
import traceback

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set up Streamlit page config first
import streamlit as st
st.set_page_config(
    page_title="AI Investment Bot - Live Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    # Import and run the dashboard
    from gui.dashboard import main
    
    if __name__ == "__main__":
        main()
except ImportError as e:
    st.error("❌ Import Error - Missing Dependencies")
    st.error(f"Error: {str(e)}")
    st.info("""
    **Common fixes:**
    1. Make sure all dependencies are in requirements.txt
    2. Check Streamlit Cloud logs for missing packages
    3. Some optional packages (like TensorFlow) may not be installed - that's OK
    """)
    st.code(traceback.format_exc())
except Exception as e:
    st.error(f"❌ Error loading dashboard: {str(e)}")
    st.error("Full error details:")
    with st.expander("Show full error traceback"):
        st.code(traceback.format_exc())
    st.info("""
    **Troubleshooting:**
    1. Check the Streamlit Cloud logs (Manage app → Logs)
    2. Make sure all required files are in the repository
    3. Verify the main file path is correct: `streamlit_app.py`
    """)

