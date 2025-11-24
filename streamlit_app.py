"""
Streamlit Cloud entry point - Main dashboard
This file should be set as the main file path in Streamlit Cloud
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import and run the dashboard
    from gui.dashboard import main
    
    if __name__ == "__main__":
        main()
except Exception as e:
    import streamlit as st
    st.error(f"Error loading dashboard: {e}")
    st.exception(e)
    st.info("Please check the Streamlit Cloud logs for more details.")

