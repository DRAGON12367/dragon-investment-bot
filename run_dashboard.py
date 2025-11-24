"""
Run the professional live trading dashboard.
Usage: streamlit run run_dashboard.py
Or: python run_dashboard.py
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Get the directory of this file
    dashboard_path = os.path.join(os.path.dirname(__file__), "gui", "dashboard.py")
    
    # Run streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])

