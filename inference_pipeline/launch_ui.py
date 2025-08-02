#!/usr/bin/env python3
"""
UI Launcher for Road Distress Classification
Date: 2025-08-01

Simple launcher script for the Streamlit web interface.
"""

import subprocess
import sys
from pathlib import Path
import webbrowser
import time
import threading

def launch_streamlit():
    """Launch the Streamlit application."""
    app_path = Path(__file__).parent / "app.py"
    
    print("🛣️ Road Distress Classification - Web UI")
    print("=" * 50)
    print("Starting Streamlit server...")
    print("The web interface will open automatically in your browser.")
    print("If it doesn't open, navigate to: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the web interface...")
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    launch_streamlit()