#!/usr/bin/env python3
"""
Restart UI Script
Date: 2025-08-01

Clears Streamlit cache and restarts the UI to pick up code changes.
"""

import subprocess
import sys
import shutil
from pathlib import Path
import time

def clear_streamlit_cache():
    """Clear Streamlit cache directories."""
    cache_dirs = [
        Path.home() / ".streamlit",
        Path(".streamlit"),
        Path("__pycache__"),
        Path("src/__pycache__")
    ]
    
    print("🧹 Clearing caches...")
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                if cache_dir.is_file():
                    cache_dir.unlink()
                else:
                    shutil.rmtree(cache_dir)
                print(f"  ✅ Cleared: {cache_dir}")
            except Exception as e:
                print(f"  ⚠️  Could not clear {cache_dir}: {e}")

def restart_streamlit():
    """Restart the Streamlit application."""
    print("\n🚀 Restarting Streamlit with fresh cache...")
    print("The web interface will open automatically.")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Clear cache first
        subprocess.run([sys.executable, "-c", "import streamlit; streamlit.cache_data.clear(); streamlit.cache_resource.clear()"], 
                      capture_output=True)
        
        # Start streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function."""
    print("🔄 UI Restart Tool")
    print("=" * 30)
    
    # Clear caches
    clear_streamlit_cache()
    
    # Wait a moment
    time.sleep(1)
    
    # Restart
    restart_streamlit()

if __name__ == "__main__":
    main()