#!/usr/bin/env python3
"""
Startup script for the Cell Tower Capacity Heatmap Visualizer
"""
import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import folium
        import pandas
        import numpy
        import scipy
        print("✅ All required packages are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def main():
    print("🚀 Starting Cell Tower Capacity Heatmap Visualizer...")
    
    # Check if requirements are met
    if not check_requirements():
        sys.exit(1)
    

    # Start Streamlit app
    print("🌐 Starting web interface...")
    print("📖 The app will open in your default browser")
    print("🔄 Use Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, 
            "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port=8507",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main() 