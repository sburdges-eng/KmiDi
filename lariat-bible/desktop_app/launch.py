#!/usr/bin/env python3
"""
Launch script for Lariat Bible Desktop Application
"""

import sys
from pathlib import Path

# Add the desktop_app directory to Python path
app_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(app_dir))

# Import and run the main application
from main import main

if __name__ == "__main__":
    print("Starting Lariat Bible Desktop Application...")
    print("-" * 50)
    print("Restaurant Management System for The Lariat")
    print("-" * 50)
    main()
