#!/usr/bin/env python3
"""Launch UI Gate. Run from python/ or with PYTHONPATH=python/."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui_gate.ui_gate import main
sys.exit(main())
