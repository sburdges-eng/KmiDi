"""
PRROT/PARROT Worker - Main Entry Point

Allows running as: python -m prrot.worker job.json
"""

from .worker import main

if __name__ == "__main__":
    main()
