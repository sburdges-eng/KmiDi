#!/usr/bin/env python
"""
The Lariat Bible - Setup Script
Comprehensive Restaurant Management System
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file if it doesn't exist"""
    env_file = Path('.env')
    if not env_file.exists():
        env_content = """# The Lariat Bible Configuration

# Database
DATABASE_URL=sqlite:///lariat.db

# Security
SECRET_KEY=change-this-to-a-random-secret-key

# Paths
INVOICE_STORAGE_PATH=./data/invoices
BACKUP_PATH=./backups

# Business Settings
RESTAURANT_NAME=The Lariat
LOCATION=Fort Collins, CO
DEFAULT_CATERING_MARGIN=0.45
DEFAULT_RESTAURANT_MARGIN=0.04

# Vendor Information
PRIMARY_VENDOR=Shamrock Foods
COMPARISON_VENDOR=SYSCO

# OCR Settings
TESSERACT_PATH=/usr/bin/tesseract  # Update based on your system

# Web Interface
FLASK_ENV=development
FLASK_DEBUG=True
HOST=127.0.0.1
PORT=5000
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file - Please update with your settings")
    else:
        print("ℹ️  .env file already exists")

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/invoices',
        'data/invoices/processed',
        'data/exports',
        'logs',
        'reports',
        'backups',
        'static',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep files to preserve directory structure
        gitkeep = Path(directory) / '.gitkeep'
        gitkeep.touch(exist_ok=True)
    
    print("✅ Created project directories")

def create_database():
    """Initialize the database"""
    try:
        from core.database import init_db
        init_db()
        print("✅ Initialized database")
    except ImportError:
        print("⚠️  Database module not yet implemented - skipping DB initialization")

def main():
    """Main setup function"""
    print("\n🤠 Setting up The Lariat Bible...\n")
    
    # Create environment file
    create_env_file()
    
    # Create necessary directories
    create_directories()
    
    # Initialize database
    create_database()
    
    print("\n✨ Setup complete! Next steps:")
    print("1. Update .env file with your settings")
    print("2. Run 'pip install -r requirements.txt' to install dependencies")
    print("3. Run 'python app.py' to start the web interface")
    print("\nHappy managing! 🎉")

if __name__ == "__main__":
    main()
