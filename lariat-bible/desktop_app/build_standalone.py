#!/usr/bin/env python3
"""
Build script for creating standalone executable of Lariat Bible Desktop Application
This script automates the process of building distributable versions for different platforms
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import platform

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message):
    """Print a formatted header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(message):
    """Print a success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message):
    """Print an info message"""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def check_requirements():
    """Check if all required packages are installed"""
    print_header("Checking Requirements")
    
    required_packages = {
        'PyInstaller': 'pyinstaller',
        'PyYAML': 'pyyaml',
        'Pillow': 'pillow',
        'openpyxl': 'openpyxl',
        'requests': 'requests',
    }
    
    missing_packages = []
    
    for package_name, pip_name in required_packages.items():
        try:
            __import__(package_name.lower().replace('-', '_'))
            print_success(f"{package_name} is installed")
        except ImportError:
            print_warning(f"{package_name} is not installed")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print_warning("\nSome packages are missing. Installing...")
        for package in missing_packages:
            print_info(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=False, capture_output=True)
        print_success("All packages installed!")
    else:
        print_success("All requirements satisfied!")
    
    return True


def clean_build_dirs():
    """Clean previous build directories"""
    print_header("Cleaning Previous Builds")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print_success(f"Removed {dir_name}/")
    
    # Clean .spec file if it exists and we're not using custom one
    spec_files = list(Path('.').glob('*.spec'))
    for spec_file in spec_files:
        if spec_file.name != 'LariatBible.spec':
            spec_file.unlink()
            print_success(f"Removed {spec_file.name}")


def build_executable():
    """Build the standalone executable"""
    print_header("Building Executable")
    
    system = platform.system()
    print_info(f"Building for {system}")
    
    # Determine build command based on platform
    if Path('LariatBible.spec').exists():
        # Use custom spec file
        build_cmd = [
            "pyinstaller",
            "LariatBible.spec",
            "--clean",
            "--noconfirm"
        ]
    else:
        # Build with default settings
        build_cmd = [
            "pyinstaller",
            "--name", "LariatBible",
            "--windowed",  # No console window
            "--onefile",   # Single executable file
            "--clean",
            "--noconfirm",
            "main.py"
        ]
        
        # Add platform-specific options
        if system == "Darwin":  # macOS
            build_cmd.extend(["--osx-bundle-identifier", "com.lariat.bible"])
        elif system == "Windows":
            if Path("lariat_icon.ico").exists():
                build_cmd.extend(["--icon", "lariat_icon.ico"])
    
    print_info(f"Running: {' '.join(build_cmd)}")
    
    # Run PyInstaller
    try:
        result = subprocess.run(build_cmd, capture_output=True, text=True, check=True)
        print_success("Build completed successfully!")
        
        # Show build output location
        if system == "Darwin":
            exe_path = Path("dist/LariatBible.app")
        elif system == "Windows":
            exe_path = Path("dist/LariatBible.exe")
        else:
            exe_path = Path("dist/LariatBible")
        
        if exe_path.exists():
            print_success(f"Executable created: {exe_path}")
            print_info(f"Size: {get_size(exe_path)}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Build failed: {e}")
        print_error("Error output:")
        print(e.stderr)
        return False


def get_size(path):
    """Get human readable file/folder size"""
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def create_installer():
    """Create platform-specific installer (optional)"""
    print_header("Creating Installer (Optional)")
    
    system = platform.system()
    
    if system == "Windows":
        # For Windows, we could use NSIS or Inno Setup
        print_info("For Windows installer, consider using Inno Setup")
        print_info("Download from: https://jrsoftware.org/isinfo.php")
        create_inno_setup_script()
        
    elif system == "Darwin":
        # For macOS, create a DMG
        print_info("Creating DMG installer for macOS...")
        create_dmg()
        
    else:
        # For Linux, create AppImage or .deb package
        print_info("For Linux, consider creating an AppImage")
        print_info("See: https://appimage.org/")


def create_inno_setup_script():
    """Create Inno Setup script for Windows installer"""
    script_content = '''[Setup]
AppName=Lariat Bible
AppVersion=1.0.0
AppPublisher=The Lariat Restaurant
AppPublisherURL=https://thelariat.com
DefaultDirName={autopf}\\LariatBible
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=LariatBible_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\\LariatBible.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "data\\*"; DestDir: "{app}\\data"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\\Lariat Bible"; Filename: "{app}\\LariatBible.exe"
Name: "{autodesktop}\\Lariat Bible"; Filename: "{app}\\LariatBible.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\\LariatBible.exe"; Description: "{cm:LaunchProgram,Lariat Bible}"; Flags: nowait postinstall skipifsilent
'''
    
    with open("LariatBible_installer.iss", "w") as f:
        f.write(script_content)
    
    print_success("Created Inno Setup script: LariatBible_installer.iss")
    print_info("Install Inno Setup and compile this script to create installer")


def create_dmg():
    """Create DMG for macOS distribution"""
    if not Path("dist/LariatBible.app").exists():
        print_warning("App bundle not found. Build the app first.")
        return
    
    try:
        # Create a temporary directory for DMG contents
        dmg_dir = Path("dist/dmg")
        dmg_dir.mkdir(exist_ok=True)
        
        # Copy app to DMG directory
        shutil.copytree("dist/LariatBible.app", dmg_dir / "LariatBible.app")
        
        # Create symbolic link to Applications
        os.symlink("/Applications", str(dmg_dir / "Applications"))
        
        # Create DMG
        dmg_name = "LariatBible-1.0.0.dmg"
        subprocess.run([
            "hdiutil", "create",
            "-volname", "Lariat Bible",
            "-srcfolder", str(dmg_dir),
            "-ov",
            "-format", "UDZO",
            f"dist/{dmg_name}"
        ])
        
        print_success(f"Created DMG: dist/{dmg_name}")
        
        # Cleanup
        shutil.rmtree(dmg_dir)
        
    except Exception as e:
        print_error(f"Failed to create DMG: {e}")


def create_run_scripts():
    """Create convenience run scripts for different platforms"""
    print_header("Creating Run Scripts")
    
    # Windows batch file
    bat_content = """@echo off
echo Starting Lariat Bible...
start "" "dist\\LariatBible.exe"
"""
    with open("run_windows.bat", "w") as f:
        f.write(bat_content)
    print_success("Created run_windows.bat")
    
    # Unix shell script
    sh_content = """#!/bin/bash
echo "Starting Lariat Bible..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open dist/LariatBible.app
else
    ./dist/LariatBible
fi
"""
    with open("run_unix.sh", "w") as f:
        f.write(sh_content)
    Path("run_unix.sh").chmod(0o755)
    print_success("Created run_unix.sh")


def main():
    """Main build process"""
    print_header("Lariat Bible - Standalone Build System")
    
    # Change to desktop_app directory
    os.chdir(Path(__file__).parent)
    
    # Check and install requirements
    if not check_requirements():
        return 1
    
    # Clean previous builds
    clean_build_dirs()
    
    # Build executable
    if not build_executable():
        return 1
    
    # Create run scripts
    create_run_scripts()
    
    # Optionally create installer
    create_installer()
    
    # Final summary
    print_header("Build Summary")
    
    system = platform.system()
    if system == "Darwin":
        exe_name = "LariatBible.app"
        run_cmd = "open dist/LariatBible.app"
    elif system == "Windows":
        exe_name = "LariatBible.exe"
        run_cmd = "dist\\LariatBible.exe"
    else:
        exe_name = "LariatBible"
        run_cmd = "./dist/LariatBible"
    
    print_success(f"✨ Build completed successfully!")
    print_info(f"📦 Executable: dist/{exe_name}")
    print_info(f"🚀 To run: {run_cmd}")
    print_info(f"📋 To distribute: Copy everything in the 'dist' folder")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Test the executable on your system")
    print("2. Share the 'dist' folder with your team")
    print("3. No Python installation required on target machines!")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_error("\nBuild cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
