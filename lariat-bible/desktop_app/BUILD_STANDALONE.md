# Building Standalone Executable for Lariat Bible

This guide explains how to create a standalone executable version of the Lariat Bible desktop application that can run without Python installed.

## Quick Build (Easiest Method)

### On macOS/Linux:
```bash
cd /Users/seanburdges/lariat-bible/desktop_app
chmod +x build_quick.sh
./build_quick.sh
```

### On Windows:
```cmd
cd C:\path\to\lariat-bible\desktop_app
build_quick.bat
```

## Comprehensive Build (Recommended)

Use the full build script for better control and options:

```bash
cd /Users/seanburdges/lariat-bible/desktop_app
python3 build_standalone.py
```

This will:
1. Check and install requirements
2. Clean previous builds
3. Build the executable
4. Create platform-specific installers (optional)
5. Generate run scripts

## Manual Build Steps

### 1. Install PyInstaller

```bash
pip3 install pyinstaller
```

### 2. Basic Build Command

For a simple single-file executable:

```bash
pyinstaller --name="LariatBible" --windowed --onefile --clean main.py
```

### 3. Advanced Build with Spec File

For more control, use the provided spec file:

```bash
pyinstaller LariatBible.spec --clean --noconfirm
```

## Build Options Explained

- `--onefile`: Creates a single executable file (larger, but portable)
- `--windowed`: No console window (GUI only)
- `--clean`: Clean PyInstaller cache before building
- `--name`: Name of the output executable
- `--icon`: Custom icon file (Windows: .ico, macOS: .icns)

## Output Locations

After building, your executable will be in:

- **macOS**: `dist/LariatBible.app` (application bundle)
- **Windows**: `dist/LariatBible.exe` (executable file)
- **Linux**: `dist/LariatBible` (executable file)

## Distribution Package Structure

```
dist/
├── LariatBible.exe (or .app for macOS)
└── (all included in single file)
```

## Platform-Specific Notes

### macOS
- Creates a `.app` bundle that can be dragged to Applications
- Can create DMG for distribution:
  ```bash
  hdiutil create -volname "Lariat Bible" -srcfolder dist/LariatBible.app -ov -format UDZO LariatBible.dmg
  ```

### Windows
- Single `.exe` file that can be run from anywhere
- Consider code signing to avoid security warnings
- Can create installer with Inno Setup (script provided)

### Linux
- Creates executable binary
- Consider creating AppImage for better portability:
  ```bash
  # Install appimagetool first
  appimagetool dist/LariatBible LariatBible.AppImage
  ```

## File Sizes

Typical executable sizes:
- **Windows**: ~25-35 MB
- **macOS**: ~30-40 MB  
- **Linux**: ~25-35 MB

The size includes:
- Python interpreter
- All required libraries
- Application code
- Embedded data files

## Troubleshooting

### "Module not found" errors
Install missing modules before building:
```bash
pip3 install pyyaml openpyxl pillow requests
```

### Antivirus warnings
- This is common with PyInstaller executables
- Code signing helps on Windows/macOS
- Submit to antivirus vendors for whitelisting

### Slow startup
- First run may be slower as it unpacks
- Consider using `--onedir` instead of `--onefile` for faster startup

### Missing data files
Add to spec file:
```python
datas=[
    ('data/', 'data'),
    ('*.xlsx', '.'),
]
```

## Testing the Executable

1. **Basic Test**: Run the executable directly
   ```bash
   ./dist/LariatBible  # Linux/macOS
   dist\LariatBible.exe  # Windows
   ```

2. **Clean Environment Test**: Test on a machine without Python

3. **Permission Test**: Ensure it can read/write necessary files

## Creating Installers

### Windows Installer (Inno Setup)
1. Install Inno Setup from https://jrsoftware.org
2. Use the generated `LariatBible_installer.iss` script
3. Compile to create `LariatBible_Setup.exe`

### macOS DMG
```bash
python3 build_standalone.py  # Automatically creates DMG
```

### Linux AppImage
```bash
# Download appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

# Create AppImage
./appimagetool-x86_64.AppImage dist/ LariatBible.AppImage
```

## Deployment Checklist

- [ ] Test executable on target OS
- [ ] Verify all features work
- [ ] Check file read/write permissions
- [ ] Test without Python installed
- [ ] Create installer/DMG/AppImage
- [ ] Document system requirements
- [ ] Prepare distribution package

## System Requirements for End Users

**No Python Required!**

### Minimum Requirements:
- **Windows**: Windows 7 or later
- **macOS**: macOS 10.12 (Sierra) or later
- **Linux**: Ubuntu 18.04 or equivalent
- **RAM**: 512 MB
- **Disk Space**: 100 MB

## Support

For build issues, check:
1. PyInstaller documentation: https://pyinstaller.readthedocs.io
2. Error logs in `build/LariatBible/warn-LariatBible.txt`
3. Console output when running without `--windowed` flag

## Version Management

Update version in:
1. `main.py` - `__version__`
2. `LariatBible.spec` - Bundle version
3. Installer scripts
4. README files
