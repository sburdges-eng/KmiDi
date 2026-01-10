# DAiW Music-Brain Installer for Windows
# ========================================
# Version: 0.2.0

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DAiW Music-Brain" -ForegroundColor Cyan
Write-Host "  Version 0.2.0" -ForegroundColor Cyan
Write-Host "  Installer for Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for Python
$pythonCmd = $null
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.(9|1[0-3])") {
        $pythonCmd = "python"
        Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Python 3.9+ is required but not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Python from:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or via Windows Package Manager:" -ForegroundColor Yellow
    Write-Host "  winget install Python.Python.3.12" -ForegroundColor Yellow
    exit 1
}

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $pythonCmd -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to upgrade pip" -ForegroundColor Red
    exit 1
}
Write-Host "✓ pip upgraded" -ForegroundColor Green

# Install DAiW package
Write-Host ""
Write-Host "Installing DAiW Music-Brain..." -ForegroundColor Cyan
& $pythonCmd -m pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install DAiW" -ForegroundColor Red
    exit 1
}
Write-Host "✓ DAiW installed" -ForegroundColor Green

# Install optional dependencies
Write-Host ""
Write-Host "Installing optional dependencies..." -ForegroundColor Cyan
& $pythonCmd -m pip install -e ".[audio,ui]"
Write-Host "✓ Optional dependencies installed" -ForegroundColor Green

# Create Start Menu shortcut (optional)
Write-Host ""
$createShortcut = Read-Host "Create Start Menu shortcut? (Y/n)"
if ($createShortcut -eq "" -or $createShortcut -eq "Y" -or $createShortcut -eq "y") {
    $WshShell = New-Object -ComObject WScript.Shell
    $StartMenuPath = [Environment]::GetFolderPath('StartMenu')
    $Shortcut = $WshShell.CreateShortcut("$StartMenuPath\Programs\DAiW Music-Brain.lnk")
    $Shortcut.TargetPath = "$pythonCmd"
    $Shortcut.Arguments = "-m music_brain.cli"
    $Shortcut.WorkingDirectory = $PWD
    $Shortcut.Description = "DAiW Music-Brain CLI"
    $Shortcut.Save()
    Write-Host "✓ Start Menu shortcut created" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ✓ Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now use DAiW from the command line:" -ForegroundColor White
Write-Host "  daiw --help" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or run the desktop UI:" -ForegroundColor White
Write-Host "  daiw-desktop" -ForegroundColor Yellow
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor White
Write-Host ""
Write-Host "Enjoy making music! 🎵" -ForegroundColor Cyan
