# Installer Creation Guide

**Date**: 2025-01-02

## Overview

This guide covers creating platform-specific installers for KmiDi distribution.

## macOS Installer (.pkg)

### Requirements

- macOS with Xcode Command Line Tools
- `pkgbuild` and `productbuild` (included with macOS)

### Basic .pkg Creation

```bash
# Create package root
mkdir -p pkg_root/Applications/KmiDi.app/Contents

# Copy application files
cp -r kmidi_gui pkg_root/Applications/KmiDi.app/Contents/

# Create package
pkgbuild \
  --root pkg_root \
  --identifier com.kmidi.app \
  --version 1.0.0 \
  --install-location "/" \
  KmiDi-1.0.0.pkg
```

### Distribution Package (.mpkg)

For multiple components:

```bash
productbuild \
  --distribution Distribution.xml \
  --package-path packages \
  --resources Resources \
  KmiDi-Installer.mpkg
```

### Distribution.xml Template

```xml
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="1">
    <title>KmiDi Installer</title>
    <organization>com.kmidi</organization>
    <domains enable_localSystem="true"/>
    <options customize="never" require-scripts="false" rootVolumeOnly="true"/>
    <pkg-ref id="com.kmidi.app"/>
    <choices-outline>
        <line choice="default">
            <line choice="com.kmidi.app"/>
        </line>
    </choices-outline>
    <choice id="default"/>
    <choice id="com.kmidi.app" visible="false">
        <pkg-ref id="com.kmidi.app"/>
    </choice>
    <pkg-ref id="com.kmidi.app" version="1.0.0" onConclusion="none">KmiDi.pkg</pkg-ref>
</installer-gui-script>
```

## Linux Installer

### Debian/Ubuntu (.deb)

```bash
# Create package structure
mkdir -p deb/DEBIAN
mkdir -p deb/usr/local/bin
mkdir -p deb/usr/local/share/kmidi

# Create control file
cat > deb/DEBIAN/control <<EOF
Package: kmidi
Version: 1.0.0
Section: sound
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.9), python3-pip
Maintainer: KmiDi Team
Description: KmiDi Music Generation
EOF

# Copy files
cp -r application/* deb/usr/local/share/kmidi/

# Build package
dpkg-deb --build deb kmidi_1.0.0_amd64.deb
```

### Red Hat/CentOS (.rpm)

Requires `rpmbuild` and `.spec` file:

```spec
# kmidi.spec
Name:           kmidi
Version:        1.0.0
Release:        1%{?dist}
Summary:        KmiDi Music Generation

License:        MIT
Source0:        kmidi-%{version}.tar.gz

Requires:       python3 >= 3.9

%description
KmiDi generates music from emotional intent.

%prep
%setup -q

%build
# Build steps

%install
# Install steps

%files
/usr/local/bin/kmidi
/usr/local/share/kmidi/

%changelog
* Wed Jan 02 2025 KmiDi Team - 1.0.0-1
- Initial release
```

Build:
```bash
rpmbuild -ba kmidi.spec
```

## Windows Installer (.msi)

### Using WiX Toolset

1. Install WiX Toolset
2. Create `.wxs` file
3. Build with `candle` and `light`

### Alternative: NSIS

NSIS (Nullsoft Scriptable Install System) is simpler:

```nsis
; kmidi.nsi
Name "KmiDi"
OutFile "KmiDi-Installer.exe"
InstallDir "$PROGRAMFILES\KmiDi"

Section "Install"
    SetOutPath "$INSTDIR"
    File /r "application\*"
    CreateShortcut "$SMPROGRAMS\KmiDi.lnk" "$INSTDIR\kmidi.exe"
SectionEnd
```

Build:
```bash
makensis kmidi.nsi
```

## Automated Build

Use the build script:

```bash
bash scripts/build_installers.sh
```

## Code Signing

See [Code Signing Guide](CODE_SIGNING.md) for signing installers.

## Distribution

1. **Test Installers**: Test on clean systems
2. **Code Sign**: Sign for distribution
3. **Upload**: Upload to distribution channels
4. **Documentation**: Provide installation instructions

## See Also

- [Code Signing Guide](CODE_SIGNING.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
