# Code Signing Guide

**Date**: 2025-01-02

## Overview

Code signing ensures users can trust KmiDi installers and applications. This guide covers signing for macOS and Windows.

## macOS Code Signing

### Requirements

- Apple Developer Account ($99/year)
- Xcode
- Certificate from Apple Developer Portal

### Obtain Certificate

1. **Apple Developer Account**: Sign up at developer.apple.com
2. **Create Certificate**: Create "Developer ID Application" certificate
3. **Download**: Download and install in Keychain

### Sign Application

```bash
# Sign application bundle
codesign --sign "Developer ID Application: Your Name" \
  --options runtime \
  --timestamp \
  KmiDi.app

# Verify signature
codesign --verify --verbose KmiDi.app

# Check entitlements
codesign -d --entitlements - KmiDi.app
```

### Sign Installer

```bash
# Sign .pkg installer
productsign --sign "Developer ID Installer: Your Name" \
  KmiDi-unsigned.pkg \
  KmiDi-signed.pkg

# Verify installer signature
pkgutil --check-signature KmiDi-signed.pkg
```

### Notarization (Required for Distribution)

```bash
# Submit for notarization
xcrun notarytool submit KmiDi-signed.pkg \
  --apple-id your@email.com \
  --team-id YOUR_TEAM_ID \
  --password app-specific-password \
  --wait

# Staple notarization ticket
xcrun stapler staple KmiDi-signed.pkg

# Verify notarization
xcrun stapler validate KmiDi-signed.pkg
```

### Automated Signing Script

```bash
#!/bin/bash
# sign_macos.sh

APP_PATH="KmiDi.app"
IDENTITY="Developer ID Application: Your Name"
INSTALLER_IDENTITY="Developer ID Installer: Your Name"

# Sign app
codesign --sign "$IDENTITY" \
  --options runtime \
  --timestamp \
  --force \
  "$APP_PATH"

# Build and sign installer
pkgbuild --root pkg_root \
  --identifier com.kmidi.app \
  --sign "$INSTALLER_IDENTITY" \
  --timestamp \
  KmiDi.pkg

# Notarize
xcrun notarytool submit KmiDi.pkg \
  --apple-id "$APPLE_ID" \
  --team-id "$TEAM_ID" \
  --password "$APP_PASSWORD" \
  --wait

# Staple
xcrun stapler staple KmiDi.pkg
```

## Windows Code Signing

### Requirements

- Code signing certificate from certificate authority
- `signtool.exe` (included with Windows SDK)

### Obtain Certificate

Options:
- **Commercial CA**: DigiCert, Sectigo, etc. ($200-500/year)
- **Internal CA**: For internal distribution
- **Self-signed**: For testing only (not trusted)

### Sign Executable

```powershell
# Sign executable
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com kmidi.exe

# Sign with certificate from store
signtool sign /n "Certificate Name" /t http://timestamp.digicert.com kmidi.exe

# Verify signature
signtool verify /pa kmidi.exe
```

### Sign Installer

```powershell
# Sign .msi installer
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com KmiDi-Installer.msi

# Sign .exe installer (NSIS)
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com KmiDi-Installer.exe
```

### Automated Signing Script

```powershell
# sign_windows.ps1

$CertificatePath = "certificate.pfx"
$CertificatePassword = "password"
$TimestampServer = "http://timestamp.digicert.com"

# Sign executable
signtool sign /f $CertificatePath /p $CertificatePassword /t $TimestampServer kmidi.exe

# Sign installer
signtool sign /f $CertificatePath /p $CertificatePassword /t $TimestampServer KmiDi-Installer.exe

# Verify
signtool verify /pa kmidi.exe
signtool verify /pa KmiDi-Installer.exe
```

## Linux Code Signing

### GPG Signing

```bash
# Create GPG key (if needed)
gpg --gen-key

# Sign package
dpkg-sig --sign builder kmidi_1.0.0_amd64.deb

# Verify signature
dpkg-sig --verify kmidi_1.0.0_amd64.deb
```

### RPM Signing

```bash
# Sign RPM package
rpm --addsign kmidi-1.0.0-1.x86_64.rpm

# Verify signature
rpm --checksig kmidi-1.0.0-1.x86_64.rpm
```

## CI/CD Integration

### GitHub Actions (macOS)

```yaml
- name: Code Sign
  run: |
    security create-keychain -p "${{ secrets.KEYCHAIN_PASSWORD }}" build.keychain
    security default-keychain -s build.keychain
    security unlock-keychain -p "${{ secrets.KEYCHAIN_PASSWORD }}" build.keychain
    security import certificate.p12 -k build.keychain -P "${{ secrets.CERT_PASSWORD }}" -T /usr/bin/codesign
    
    codesign --sign "Developer ID Application: Name" KmiDi.app
```

### GitHub Actions (Windows)

```yaml
- name: Code Sign
  run: |
    $pfx = [Convert]::FromBase64String("${{ secrets.CERT_BASE64 }}")
    [IO.File]::WriteAllBytes("cert.pfx", $pfx)
    signtool sign /f cert.pfx /p "${{ secrets.CERT_PASSWORD }}" kmidi.exe
```

## Best Practices

1. **Store Certificates Securely**: Use secrets management
2. **Automate**: Integrate into CI/CD pipeline
3. **Test Signing**: Test on clean systems
4. **Keep Certificates Updated**: Renew before expiration
5. **Document Process**: Keep signing process documented

## Troubleshooting

### macOS

- **"code object is not signed"**: Ensure certificate is in Keychain
- **Notarization fails**: Check for malware, review logs
- **Timestamp fails**: Check internet connection, timestamp server

### Windows

- **"The specified PFX password is not correct"**: Verify password
- **Timestamp fails**: Check timestamp server URL
- **Certificate not trusted**: Ensure certificate is from trusted CA

## See Also

- [Installer Guide](INSTALLER_GUIDE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
