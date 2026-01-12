# iDAW Tauri Backend

Rust backend for the iDAW desktop application.

## Building

```bash
# Development
npm run tauri dev

# Production build
npm run tauri build
```

## Troubleshooting

### macOS External Drive Build Errors

If building from an external drive and you see errors like:
```
failed to read file '..._default.toml': stream did not contain valid UTF-8
```

This is caused by macOS creating `._*` resource fork files on external drives.

**Fix:** Create `.cargo/config.toml` to redirect build artifacts to internal storage:

```toml
[build]
target-dir = "/Users/YOUR_USERNAME/.cargo/idaw-target"
```

This file is in `.gitignore` so each developer can set their own path.
