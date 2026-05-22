# Rust Vendor Directory

This directory is the deterministic source for Rust crates used by `engine/intent_ir`.

Offline policy:
- Cargo is configured to refuse network access.
- `crates.io` is replaced by this local directory.
- Builds fail closed when required crates are missing.

To populate/update this directory in a controlled environment:

```bash
cargo vendor third_party/rust_vendor --manifest-path engine/intent_ir/Cargo.toml
```

Commit the resulting crate sources and checksums to keep builds reproducible.
