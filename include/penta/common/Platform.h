#pragma once

/**
 * Platform compatibility layer for Penta-Core
 * Handles macOS SDK compatibility issues and cross-platform definitions
 */

// Use platform headers/macros as provided by the SDK. Local macro shims here
// can shadow Apple availability attributes and break unrelated headers.
// Keep this file as a lightweight include wrapper only.
#include <cstdint>
#include <cstddef>
