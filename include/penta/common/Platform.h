#pragma once

/**
 * Platform compatibility layer for Penta-Core
 * Handles macOS SDK compatibility issues and cross-platform definitions
 *
 * IMPORTANT: This file MUST be included before any other system headers!
 */

// ============================================================================
// macOS SDK availability macro handling
// Include system Availability.h FIRST to properly define all availability macros
// before any other system headers are included
// ============================================================================
#if defined(__APPLE__) || defined(__APPLE_CPP__) || defined(__APPLE_CC__)

#include <Availability.h>
#include <AvailabilityMacros.h>
#include <TargetConditionals.h>

#endif // __APPLE__

// Standard includes - safe after the above definitions
#include <cstddef>
