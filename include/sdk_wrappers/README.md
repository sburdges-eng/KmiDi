# SDK wrappers (macOS 26.2+)

This directory supplies wrapped system headers so the project builds with macOS SDK 26.2+.

## `_time.h`

- **What:** Copy of the SDK `_time.h` with the `__CLOCK_AVAILABILITY` block forced to the empty branch.
- **Why:** On SDK 26.2+, the `__has_feature(enumerator_attributes)` branch makes the `clockid_t` enum fail to parse. Using the empty definition avoids that.
- **How:** CMake adds this directory as a system include path **before** the SDK (`SYSTEM BEFORE PUBLIC`), so `#include <_time.h>` resolves here first.
- **Other edits in the wrap:** The `__API_AVAILABLE(...)` on `timespec_get` is removed to prevent parse issues when this header is pulled in from some C++ include orders.

## `availability_stub.h`

- Defines `__OSX_AVAILABLE_STARTING` and `__API_AVAILABLE` to empty.
- **Not** used by default: force-including it broke other headers (e.g. `pthread.h`). Kept only as reference if you need a minimal stub for a different build or SDK.

## If you still see _wchar.h / availability errors

Those come from the SDK’s `_wchar.h` and related headers, not from `_time.h`. To get a clean build you may need to:

- Build with **Xcode** (and its bundled SDK) instead of Command Line Tools only, or  
- Use an **older SDK** (e.g. 14.x or 15.x), or  
- Use a **different toolchain** (e.g. Homebrew LLVM) and point CMake at it.
