/* KmiDi SDK 26.2+ wrap: stub availability macros so _wchar.h etc. parse.
 * Include this before system headers when building with SDK 26.2+.
 * Defines to empty so __OSX_AVAILABLE_STARTING(...) and __API_AVAILABLE(...)
 * expand to nothing in decls.
 */
#ifndef KMIDI_AVAILABILITY_STUB_H
#define KMIDI_AVAILABILITY_STUB_H

#if defined(__APPLE__) || defined(__APPLE_CPP__)
#  ifndef __OSX_AVAILABLE_STARTING
#    define __OSX_AVAILABLE_STARTING(_mac, _ios)
#  endif
#  ifndef __API_AVAILABLE
#    define __API_AVAILABLE(...)
#  endif
#endif

#endif
