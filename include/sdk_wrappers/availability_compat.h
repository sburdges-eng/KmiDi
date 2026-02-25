/* KmiDi SDK 26.2+ compatibility: comprehensive availability macro stubs.
 * Force-include this before any system headers to ensure availability macros
 * expand to nothing when the SDK's Availability.h doesn't define them properly.
 */
#ifndef KMIDI_AVAILABILITY_COMPAT_H
#define KMIDI_AVAILABILITY_COMPAT_H

#if (defined(__APPLE__) || defined(__APPLE_CPP__))
 // Only define if not already defined (let SDK's Availability.h define them if it can)
 #ifndef __OSX_AVAILABLE_STARTING
  #define __OSX_AVAILABLE_STARTING(_mac, _ios)
 #endif
 #ifndef __OSX_AVAILABLE_BUT_DEPRECATED
  #define __OSX_AVAILABLE_BUT_DEPRECATED(_mac_intro, _mac_dep, _ios_intro, _ios_dep)
 #endif
 #ifndef __OSX_AVAILABLE_BUT_DEPRECATED_MSG
  #define __OSX_AVAILABLE_BUT_DEPRECATED_MSG(_mac_intro, _mac_dep, _ios_intro, _ios_dep, _msg)
 #endif
 #ifndef __OSX_DEPRECATED
  #define __OSX_DEPRECATED(_mac_intro, _mac_dep, _msg)
 #endif
 #ifndef __IOS_DEPRECATED
  #define __IOS_DEPRECATED(_ios_intro, _ios_dep, _msg)
 #endif
 #ifndef __TVOS_DEPRECATED
  #define __TVOS_DEPRECATED(_tvos_intro, _tvos_dep, _msg)
 #endif
 #ifndef __WATCHOS_DEPRECATED
  #define __WATCHOS_DEPRECATED(_watchos_intro, _watchos_dep, _msg)
 #endif
 #ifndef __WATCHOS_PROHIBITED
  #define __WATCHOS_PROHIBITED
 #endif
 #ifndef __TVOS_PROHIBITED
  #define __TVOS_PROHIBITED
 #endif
 #ifndef __IOS_PROHIBITED
  #define __IOS_PROHIBITED
 #endif
 #ifndef __API_AVAILABLE
  #define __API_AVAILABLE(...)
 #endif
 #ifndef __API_UNAVAILABLE
  #define __API_UNAVAILABLE(...)
 #endif
 #ifndef __API_DEPRECATED
  #define __API_DEPRECATED(...)
 #endif
 #ifndef __OS_AVAILABILITY_MSG
  #define __OS_AVAILABILITY_MSG(_platform, ...)
 #endif
 #ifndef __SWIFT_UNAVAILABLE_MSG
  #define __SWIFT_UNAVAILABLE_MSG(_msg)
 #endif
 #ifndef __OSX_AVAILABLE
  #define __OSX_AVAILABLE(_v)
 #endif
 #ifndef __IOS_AVAILABLE
  #define __IOS_AVAILABLE(_v)
 #endif
 #ifndef __TVOS_AVAILABLE
  #define __TVOS_AVAILABLE(_v)
 #endif
 #ifndef __WATCHOS_AVAILABLE
  #define __WATCHOS_AVAILABLE(_v)
 #endif
 // libc++ internal availability macros
 #ifndef _LIBCPP_AVAILABILITY_FROM_CHARS_FLOATING_POINT
  #define _LIBCPP_AVAILABILITY_FROM_CHARS_FLOATING_POINT
 #endif
#endif

#endif
