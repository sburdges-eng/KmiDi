/* macOS SDK 26.2+ compatibility: define __CLOCK_AVAILABILITY to empty so that
 * _time.h enum expansions parse when __has_feature(enumerator_attributes).
 * Force-include via target_compile_options(-include ...) for KellyCore/plugin.
 * Only define __CLOCK_AVAILABILITY; do not override __OSX_AVAILABLE etc.
 */
#if (defined(__APPLE__) || defined(__APPLE_CPP__)) && !defined(__CLOCK_AVAILABILITY)
#define __CLOCK_AVAILABILITY
#endif
