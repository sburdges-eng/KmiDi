"""
Mobile/Web Platform Support - Cross-platform deployment for iDAW.

Provides:
- PWA (Progressive Web App) configuration
- Mobile platform evaluation
- iOS Audio Unit stubs
- Android AAP (Audio Plugins) stubs
- Streamlit cloud deployment helpers
"""

from mobile.pwa import (
    PWAManifest,
    PWAConfig,
    generate_manifest,
    generate_service_worker,
)

from mobile.platforms import (
    MobilePlatform,
    PlatformCapabilities,
    evaluate_platforms,
    get_platform_recommendation,
)

from mobile.ios_audio_unit import (
    iOSAudioUnitConfig,
    generate_ios_plugin_stub,
    get_ios_requirements,
)

from mobile.android_aap import (
    AndroidAAPConfig,
    generate_android_plugin_stub,
    get_android_requirements,
)

__all__ = [
    # PWA
    "PWAManifest",
    "PWAConfig",
    "generate_manifest",
    "generate_service_worker",
    # Platforms
    "MobilePlatform",
    "PlatformCapabilities",
    "evaluate_platforms",
    "get_platform_recommendation",
    # iOS
    "iOSAudioUnitConfig",
    "generate_ios_plugin_stub",
    "get_ios_requirements",
    # Android
    "AndroidAAPConfig",
    "generate_android_plugin_stub",
    "get_android_requirements",
]
