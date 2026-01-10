"""
Mobile Platform Evaluation - React Native vs Flutter comparison for iDAW.

Provides detailed analysis and recommendations for mobile development approach.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class MobilePlatform(Enum):
    """Mobile development platforms."""
    REACT_NATIVE = "react_native"
    FLUTTER = "flutter"
    NATIVE_IOS = "native_ios"
    NATIVE_ANDROID = "native_android"
    CAPACITOR = "capacitor"
    PWA = "pwa"


class FeatureSupport(Enum):
    """Feature support level."""
    FULL = "full"
    PARTIAL = "partial"
    PLUGIN = "plugin"
    UNSUPPORTED = "unsupported"


@dataclass
class PlatformCapabilities:
    """Capabilities of a mobile platform."""
    platform: MobilePlatform
    name: str

    # Audio capabilities
    audio_playback: FeatureSupport = FeatureSupport.FULL
    audio_recording: FeatureSupport = FeatureSupport.FULL
    midi_support: FeatureSupport = FeatureSupport.UNSUPPORTED
    low_latency_audio: FeatureSupport = FeatureSupport.UNSUPPORTED
    audio_units: FeatureSupport = FeatureSupport.UNSUPPORTED

    # Development
    code_sharing: float = 0.0  # 0-100%
    hot_reload: bool = False
    native_modules: bool = False
    typescript_support: bool = False

    # Performance
    performance_rating: float = 0.0  # 0-10
    startup_time_ms: int = 0
    memory_usage_mb: int = 0

    # Ecosystem
    npm_packages: int = 0
    pub_packages: int = 0
    community_size: str = ""

    # Pros and Cons
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)

    # iDAW-specific
    idaw_recommendation: str = ""
    implementation_effort: str = ""  # "low", "medium", "high"


def evaluate_platforms() -> Dict[MobilePlatform, PlatformCapabilities]:
    """
    Evaluate all mobile platforms for iDAW development.

    Returns:
        Dict mapping platforms to their capabilities
    """
    platforms = {}

    # React Native
    platforms[MobilePlatform.REACT_NATIVE] = PlatformCapabilities(
        platform=MobilePlatform.REACT_NATIVE,
        name="React Native",
        audio_playback=FeatureSupport.FULL,
        audio_recording=FeatureSupport.PLUGIN,
        midi_support=FeatureSupport.PLUGIN,
        low_latency_audio=FeatureSupport.PLUGIN,
        audio_units=FeatureSupport.UNSUPPORTED,
        code_sharing=85.0,
        hot_reload=True,
        native_modules=True,
        typescript_support=True,
        performance_rating=7.5,
        startup_time_ms=300,
        memory_usage_mb=80,
        npm_packages=1_500_000,
        community_size="Very Large",
        pros=[
            "Large ecosystem and community",
            "TypeScript support (matches web codebase)",
            "Hot reload for fast development",
            "React skills transfer from web",
            "Good audio libraries (expo-av, react-native-audio-api)",
            "Native module bridge for performance-critical code",
        ],
        cons=[
            "JavaScript bridge overhead for audio",
            "No built-in MIDI support",
            "Audio latency higher than native",
            "Requires native modules for low-latency audio",
        ],
        idaw_recommendation="Recommended for UI-heavy features, intent input, project management",
        implementation_effort="medium",
    )

    # Flutter
    platforms[MobilePlatform.FLUTTER] = PlatformCapabilities(
        platform=MobilePlatform.FLUTTER,
        name="Flutter",
        audio_playback=FeatureSupport.FULL,
        audio_recording=FeatureSupport.PLUGIN,
        midi_support=FeatureSupport.PLUGIN,
        low_latency_audio=FeatureSupport.PLUGIN,
        audio_units=FeatureSupport.UNSUPPORTED,
        code_sharing=95.0,
        hot_reload=True,
        native_modules=True,
        typescript_support=False,
        performance_rating=8.5,
        startup_time_ms=200,
        memory_usage_mb=60,
        pub_packages=30_000,
        community_size="Large",
        pros=[
            "Better performance than React Native",
            "Single codebase for iOS/Android",
            "Beautiful custom UI widgets",
            "Hot reload with state preservation",
            "Dart's sound null safety",
            "Flutter Audio plugin for low-latency",
        ],
        cons=[
            "Different language (Dart) from web stack",
            "Smaller ecosystem than React Native",
            "No code sharing with existing Python/TypeScript",
            "Larger app bundle size",
        ],
        idaw_recommendation="Recommended for performance-critical UI, piano roll, visualization",
        implementation_effort="high",
    )

    # Native iOS
    platforms[MobilePlatform.NATIVE_IOS] = PlatformCapabilities(
        platform=MobilePlatform.NATIVE_IOS,
        name="Native iOS (Swift)",
        audio_playback=FeatureSupport.FULL,
        audio_recording=FeatureSupport.FULL,
        midi_support=FeatureSupport.FULL,
        low_latency_audio=FeatureSupport.FULL,
        audio_units=FeatureSupport.FULL,
        code_sharing=0.0,
        hot_reload=False,
        native_modules=True,
        typescript_support=False,
        performance_rating=10.0,
        startup_time_ms=100,
        memory_usage_mb=40,
        community_size="Large",
        pros=[
            "Best audio performance",
            "Full Audio Units support",
            "CoreMIDI integration",
            "AVAudioEngine for low latency",
            "Best iOS integration",
        ],
        cons=[
            "iOS only",
            "No code sharing",
            "Separate codebase required",
            "Swift/Obj-C expertise needed",
        ],
        idaw_recommendation="Required for iOS Audio Unit plugin (AUv3)",
        implementation_effort="high",
    )

    # Native Android
    platforms[MobilePlatform.NATIVE_ANDROID] = PlatformCapabilities(
        platform=MobilePlatform.NATIVE_ANDROID,
        name="Native Android (Kotlin)",
        audio_playback=FeatureSupport.FULL,
        audio_recording=FeatureSupport.FULL,
        midi_support=FeatureSupport.FULL,
        low_latency_audio=FeatureSupport.FULL,
        audio_units=FeatureSupport.UNSUPPORTED,
        code_sharing=0.0,
        hot_reload=False,
        native_modules=True,
        typescript_support=False,
        performance_rating=9.0,
        startup_time_ms=150,
        memory_usage_mb=50,
        community_size="Large",
        pros=[
            "Oboe library for low-latency audio",
            "Android MIDI API",
            "AAudio for pro audio",
            "Best Android integration",
        ],
        cons=[
            "Android only",
            "No code sharing",
            "Device fragmentation",
            "Kotlin/Java expertise needed",
        ],
        idaw_recommendation="Required for Android AAP plugin",
        implementation_effort="high",
    )

    # Capacitor (Ionic)
    platforms[MobilePlatform.CAPACITOR] = PlatformCapabilities(
        platform=MobilePlatform.CAPACITOR,
        name="Capacitor (Ionic)",
        audio_playback=FeatureSupport.FULL,
        audio_recording=FeatureSupport.PLUGIN,
        midi_support=FeatureSupport.UNSUPPORTED,
        low_latency_audio=FeatureSupport.UNSUPPORTED,
        audio_units=FeatureSupport.UNSUPPORTED,
        code_sharing=90.0,
        hot_reload=True,
        native_modules=True,
        typescript_support=True,
        performance_rating=6.0,
        startup_time_ms=400,
        memory_usage_mb=100,
        npm_packages=1_500_000,
        community_size="Medium",
        pros=[
            "Uses existing web code",
            "TypeScript/React compatible",
            "Easy web to mobile",
            "Native plugins available",
        ],
        cons=[
            "WebView performance limitations",
            "Not suitable for audio processing",
            "Higher latency than native",
            "Limited audio capabilities",
        ],
        idaw_recommendation="Not recommended for audio features, suitable for project management",
        implementation_effort="low",
    )

    # PWA
    platforms[MobilePlatform.PWA] = PlatformCapabilities(
        platform=MobilePlatform.PWA,
        name="Progressive Web App",
        audio_playback=FeatureSupport.FULL,
        audio_recording=FeatureSupport.PARTIAL,
        midi_support=FeatureSupport.PARTIAL,
        low_latency_audio=FeatureSupport.PARTIAL,
        audio_units=FeatureSupport.UNSUPPORTED,
        code_sharing=100.0,
        hot_reload=True,
        native_modules=False,
        typescript_support=True,
        performance_rating=5.0,
        startup_time_ms=200,
        memory_usage_mb=40,
        npm_packages=1_500_000,
        community_size="Very Large",
        pros=[
            "100% code sharing with web",
            "No app store required",
            "Instant updates",
            "Web Audio API available",
            "Web MIDI API (Chrome)",
            "Easy deployment",
        ],
        cons=[
            "Limited iOS support",
            "No background audio on iOS",
            "No access to Audio Units",
            "Browser limitations",
            "Not a 'real' app experience",
        ],
        idaw_recommendation="Recommended as first mobile step, intent input and project viewing",
        implementation_effort="low",
    )

    return platforms


def get_platform_recommendation(requirements: Dict[str, Any]) -> List[MobilePlatform]:
    """
    Get platform recommendations based on requirements.

    Args:
        requirements: Dict with requirement flags
            - need_audio_units: bool
            - need_low_latency: bool
            - need_midi: bool
            - share_code: bool
            - timeline: str ("short", "medium", "long")

    Returns:
        List of recommended platforms in priority order
    """
    platforms = evaluate_platforms()
    recommendations = []

    need_audio_units = requirements.get("need_audio_units", False)
    need_low_latency = requirements.get("need_low_latency", False)
    need_midi = requirements.get("need_midi", False)
    share_code = requirements.get("share_code", True)
    timeline = requirements.get("timeline", "medium")

    # Audio Units require native iOS
    if need_audio_units:
        recommendations.append(MobilePlatform.NATIVE_IOS)

    # Low latency requires native or specialized frameworks
    if need_low_latency:
        if MobilePlatform.NATIVE_IOS not in recommendations:
            recommendations.append(MobilePlatform.NATIVE_IOS)
        recommendations.append(MobilePlatform.NATIVE_ANDROID)

    # MIDI support
    if need_midi:
        if MobilePlatform.NATIVE_IOS not in recommendations:
            recommendations.append(MobilePlatform.NATIVE_IOS)
        if MobilePlatform.NATIVE_ANDROID not in recommendations:
            recommendations.append(MobilePlatform.NATIVE_ANDROID)

    # Code sharing preference
    if share_code and timeline == "short":
        recommendations.insert(0, MobilePlatform.PWA)
        recommendations.append(MobilePlatform.REACT_NATIVE)
    elif share_code:
        recommendations.append(MobilePlatform.REACT_NATIVE)
        recommendations.append(MobilePlatform.FLUTTER)

    # Default: PWA for quick start
    if not recommendations:
        recommendations = [
            MobilePlatform.PWA,
            MobilePlatform.REACT_NATIVE,
            MobilePlatform.FLUTTER,
        ]

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for p in recommendations:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


def generate_comparison_report() -> str:
    """
    Generate a markdown comparison report of all platforms.

    Returns:
        Markdown formatted report
    """
    platforms = evaluate_platforms()

    report = """# Mobile Platform Comparison for iDAW

## Executive Summary

Based on iDAW's requirements for audio processing, MIDI support, and cross-platform development,
we recommend a **phased approach**:

1. **Phase 1 (Immediate)**: PWA for basic mobile access
2. **Phase 2 (3-6 months)**: React Native for full mobile app
3. **Phase 3 (6-12 months)**: Native iOS/Android for Audio Unit plugins

## Platform Comparison

| Feature | React Native | Flutter | Native iOS | Native Android | PWA |
|---------|--------------|---------|------------|----------------|-----|
"""

    # Add feature rows
    features = [
        ("Audio Playback", "audio_playback"),
        ("Audio Recording", "audio_recording"),
        ("MIDI Support", "midi_support"),
        ("Low Latency", "low_latency_audio"),
        ("Audio Units", "audio_units"),
    ]

    for feature_name, attr in features:
        row = f"| {feature_name} |"
        for platform in [
            MobilePlatform.REACT_NATIVE,
            MobilePlatform.FLUTTER,
            MobilePlatform.NATIVE_IOS,
            MobilePlatform.NATIVE_ANDROID,
            MobilePlatform.PWA,
        ]:
            cap = platforms[platform]
            value = getattr(cap, attr)
            emoji = {
                FeatureSupport.FULL: "Full",
                FeatureSupport.PARTIAL: "Partial",
                FeatureSupport.PLUGIN: "Plugin",
                FeatureSupport.UNSUPPORTED: "No",
            }[value]
            row += f" {emoji} |"
        report += row + "\n"

    # Add performance metrics
    report += """
## Performance Metrics

| Metric | React Native | Flutter | Native iOS | Native Android | PWA |
|--------|--------------|---------|------------|----------------|-----|
"""

    for platform in [
        MobilePlatform.REACT_NATIVE,
        MobilePlatform.FLUTTER,
        MobilePlatform.NATIVE_IOS,
        MobilePlatform.NATIVE_ANDROID,
        MobilePlatform.PWA,
    ]:
        cap = platforms[platform]
        report += f"| Code Sharing | {cap.code_sharing}% |\n"
        break

    # Add recommendations
    report += """
## iDAW Recommendations

### For UI-Heavy Features (Intent Input, Project Management)
**Recommendation: React Native**
- Code sharing with web
- TypeScript support
- Large ecosystem

### For Audio Plugins
**Recommendation: Native iOS (AUv3) + Native Android (AAP)**
- Required for host integration
- Best audio performance
- Full MIDI support

### For Quick Mobile Access
**Recommendation: Progressive Web App (PWA)**
- Immediate deployment
- No app store review
- Basic audio via Web Audio API
"""

    return report
