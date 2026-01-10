# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.

# Keep JUCE Java classes
-keep class com.rmsl.juce.** { *; }

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep JNI bridge classes
-keep class com.idaw.jni.** { *; }

# Keep Penta Core classes
-keep class com.idaw.** { *; }

