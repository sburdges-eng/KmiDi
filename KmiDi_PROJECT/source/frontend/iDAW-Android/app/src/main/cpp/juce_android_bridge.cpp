// juce_android_bridge.cpp
// JNI bridge implementation for JUCE Android Java classes

#include <jni.h>
#include <android/log.h>
#include <string>
#include <cstring>

#define LOG_TAG "JUCEBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Forward declarations (defined in penta_core_wrapper.cpp)
extern void pentaCoreHandleMidiMessage(const uint8_t* data, size_t length, long timestamp);
extern void pentaCoreOnMidiDevicesChanged();

//==============================================================================
// JuceMidiSupport native methods
//==============================================================================

extern "C" {

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceMidiSupport_00024JuceMidiInputPort_handleReceive(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jbyteArray msg,
    jint offset,
    jint count,
    jlong timestamp)
{
    if (msg == nullptr || count <= 0) {
        return;
    }

    jbyte* bytes = env->GetByteArrayElements(msg, nullptr);
    if (bytes == nullptr) {
        LOGE("Failed to get byte array elements");
        return;
    }

    // Forward MIDI message to Penta Core
    pentaCoreHandleMidiMessage(
        reinterpret_cast<const uint8_t*>(bytes + offset),
        static_cast<size_t>(count),
        timestamp
    );

    env->ReleaseByteArrayElements(msg, bytes, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceMidiSupport_handleDevicesChanged(
    JNIEnv* /* env */,
    jclass /* clazz */)
{
    LOGI("MIDI devices changed");
    pentaCoreOnMidiDevicesChanged();
}

//==============================================================================
// FragmentOverlay native methods
//==============================================================================

JNIEXPORT void JNICALL
Java_com_rmsl_juce_FragmentOverlay_onCreateNative(
    JNIEnv* env,
    jobject /* this */,
    jlong myself,
    jobject state)
{
    LOGI("FragmentOverlay onCreateNative: %ld", myself);
    // Fragment lifecycle handling - can be extended for specific needs
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_FragmentOverlay_onStartNative(
    JNIEnv* env,
    jobject /* this */,
    jlong myself)
{
    LOGI("FragmentOverlay onStartNative: %ld", myself);
    // Fragment start handling
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_FragmentOverlay_onActivityResultNative(
    JNIEnv* env,
    jobject /* this */,
    jlong myself,
    jint requestCode,
    jint resultCode,
    jobject data)
{
    LOGI("FragmentOverlay onActivityResultNative: %ld, requestCode: %d, resultCode: %d",
         myself, requestCode, resultCode);
    // Activity result handling - typically for permission requests
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_FragmentOverlay_onRequestPermissionsResultNative(
    JNIEnv* env,
    jobject /* this */,
    jlong myself,
    jint requestCode,
    jobjectArray permissions,
    jintArray grantResults)
{
    LOGI("FragmentOverlay onRequestPermissionsResultNative: %ld, requestCode: %d",
         myself, requestCode);
    
    if (grantResults != nullptr) {
        jint* results = env->GetIntArrayElements(grantResults, nullptr);
        if (results != nullptr) {
            jsize len = env->GetArrayLength(grantResults);
            for (jsize i = 0; i < len; ++i) {
                LOGI("Permission result[%zd]: %d", i, results[i]);
            }
            env->ReleaseIntArrayElements(grantResults, results, JNI_ABORT);
        }
    }
}

//==============================================================================
// JuceWebView native methods
//==============================================================================

JNIEXPORT jboolean JNICALL
Java_com_rmsl_juce_JuceWebView_00024Client_webViewPageLoadStarted(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject view,
    jstring url)
{
    if (url == nullptr) {
        return JNI_TRUE;
    }
    
    const char* urlStr = env->GetStringUTFChars(url, nullptr);
    if (urlStr != nullptr) {
        LOGI("WebView page load started: %s", urlStr);
        env->ReleaseStringUTFChars(url, urlStr);
    }
    
    return JNI_TRUE; // Allow loading
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceWebView_00024Client_webViewPageLoadFinished(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject view,
    jstring url)
{
    if (url != nullptr) {
        const char* urlStr = env->GetStringUTFChars(url, nullptr);
        if (urlStr != nullptr) {
            LOGI("WebView page load finished: %s", urlStr);
            env->ReleaseStringUTFChars(url, urlStr);
        }
    }
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceWebView_00024Client_webViewReceivedSslError(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject view,
    jobject handler,
    jobject error)
{
    LOGI("WebView SSL error received");
    // SSL error handling - typically proceed with caution in development
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceWebView_00024ChromeClient_webViewCloseWindowRequest(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject window)
{
    LOGI("WebView close window requested");
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceWebView_00024ChromeClient_webViewCreateWindowRequest(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject view)
{
    LOGI("WebView create window requested");
}

//==============================================================================
// JuceWebView21 native methods (same as JuceWebView)
//==============================================================================

JNIEXPORT jboolean JNICALL
Java_com_rmsl_juce_JuceWebView21_00024Client_webViewPageLoadStarted(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject view,
    jstring url)
{
    return Java_com_rmsl_juce_JuceWebView_00024Client_webViewPageLoadStarted(env, nullptr, host, view, url);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceWebView21_00024Client_webViewPageLoadFinished(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject view,
    jstring url)
{
    Java_com_rmsl_juce_JuceWebView_00024Client_webViewPageLoadFinished(env, nullptr, host, view, url);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceWebView21_00024Client_webViewReceivedSslError(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject view,
    jobject handler,
    jobject error)
{
    Java_com_rmsl_juce_JuceWebView_00024Client_webViewReceivedSslError(env, nullptr, host, view, handler, error);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceWebView21_00024ChromeClient_webViewCloseWindowRequest(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject window)
{
    Java_com_rmsl_juce_JuceWebView_00024ChromeClient_webViewCloseWindowRequest(env, nullptr, host, window);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_JuceWebView21_00024ChromeClient_webViewCreateWindowRequest(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject view)
{
    Java_com_rmsl_juce_JuceWebView_00024ChromeClient_webViewCreateWindowRequest(env, nullptr, host, view);
}

//==============================================================================
// Camera native methods (stubs - implement as needed)
//==============================================================================

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraDeviceStateCallback_cameraDeviceStateOpened(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject camera)
{
    LOGI("Camera device opened: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraDeviceStateCallback_cameraDeviceStateClosed(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject camera)
{
    LOGI("Camera device closed: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraDeviceStateCallback_cameraDeviceStateDisconnected(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject camera)
{
    LOGI("Camera device disconnected: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraDeviceStateCallback_cameraDeviceStateError(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject camera,
    jint error)
{
    LOGE("Camera device error: %ld, error code: %d", host, error);
}

// Camera capture session callbacks (stubs)
JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionStateCallback_cameraCaptureSessionConfigured(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject session)
{
    LOGI("Camera capture session configured: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionStateCallback_cameraCaptureSessionConfigureFailed(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject session)
{
    LOGE("Camera capture session configure failed: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionStateCallback_cameraCaptureSessionActive(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject session)
{
    LOGI("Camera capture session active: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionStateCallback_cameraCaptureSessionReady(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject session)
{
    LOGI("Camera capture session ready: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionStateCallback_cameraCaptureSessionClosed(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject session)
{
    LOGI("Camera capture session closed: %ld", host);
}

// Camera capture callbacks (stubs)
JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionCaptureCallback_cameraCaptureSessionCaptureStarted(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jboolean isPreview,
    jobject session,
    jobject request,
    jlong timestamp,
    jlong frameNumber)
{
    LOGI("Camera capture started: %ld, preview: %d", host, isPreview);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionCaptureCallback_cameraCaptureSessionCaptureCompleted(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jboolean isPreview,
    jobject session,
    jobject request,
    jobject result)
{
    LOGI("Camera capture completed: %ld, preview: %d", host, isPreview);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionCaptureCallback_cameraCaptureSessionCaptureFailed(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jboolean isPreview,
    jobject session,
    jobject request,
    jobject failure)
{
    LOGE("Camera capture failed: %ld, preview: %d", host, isPreview);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionCaptureCallback_cameraCaptureSessionCaptureProgressed(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jboolean isPreview,
    jobject session,
    jobject request,
    jobject partialResult)
{
    // Progress updates - typically not logged to avoid spam
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionCaptureCallback_cameraCaptureSessionCaptureSequenceCompleted(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jboolean isPreview,
    jobject session,
    jint sequenceId,
    jlong frameNumber)
{
    LOGI("Camera capture sequence completed: %ld, sequence: %d", host, sequenceId);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_CameraCaptureSessionCaptureCallback_cameraCaptureSessionCaptureSequenceAborted(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jboolean isPreview,
    jobject session,
    jint sequenceId)
{
    LOGE("Camera capture sequence aborted: %ld, sequence: %d", host, sequenceId);
}

//==============================================================================
// Media session native methods (stubs)
//==============================================================================

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_MediaSessionCallback_mediaSessionPlay(
    JNIEnv* env,
    jobject /* this */,
    jlong host)
{
    LOGI("Media session play: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_MediaSessionCallback_mediaSessionPause(
    JNIEnv* env,
    jobject /* this */,
    jlong host)
{
    LOGI("Media session pause: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_MediaSessionCallback_mediaSessionStop(
    JNIEnv* env,
    jobject /* this */,
    jlong host)
{
    LOGI("Media session stop: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_MediaSessionCallback_mediaSessionSeekTo(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jlong pos)
{
    LOGI("Media session seek to: %ld, position: %ld", host, pos);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_MediaSessionCallback_mediaSessionPlayFromMediaId(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jstring mediaId,
    jobject extras)
{
    if (mediaId != nullptr) {
        const char* idStr = env->GetStringUTFChars(mediaId, nullptr);
        if (idStr != nullptr) {
            LOGI("Media session play from media ID: %ld, id: %s", host, idStr);
            env->ReleaseStringUTFChars(mediaId, idStr);
        }
    }
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_MediaControllerCallback_mediaControllerPlaybackStateChanged(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject state)
{
    LOGI("Media controller playback state changed: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_MediaControllerCallback_mediaControllerMetadataChanged(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject metadata)
{
    LOGI("Media controller metadata changed: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_MediaControllerCallback_mediaControllerAudioInfoChanged(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jobject info)
{
    LOGI("Media controller audio info changed: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_MediaControllerCallback_mediaControllerSessionDestroyed(
    JNIEnv* env,
    jobject /* this */,
    jlong host)
{
    LOGI("Media controller session destroyed: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_SystemVolumeObserver_mediaSessionSystemVolumeChanged(
    JNIEnv* env,
    jobject /* this */,
    jlong host)
{
    LOGI("System volume changed: %ld", host);
}

JNIEXPORT void JNICALL
Java_com_rmsl_juce_video_JuceOrientationEventListener_deviceOrientationChanged(
    JNIEnv* env,
    jobject /* this */,
    jlong host,
    jint orientation)
{
    LOGI("Device orientation changed: %ld, orientation: %d", host, orientation);
}

} // extern "C"

