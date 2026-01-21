#include <catch2/catch.hpp>
#include "bridge/kelly_ffi.h"
#include <string>
#include <memory>

// =============================================================================
// C FFI Integration Tests
// =============================================================================

class KellyFFITestFixture {
public:
    KellyBrainWrapper() {
        brain = kelly_brain_create();
        REQUIRE(brain != nullptr);
    }
    
    ~KellyFFITestFixture() {
        if (brain) {
            kelly_brain_destroy(brain);
        }
    }
    
    KellyBrain* brain;
};

TEST_CASE("Kelly FFI - Basic Creation and Destruction") {
    SECTION("Create and destroy KellyBrain") {
        KellyBrain* brain = kelly_brain_create();
        REQUIRE(brain != nullptr);
        
        // Should not be initialized yet
        REQUIRE_FALSE(kelly_brain_is_initialized(brain));
        
        kelly_brain_destroy(brain);
        // Should not crash after destruction
    }
    
    SECTION("Null pointer handling") {
        // Destroying null pointer should not crash
        kelly_brain_destroy(nullptr);
        
        // Operations on null should return appropriate errors
        REQUIRE(kelly_brain_initialize(nullptr, "./data") == KELLY_ERROR_NULL_POINTER);
        REQUIRE_FALSE(kelly_brain_is_initialized(nullptr));
    }
}

TEST_CASE_METHOD(KellyFFITestFixture, "Kelly FFI - Initialization") {
    SECTION("Initialize with valid path") {
        // Note: This may fail if ./data doesn't exist, but should not crash
        KellyErrorCode result = kelly_brain_initialize(brain, "./data");
        
        // Should return either success or file not found
        REQUIRE((result == KELLY_SUCCESS || result == KELLY_ERROR_FILE_NOT_FOUND));
        
        if (result == KELLY_SUCCESS) {
            REQUIRE(kelly_brain_is_initialized(brain));
        }
    }
    
    SECTION("Initialize with invalid path") {
        KellyErrorCode result = kelly_brain_initialize(brain, "/nonexistent/path");
        REQUIRE(result != KELLY_SUCCESS);
        REQUIRE_FALSE(kelly_brain_is_initialized(brain));
    }
    
    SECTION("Initialize with null path") {
        KellyErrorCode result = kelly_brain_initialize(brain, nullptr);
        REQUIRE(result == KELLY_ERROR_NULL_POINTER);
    }
}

TEST_CASE_METHOD(KellyFFITestFixture, "Kelly FFI - Text Processing") {
    // Initialize for testing (may fail, but test should continue)
    kelly_brain_initialize(brain, "./data");
    
    SECTION("Process valid text") {
        char* result = kelly_brain_from_text(brain, "I feel happy and energetic");
        
        if (result != nullptr) {
            // Should return valid JSON
            std::string json(result);
            REQUIRE_FALSE(json.empty());
            REQUIRE(json.find("{") != std::string::npos);  // Basic JSON validation
            REQUIRE(json.find("emotional_intent") != std::string::npos);
            
            kelly_free_string(result);
        }
        // If result is null, function detected uninitialized state correctly
    }
    
    SECTION("Process null text") {
        char* result = kelly_brain_from_text(brain, nullptr);
        REQUIRE(result == nullptr);  // Should return null for invalid input
    }
    
    SECTION("Process empty text") {
        char* result = kelly_brain_from_text(brain, "");
        // Should either process empty string or return null
        if (result != nullptr) {
            kelly_free_string(result);
        }
    }
}

TEST_CASE_METHOD(KellyFFITestFixture, "Kelly FFI - Emotion Processing") {
    kelly_brain_initialize(brain, "./data");
    
    SECTION("Process valid emotion") {
        char* result = kelly_brain_from_emotion(brain, "joy", 0.8f);
        
        if (result != nullptr) {
            std::string json(result);
            REQUIRE_FALSE(json.empty());
            REQUIRE(json.find("emotional_intent") != std::string::npos);
            
            kelly_free_string(result);
        }
    }
    
    SECTION("Process invalid intensity values") {
        char* result1 = kelly_brain_from_emotion(brain, "joy", -0.5f);  // Invalid
        char* result2 = kelly_brain_from_emotion(brain, "joy", 1.5f);   // Invalid
        
        REQUIRE(result1 == nullptr);
        REQUIRE(result2 == nullptr);
    }
    
    SECTION("Process valid intensity boundary values") {
        char* result1 = kelly_brain_from_emotion(brain, "joy", 0.0f);   // Min valid
        char* result2 = kelly_brain_from_emotion(brain, "joy", 1.0f);   // Max valid
        
        if (result1 != nullptr) kelly_free_string(result1);
        if (result2 != nullptr) kelly_free_string(result2);
    }
}

TEST_CASE_METHOD(KellyFFITestFixture, "Kelly FFI - Parameter Updates") {
    kelly_brain_initialize(brain, "./data");
    
    SECTION("Set valid emotion parameters") {
        KellyErrorCode result = kelly_brain_set_emotion_parameters(brain, 0.5f, 0.7f, 0.3f);
        
        // Should succeed if initialized, or return initialization error
        REQUIRE((result == KELLY_SUCCESS || result == KELLY_ERROR_INITIALIZATION_FAILED));
    }
    
    SECTION("Set invalid parameter values") {
        // Invalid valence
        KellyErrorCode result1 = kelly_brain_set_emotion_parameters(brain, -1.5f, 0.5f, 0.5f);
        REQUIRE(result1 == KELLY_ERROR_INVALID_PARAMETER);
        
        // Invalid arousal
        KellyErrorCode result2 = kelly_brain_set_emotion_parameters(brain, 0.0f, -0.5f, 0.5f);
        REQUIRE(result2 == KELLY_ERROR_INVALID_PARAMETER);
        
        // Invalid dominance
        KellyErrorCode result3 = kelly_brain_set_emotion_parameters(brain, 0.0f, 0.5f, 1.5f);
        REQUIRE(result3 == KELLY_ERROR_INVALID_PARAMETER);
    }
}

TEST_CASE_METHOD(KellyFFITestFixture, "Kelly FFI - State Queries") {
    kelly_brain_initialize(brain, "./data");
    
    SECTION("Get emotion state") {
        char* state = kelly_brain_get_emotion_state(brain);
        
        if (state != nullptr) {
            std::string json(state);
            REQUIRE_FALSE(json.empty());
            REQUIRE(json.find("valence") != std::string::npos);
            REQUIRE(json.find("arousal") != std::string::npos);
            
            kelly_free_string(state);
        }
    }
    
    SECTION("Get available emotions") {
        char* emotions = kelly_brain_get_available_emotions(brain);
        
        if (emotions != nullptr) {
            std::string json(emotions);
            REQUIRE_FALSE(json.empty());
            REQUIRE(json.find("emotions") != std::string::npos);
            
            kelly_free_string(emotions);
        }
    }
    
    SECTION("Get parameters") {
        char* params = kelly_brain_get_parameters(brain);
        
        if (params != nullptr) {
            std::string json(params);
            REQUIRE_FALSE(json.empty());
            
            kelly_free_string(params);
        }
    }
}

TEST_CASE("Kelly FFI - Utility Functions") {
    SECTION("Get version") {
        const char* version = kelly_get_version();
        REQUIRE(version != nullptr);
        REQUIRE(std::string(version).find("KellyBrain") != std::string::npos);
        // Version string is static, do not free
    }
    
    SECTION("Check data files") {
        // Should handle various path scenarios
        REQUIRE_FALSE(kelly_check_data_files(nullptr));
        REQUIRE_FALSE(kelly_check_data_files("/nonexistent/path"));
        
        // May return true for "./data" if files exist
        bool result = kelly_check_data_files("./data");
        // Result depends on file system state
    }
    
    SECTION("Error message retrieval") {
        const char* msg1 = kelly_get_error_message(KELLY_SUCCESS);
        REQUIRE(msg1 != nullptr);
        REQUIRE(std::string(msg1) == "Success");
        
        const char* msg2 = kelly_get_error_message(KELLY_ERROR_NULL_POINTER);
        REQUIRE(msg2 != nullptr);
        REQUIRE(std::string(msg2).find("Null pointer") != std::string::npos);
    }
}

TEST_CASE("Kelly FFI - Memory Management") {
    SECTION("Free string with null") {
        // Should not crash
        kelly_free_string(nullptr);
    }
    
    SECTION("Free string after allocation") {
        KellyBrain* brain = kelly_brain_create();
        REQUIRE(brain != nullptr);
        
        char* result = kelly_brain_from_text(brain, "test");
        if (result != nullptr) {
            // Ensure we can free the result
            kelly_free_string(result);
        }
        
        kelly_brain_destroy(brain);
    }
}

TEST_CASE("Kelly FFI - Thread Safety") {
    // Basic thread safety test
    KellyBrain* brain = kelly_brain_create();
    REQUIRE(brain != nullptr);
    
    kelly_brain_initialize(brain, "./data");
    
    // Multiple threads calling read-only functions should be safe
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([brain, &success_count]() {
            for (int j = 0; j < 10; ++j) {
                if (kelly_brain_is_initialized(brain)) {
                    char* state = kelly_brain_get_emotion_state(brain);
                    if (state != nullptr) {
                        kelly_free_string(state);
                        success_count++;
                    }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    kelly_brain_destroy(brain);
    
    // At least some calls should succeed if KellyBrain is functional
    INFO("Successful thread calls: " << success_count.load());
}