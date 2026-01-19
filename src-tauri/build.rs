use std::env;
use std::path::PathBuf;

fn main() {
    // Standard Tauri build setup
    tauri_build::build();
    
    // =============================================================================
    // KellyFFI Library Linking Configuration
    // =============================================================================
    
    println!("cargo:rerun-if-changed=../src/bridge/kelly_ffi.h");
    println!("cargo:rerun-if-changed=../src/bridge/kelly_ffi.cpp");
    println!("cargo:rerun-if-changed=../CMakeLists.txt");
    
    // Get build configuration
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    
    // Determine build directory based on profile
    let cmake_build_dir = if profile == "release" {
        "../build/release"
    } else {
        "../build/debug"
    };
    
    // Set library search path
    println!("cargo:rustc-link-search=native={}", cmake_build_dir);
    
    // Platform-specific library linking
    match target_os.as_str() {
        "macos" => {
            // Link Kelly FFI library
            println!("cargo:rustc-link-lib=dylib=KellyFFI");
            
            // macOS-specific frameworks that KellyCore might need
            println!("cargo:rustc-link-lib=framework=CoreAudio");
            println!("cargo:rustc-link-lib=framework=CoreMIDI");
            println!("cargo:rustc-link-lib=framework=AudioUnit");
            println!("cargo:rustc-link-lib=framework=AudioToolbox");
            println!("cargo:rustc-link-lib=framework=CoreFoundation");
            println!("cargo:rustc-link-lib=framework=AppKit");
            
            // Set RPATH for dynamic library loading
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../Resources");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../Frameworks");
        },
        "linux" => {
            // Link Kelly FFI library
            println!("cargo:rustc-link-lib=dylib=KellyFFI");
            
            // Linux-specific libraries
            println!("cargo:rustc-link-lib=asound");  // ALSA
            println!("cargo:rustc-link-lib=jack");     // JACK
            println!("cargo:rustc-link-lib=pthread");  // Threading
            
            // Set RPATH for dynamic library loading
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
        },
        "windows" => {
            // Link Kelly FFI library
            println!("cargo:rustc-link-lib=dylib=KellyFFI");
            
            // Windows-specific libraries
            println!("cargo:rustc-link-lib=winmm");
            println!("cargo:rustc-link-lib=ole32");
            println!("cargo:rustc-link-lib=user32");
        },
        _ => {
            println!("cargo:warning=Unknown target OS: {}", target_os);
        }
    }
    
    // =============================================================================
    // Environment Validation
    // =============================================================================
    
    // Check if Kelly FFI library exists
    let mut ffi_lib_found = false;
    let possible_lib_paths = [
        format!("{}/libKellyFFI.dylib", cmake_build_dir),    // macOS
        format!("{}/libKellyFFI.so", cmake_build_dir),       // Linux
        format!("{}/KellyFFI.dll", cmake_build_dir),         // Windows
    ];
    
    for lib_path in &possible_lib_paths {
        if std::path::Path::new(lib_path).exists() {
            ffi_lib_found = true;
            println!("cargo:warning=Found Kelly FFI library at: {}", lib_path);
            break;
        }
    }
    
    if !ffi_lib_found {
        println!("cargo:warning=Kelly FFI library not found. Build may fail.");
        println!("cargo:warning=Run CMake build first: cd ../build && cmake .. && make KellyFFI");
        println!("cargo:warning=Expected locations: {:?}", possible_lib_paths);
    }
    
    // =============================================================================
    // Development vs Release Configuration
    // =============================================================================
    
    if profile == "debug" {
        // Development configuration
        println!("cargo:rustc-cfg=debug_build");
        
        // Add debug symbols and linking
        println!("cargo:rustc-link-arg=-g");
        
        // Additional debug libraries if needed
        if target_os == "linux" {
            println!("cargo:rustc-link-lib=dl");  // Dynamic loading
        }
    } else {
        // Release configuration
        println!("cargo:rustc-cfg=release_build");
        
        // Optimization flags
        println!("cargo:rustc-link-arg=-O2");
        
        // Strip symbols for smaller binary
        if target_os != "windows" {
            println!("cargo:rustc-link-arg=-s");
        }
    }
    
    // =============================================================================
    // Include Path Configuration
    // =============================================================================
    
    // Add include paths for any C++ header access (if needed in future)
    println!("cargo:include=../src");
    println!("cargo:include=../include");
    
    // =============================================================================
    // Resource Management
    // =============================================================================
    
    // Ensure resources directory exists
    let resources_dir = PathBuf::from("resources");
    if !resources_dir.exists() {
        std::fs::create_dir_all(&resources_dir).unwrap_or_else(|e| {
            println!("cargo:warning=Failed to create resources directory: {}", e);
        });
    }
    
    // Copy FFI library to resources if it exists
    for lib_path in &possible_lib_paths {
        if std::path::Path::new(lib_path).exists() {
            let lib_name = std::path::Path::new(lib_path)
                .file_name()
                .unwrap()
                .to_string_lossy();
            
            let dest_path = resources_dir.join(&*lib_name);
            
            if let Err(e) = std::fs::copy(lib_path, &dest_path) {
                println!("cargo:warning=Failed to copy FFI library to resources: {}", e);
            } else {
                println!("cargo:warning=Copied FFI library to: {}", dest_path.display());
            }
        }
    }
    
    println!("cargo:warning=Tauri build.rs configuration completed");
}