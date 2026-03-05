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
    
    // Determine build directories based on profile.
    // Support both multi-config (build/{debug,release}) and single-config (build).
    let cmake_build_dirs = if profile == "release" {
        vec!["../build/release", "../build"]
    } else {
        vec!["../build/debug", "../build"]
    };
    
    // Set library search paths
    for build_dir in &cmake_build_dirs {
        println!("cargo:rustc-link-search=native={}", build_dir);
    }
    
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
    let mut possible_lib_paths = Vec::new();
    for build_dir in &cmake_build_dirs {
        possible_lib_paths.push(format!("{}/libKellyFFI.1.dylib", build_dir)); // macOS (install name)
        possible_lib_paths.push(format!("{}/libKellyFFI.dylib", build_dir));   // macOS (unversioned symlink)
        possible_lib_paths.push(format!("{}/libKellyFFI.so", build_dir));      // Linux
        possible_lib_paths.push(format!("{}/KellyFFI.dll", build_dir));         // Windows
    }
    
    for lib_path in &possible_lib_paths {
        if std::path::Path::new(lib_path).exists() {
            ffi_lib_found = true;
            println!("cargo:warning=Found Kelly FFI library at: {}", lib_path);
            break;
        }
    }
    
    if !ffi_lib_found {
        let is_ci = env::var("CI").as_deref() == Ok("true");
        if is_ci {
            panic!(
                "Kelly FFI library not found. Expected one of: {:?}. Run CMake build for KellyFFI first (e.g. cmake -B build && cmake --build build --target KellyFFI).",
                possible_lib_paths
            );
        }
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

    // For macOS tests, Cargo test binaries resolve @rpath from target/<profile>.
    // Stage KellyFFI into those runtime locations using the install-name filename.
    if target_os == "macos" {
        if let Ok(out_dir) = env::var("OUT_DIR") {
            let out_dir = PathBuf::from(out_dir);
            if let Some(target_profile_dir) = out_dir.ancestors().nth(3) {
                let runtime_dirs = [
                    target_profile_dir.join("Resources"),
                    target_profile_dir.join("Frameworks"),
                    target_profile_dir.to_path_buf(),
                ];

                let source_lib = possible_lib_paths
                    .iter()
                    .find(|path| std::path::Path::new(path).exists());

                if let Some(source_lib) = source_lib {
                    for runtime_dir in &runtime_dirs {
                        if let Err(e) = std::fs::create_dir_all(runtime_dir) {
                            println!(
                                "cargo:warning=Failed to create runtime dir {}: {}",
                                runtime_dir.display(),
                                e
                            );
                            continue;
                        }

                        for lib_name in ["libKellyFFI.1.dylib", "libKellyFFI.dylib"] {
                            let dest_path = runtime_dir.join(lib_name);
                            if let Err(e) = std::fs::copy(source_lib, &dest_path) {
                                println!(
                                    "cargo:warning=Failed to stage {} in {}: {}",
                                    lib_name,
                                    runtime_dir.display(),
                                    e
                                );
                            } else {
                                println!(
                                    "cargo:warning=Staged {} in {}",
                                    lib_name,
                                    runtime_dir.display()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    
    println!("cargo:warning=Tauri build.rs configuration completed");
}
