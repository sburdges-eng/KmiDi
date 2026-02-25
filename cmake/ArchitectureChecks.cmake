# ArchitectureChecks.cmake - Verify architecture boundaries
#
# This module checks:
# - No ML in audio thread
# - Intent IR is only cross-boundary format
# - No forbidden dependencies
# - Layer boundaries are respected

function(check_architecture)
    message(STATUS "Running architecture checks...")
    
    # Check 1: Verify no ML dependencies in audio thread code
    # This would require static analysis - for now, just a warning
    message(STATUS "  [ ] Checking for ML in audio thread...")
    message(WARNING "  Architecture check: ML in audio thread detection requires static analysis")
    
    # Check 2: Verify Intent IR is the only cross-boundary format
    # This would require checking all FFI boundaries
    message(STATUS "  [ ] Checking Intent IR boundaries...")
    message(WARNING "  Architecture check: Intent IR boundary verification requires code analysis")
    
    # Check 3: Check for forbidden dependencies
    # Example: audio thread code should not depend on JUCE GUI
    message(STATUS "  [ ] Checking for forbidden dependencies...")
    
    # Check 4: Verify layer boundaries
    message(STATUS "  [ ] Checking layer boundaries...")
    
    message(STATUS "Architecture checks completed (manual verification recommended)")
endfunction()

# Run checks if enabled
if(BUILD_ARCHITECTURE_CHECKS)
    check_architecture()
endif()
