# Python bindings using pybind11

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

pybind11_add_module(penta_core_native
    bindings.cpp
    harmony_bindings.cpp
    groove_bindings.cpp
    diagnostics_bindings.cpp
    osc_bindings.cpp
    ml_bindings.cpp
    prrot_bindings.cpp
)

target_link_libraries(penta_core_native PRIVATE
    penta_core
    juce::juce_osc
    juce::juce_dsp
    $<$<TARGET_EXISTS:KellyCore>:KellyCore>
)

# PRROT bindings need access to src/ headers (PRROTEngine, VoiceProfile, etc.)
target_include_directories(penta_core_native PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Set output directory for Python module
set_target_properties(penta_core_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

# Install Python module
install(TARGETS penta_core_native
    LIBRARY DESTINATION python
)
