/**
 * PRROT/PARROT pybind11 bindings (Phase 4b).
 *
 * Exposes PRROTEngine and key data types to Python for voice analysis
 * and control data generation.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "prrot/PRROTEngine.h"
#include "prrot/VoiceProfile.h"
#include "prrot/PhonemeControlData.h"

namespace py = pybind11;
using namespace prrot;

void bind_prrot(py::module_& m) {
    auto prrot_m = m.def_submodule("prrot", "PRROT/PARROT voice engine bindings");

    // -------------------------------------------------------------------------
    // PhonemeType enum
    // -------------------------------------------------------------------------
    py::enum_<PhonemeType>(prrot_m, "PhonemeType")
        .value("AA", PhonemeType::AA)
        .value("AE", PhonemeType::AE)
        .value("AH", PhonemeType::AH)
        .value("AO", PhonemeType::AO)
        .value("AW", PhonemeType::AW)
        .value("AY", PhonemeType::AY)
        .value("EH", PhonemeType::EH)
        .value("ER", PhonemeType::ER)
        .value("EY", PhonemeType::EY)
        .value("IH", PhonemeType::IH)
        .value("IY", PhonemeType::IY)
        .value("OW", PhonemeType::OW)
        .value("OY", PhonemeType::OY)
        .value("UH", PhonemeType::UH)
        .value("UW", PhonemeType::UW)
        .value("B", PhonemeType::B)
        .value("D", PhonemeType::D)
        .value("G", PhonemeType::G)
        .value("K", PhonemeType::K)
        .value("P", PhonemeType::P)
        .value("T", PhonemeType::T)
        .value("CH", PhonemeType::CH)
        .value("DH", PhonemeType::DH)
        .value("F", PhonemeType::F)
        .value("HH", PhonemeType::HH)
        .value("JH", PhonemeType::JH)
        .value("S", PhonemeType::S)
        .value("SH", PhonemeType::SH)
        .value("TH", PhonemeType::TH)
        .value("V", PhonemeType::V)
        .value("Z", PhonemeType::Z)
        .value("ZH", PhonemeType::ZH)
        .value("M", PhonemeType::M)
        .value("N", PhonemeType::N)
        .value("NG", PhonemeType::NG)
        .value("L", PhonemeType::L)
        .value("R", PhonemeType::R)
        .value("W", PhonemeType::W)
        .value("Y", PhonemeType::Y)
        .value("SIL", PhonemeType::SIL)
        .value("UNKNOWN", PhonemeType::UNKNOWN);

    // -------------------------------------------------------------------------
    // PhonemeTiming
    // -------------------------------------------------------------------------
    py::class_<PhonemeTiming>(prrot_m, "PhonemeTiming")
        .def(py::init<>())
        .def_readwrite("phoneme", &PhonemeTiming::phoneme)
        .def_readwrite("start_time_ms", &PhonemeTiming::start_time_ms)
        .def_readwrite("duration_ms", &PhonemeTiming::duration_ms)
        .def_readwrite("is_stressed", &PhonemeTiming::is_stressed)
        .def_readwrite("stress_level", &PhonemeTiming::stress_level)
        .def_readwrite("prominence", &PhonemeTiming::prominence)
        .def("__repr__", [](const PhonemeTiming& pt) {
            return "PhonemeTiming(phoneme=" +
                   std::to_string(static_cast<int>(pt.phoneme)) +
                   ", start=" + std::to_string(pt.start_time_ms) +
                   "ms, dur=" + std::to_string(pt.duration_ms) + "ms)";
        });

    // -------------------------------------------------------------------------
    // PitchTarget
    // -------------------------------------------------------------------------
    py::class_<PitchTarget>(prrot_m, "PitchTarget")
        .def(py::init<>())
        .def_readwrite("time_ms", &PitchTarget::time_ms)
        .def_readwrite("midi_note", &PitchTarget::midi_note)
        .def_readwrite("cents_offset", &PitchTarget::cents_offset)
        .def_readwrite("confidence", &PitchTarget::confidence);

    // -------------------------------------------------------------------------
    // BreathMarker
    // -------------------------------------------------------------------------
    py::class_<BreathMarker>(prrot_m, "BreathMarker")
        .def(py::init<>())
        .def_readwrite("time_ms", &BreathMarker::time_ms)
        .def_readwrite("duration_ms", &BreathMarker::duration_ms)
        .def_readwrite("confidence", &BreathMarker::confidence);

    // -------------------------------------------------------------------------
    // VoiceProfile
    // -------------------------------------------------------------------------
    py::class_<VoiceProfile>(prrot_m, "VoiceProfile")
        .def(py::init<>())
        .def_readwrite("profile_id", &VoiceProfile::profile_id)
        .def_readwrite("profile_name", &VoiceProfile::profile_name)
        .def("is_valid", &VoiceProfile::isValid);

    // -------------------------------------------------------------------------
    // PhonemeControlData (output of processAudioSegment)
    // -------------------------------------------------------------------------
    py::class_<PhonemeControlData>(prrot_m, "PhonemeControlData")
        .def(py::init<>())
        .def_readonly("phoneme_sequence", &PhonemeControlData::phoneme_sequence)
        .def_readonly("pitch_targets", &PhonemeControlData::pitch_targets)
        .def("__repr__", [](const PhonemeControlData& cd) {
            return "PhonemeControlData(phonemes=" +
                   std::to_string(cd.phoneme_sequence.size()) +
                   ", pitch_targets=" +
                   std::to_string(cd.pitch_targets.size()) + ")";
        });

    // -------------------------------------------------------------------------
    // PRROTEngine
    // -------------------------------------------------------------------------
    py::class_<PRROTEngine>(prrot_m, "PRROTEngine")
        .def(py::init<>())
        .def("initialize", &PRROTEngine::initialize,
             "Initialize engine (call before use)")
        .def("load_voice_profile", &PRROTEngine::loadVoiceProfile,
             py::arg("profile"),
             "Load a voice profile (non-RT, call before processing)")
        .def("get_voice_profile", &PRROTEngine::getVoiceProfile,
             py::return_value_policy::reference_internal)
        .def("process_audio_segment",
             [](PRROTEngine& engine, py::array_t<float> audio,
                float sample_rate, float tempo_bpm) {
                 auto buf = audio.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("Audio must be a 1D float array");
                 }
                 // Release GIL during processing (RT-safe C++ code)
                 py::gil_scoped_release release;
                 return engine.processAudioSegment(
                     static_cast<const float*>(buf.ptr),
                     buf.shape[0],
                     sample_rate,
                     tempo_bpm
                 );
             },
             py::arg("audio"), py::arg("sample_rate") = 48000.0f,
             py::arg("tempo_bpm") = 120.0f,
             "Process audio segment to generate control data (RT-safe)")
        .def("analyze_phonemes",
             [](PRROTEngine& engine, py::array_t<float> audio,
                float sample_rate) {
                 auto buf = audio.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("Audio must be a 1D float array");
                 }
                 py::gil_scoped_release release;
                 return engine.analyzePhonemes(
                     static_cast<const float*>(buf.ptr),
                     buf.shape[0],
                     sample_rate
                 );
             },
             py::arg("audio"), py::arg("sample_rate") = 48000.0f,
             "Analyze audio and extract phoneme timing (RT-safe)")
        .def("detect_breath_markers",
             [](PRROTEngine& engine, py::array_t<float> audio,
                float sample_rate) {
                 auto buf = audio.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("Audio must be a 1D float array");
                 }
                 py::gil_scoped_release release;
                 return engine.detectBreathMarkers(
                     static_cast<const float*>(buf.ptr),
                     buf.shape[0],
                     sample_rate
                 );
             },
             py::arg("audio"), py::arg("sample_rate") = 48000.0f,
             "Detect breath markers in audio (RT-safe)")
        .def("generate_control_data", &PRROTEngine::generateControlData,
             py::arg("phoneme_sequence"), py::arg("pitch_targets"),
             py::arg("tempo_bpm") = 120.0f,
             "Generate control data from phoneme sequence and pitch targets");

    // Utility functions
    prrot_m.def("phoneme_to_string", &phonemeTypeToString,
                "Convert PhonemeType to string symbol");
    prrot_m.def("string_to_phoneme", &stringToPhonemeType,
                "Convert string symbol to PhonemeType");
    prrot_m.def("is_vowel", &isVowel, "Check if phoneme is a vowel");
    prrot_m.def("is_consonant", &isConsonant, "Check if phoneme is a consonant");
}
