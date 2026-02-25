/**
 * @file StemExporter.h
 * @brief Multi-track stem export functionality
 *
 * Provides stem export capabilities:
 * - Export individual tracks as separate audio files
 * - Multi-track bounce to stems
 * - Metadata support (track names, timestamps)
 * - Batch export with naming conventions
 */

#pragma once

#include "daiw/audio/AudioFile.h"
#include "daiw/project/ProjectFile.h"
#include <string>
#include <vector>
#include <functional>

namespace daiw {
namespace export_ns {  // Using export_ns to avoid 'export' keyword conflict

/**
 * @brief Export format options
 */
struct ExportOptions {
    audio::AudioFormat format = audio::AudioFormat::WAV;
    audio::SampleFormat sampleFormat = audio::SampleFormat::Float32;
    SampleRate sampleRate = DEFAULT_SAMPLE_RATE;
    bool normalizeStems = false;    // Normalize each stem to -0.1dBFS
    bool includeMetadata = true;    // Include track name in file metadata
    std::string filenameSuffix;     // Optional suffix for filenames
};

/**
 * @brief Stem export result
 */
struct StemExportResult {
    std::string trackName;
    std::string filepath;
    bool success = false;
    std::string errorMessage;
    uint64_t numSamples = 0;
    double durationSeconds = 0.0;
};

/**
 * @brief Progress callback for export operations
 * @param currentTrack Current track being exported (0-based)
 * @param totalTracks Total number of tracks to export
 * @param trackName Name of current track
 */
using ExportProgressCallback = std::function<void(size_t currentTrack, 
                                                  size_t totalTracks,
                                                  const std::string& trackName)>;

/**
 * @brief Multi-track stem exporter
 */
class StemExporter {
public:
    /**
     * @brief Default constructor
     */
    StemExporter() = default;

    /**
     * @brief Export all tracks from a project as individual stems
     * @param project Project to export
     * @param outputDirectory Directory to write stems to
     * @param options Export options
     * @return Vector of export results, one per track
     */
    std::vector<StemExportResult> exportAllStems(
        const project::ProjectFile& project,
        const std::string& outputDirectory,
        const ExportOptions& options = ExportOptions());

    /**
     * @brief Export specific tracks as stems
     * @param project Project containing tracks
     * @param trackIndices Indices of tracks to export
     * @param outputDirectory Directory to write stems to
     * @param options Export options
     * @return Vector of export results
     */
    std::vector<StemExportResult> exportSelectedStems(
        const project::ProjectFile& project,
        const std::vector<size_t>& trackIndices,
        const std::string& outputDirectory,
        const ExportOptions& options = ExportOptions());

    /**
     * @brief Export a single track
     * @param track Track to export
     * @param filepath Output file path
     * @param options Export options
     * @return Export result
     */
    StemExportResult exportTrack(
        const project::Track& track,
        const std::string& filepath,
        const ExportOptions& options = ExportOptions());

    /**
     * @brief Set progress callback
     */
    void setProgressCallback(ExportProgressCallback callback) {
        progressCallback_ = callback;
    }

    /**
     * @brief Render MIDI track to audio (stub)
     * @param track MIDI track to render
     * @param durationSeconds Duration to render
     * @param sampleRate Output sample rate
     * @return Rendered audio file
     *
     * Basic MIDI rendering implemented - for production, integrate full synth/sampler engine
     */
    static audio::AudioFile renderMidiTrack(
        const project::Track& track,
        double durationSeconds,
        SampleRate sampleRate = DEFAULT_SAMPLE_RATE);

    /**
     * @brief Normalize audio to target level
     * @param audio Audio file to normalize (modified in place)
     * @param targetLevel Target peak level (0.0 to 1.0)
     */
    static void normalizeAudio(audio::AudioFile& audio, float targetLevel = 0.9f);

    /**
     * @brief Generate stem filename from track
     * @param trackName Track name
     * @param trackIndex Track index
     * @param outputDirectory Output directory
     * @param format Output format
     * @param suffix Optional suffix
     * @return Full filepath for stem
     */
    static std::string generateStemFilename(
        const std::string& trackName,
        size_t trackIndex,
        const std::string& outputDirectory,
        audio::AudioFormat format,
        const std::string& suffix = "");

private:
    ExportProgressCallback progressCallback_;

    // Helper: Sanitize filename (remove invalid characters)
    static std::string sanitizeFilename(const std::string& name);
    
    // Helper: Get file extension for format
    static std::string getFileExtension(audio::AudioFormat format);
};

} // namespace export_ns
} // namespace daiw
