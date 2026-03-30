"""
PRROT/PARROT Worker - Disposable Python Worker Process

Tier B worker that loads exactly one model per job, processes the job,
writes structured output, and exits fully so memory is reclaimed.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from .job_schema import (
    load_job_from_file,
    save_job_to_file,
    JobType,
    JobStatus,
    VoiceProfileExtractionJob,
    ControlDataGenerationJob,
    ArticulationAnalysisJob,
)
from .utils.memory_monitor import MemoryMonitor
from .utils.external_ssd import ExternalSSDManager
from .utils.process_manager import ProcessManager, ensure_single_worker
from .model_manager import ModelManager


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PRROTWorker:
    """Disposable worker process for PRROT/PARROT jobs"""

    def __init__(self, job_path: Path, external_ssd_path: Optional[Path] = None):
        """Initialize worker"""
        self.job_path = job_path
        self.ssd_manager = ExternalSSDManager(external_ssd_path)
        self.memory_monitor = MemoryMonitor()
        self.process_manager = ProcessManager()
        self.model_manager = ModelManager()

        # Job will be loaded in process()
        self.job = None
        self.model = None  # Model loaded per job (deprecated, use model_manager)

    def process(self) -> bool:
        """Process the job and exit"""
        # Note: Process lock is handled by @ensure_single_worker decorator
        # This method assumes lock is already acquired

        try:
            # System memory check is done by @ensure_single_worker decorator
            # Additional verification here
            has_memory, memory_msg = self.process_manager.check_system_memory()
            if not has_memory:
                logger.error(f"System memory check failed: {memory_msg}")
                self._mark_failed(memory_msg)
                return False

            # Load job
            logger.info(f"Loading job from {self.job_path}")
            self.job = load_job_from_file(self.job_path)

            # Update status
            self.job.status = JobStatus.PROCESSING.value
            self.job.created_at = datetime.now().isoformat()
            save_job_to_file(self.job, self.job_path)

            # Check memory before loading model
            within_limit, warning = self.memory_monitor.check_memory_limit()
            if not within_limit:
                error_msg = f"Memory limit check failed: {warning}"
                logger.error(error_msg)
                self._mark_failed(error_msg)
                return False

            if warning:
                logger.warning(f"Memory warning: {warning}")

            # Process job based on type
            success = False
            if self.job.job_type == JobType.EXTRACT_VOICE_PROFILE.value:
                success = self._process_voice_profile_extraction(self.job)
            elif self.job.job_type == JobType.GENERATE_CONTROL_DATA.value:
                success = self._process_control_data_generation(self.job)
            elif self.job.job_type == JobType.ANALYZE_ARTICULATION.value:
                success = self._process_articulation_analysis(self.job)
            else:
                error_msg = f"Unknown job type: {self.job.job_type}"
                logger.error(error_msg)
                self._mark_failed(error_msg)
                return False

            if success:
                self.job.status = JobStatus.COMPLETED.value
                self.job.completed_at = datetime.now().isoformat()
                save_job_to_file(self.job, self.job_path)
                logger.info("Job completed successfully")
                return True
            else:
                return False

        except Exception as e:
            logger.exception(f"Error processing job: {e}")
            self._mark_failed(str(e))
            return False

        finally:
            # Cleanup: unload all models and exit
            self.model_manager.unload_model()  # Unload all models
            self.model = None
            import gc

            gc.collect()

            # Note: Process lock released by @ensure_single_worker decorator
            logger.info("Worker cleanup complete - memory reclaimed")

    def _process_voice_profile_extraction(self, job: VoiceProfileExtractionJob) -> bool:
        """Process voice profile extraction job"""
        logger.info(f"Processing voice profile extraction for profile_id: {job.profile_id}")

        try:
            # Import here to avoid loading unless needed
            from .phoneme_aligner import PhonemeAligner
            from .articulation_analyzer import ArticulationAnalyzer
            from .timbre_embeddings import TimbreEmbeddingExtractor
            from .prosody_analyzer import ProsodyAnalyzer
            from .voice_profile import VoiceProfile

            # Load model (one model per job) - use model manager
            logger.info("Loading phoneme alignment model...")

            # Ensure only one model loaded
            model_key = "phoneme_aligner"
            self.model_manager.ensure_single_model(model_key)

            # Load model via model manager (with memory checks)
            model_path = self.ssd_manager.get_model_path("phoneme_aligner_q4.bin")
            model = self.model_manager.load_model(
                model_path, model_type="phoneme_aligner", quantization="Q4"
            )

            if model is None:
                raise RuntimeError("Failed to load phoneme alignment model (memory constraint?)")

            # Create aligner and set model
            aligner = PhonemeAligner()
            aligner.model = model  # Set loaded model (loaded via ModelManager)

            # Check memory after model load
            within_limit, warning = self.memory_monitor.check_memory_limit()
            if not within_limit:
                # Unload model before failing
                self.model_manager.unload_model()
                raise RuntimeError(f"Memory limit exceeded after model load: {warning}")

            # Load reference audio files
            audio_files = self.ssd_manager.list_reference_audio_files(job.profile_id)
            if not audio_files:
                raise ValueError(f"No reference audio files found for profile_id: {job.profile_id}")

            logger.info(f"Found {len(audio_files)} reference audio files")

            # Extract voice profile (placeholder - would call actual extraction)
            # This would involve:
            # 1. Deep phoneme alignment
            # 2. Speaker-specific articulation analysis
            # 3. Timbre embedding extraction
            # 4. Prosody analysis

            # For now, create a basic profile structure
            profile = VoiceProfile(
                profile_id=job.profile_id,
                profile_name=job.profile_name or job.profile_id,
                created_timestamp=int(datetime.now().timestamp()),
            )

            # Save profile
            profile_dir = self.ssd_manager.get_voice_profile_path(job.profile_id)
            profile_path = profile_dir / "profile.json"
            profile.save(profile_path)

            job.output_profile_path = str(profile_path)
            job.output_metadata_path = str(profile_dir / "metadata.json")

            logger.info(f"Voice profile saved to {profile_path}")
            return True

        except Exception as e:
            logger.exception(f"Error in voice profile extraction: {e}")
            return False

    def _process_control_data_generation(self, job: ControlDataGenerationJob) -> bool:
        """Process control data generation job"""
        logger.info(f"Processing control data generation for profile_id: {job.voice_profile_id}")

        try:
            # Import here to avoid loading unless needed
            from .lyric_planner import LyricPlanner
            from .voice_profile import VoiceProfile

            # Load voice profile
            if job.voice_profile_path:
                profile_path = Path(job.voice_profile_path)
            else:
                profile_path = (
                    self.ssd_manager.get_voice_profile_path(job.voice_profile_id) / "profile.json"
                )

            if not profile_path.exists():
                raise FileNotFoundError(f"Voice profile not found: {profile_path}")

            profile = VoiceProfile.load(profile_path)
            logger.info(f"Loaded voice profile: {profile.profile_id}")

            # Generate control data from lyrics
            planner = LyricPlanner(profile)
            control_data = planner.generate_control_data(
                lyrics=job.lyrics,
                melody_midi_notes=job.melody_midi_notes,
                melody_timing=job.melody_timing,
                tempo_bpm=job.tempo_bpm,
                expression_params=job.expression_params,
            )

            # Save output
            output_dir = self.ssd_manager.get_output_path(f"output_{job.job_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            control_data_path = output_dir / "phoneme_sequence.json"
            with open(control_data_path, "w") as f:
                json.dump(control_data.to_dict(), f, indent=2)

            job.output_control_data_path = str(control_data_path)
            job.output_midi_path = str(output_dir / "midi.mid")
            job.output_automation_path = str(output_dir / "automation.json")

            logger.info(f"Control data saved to {control_data_path}")
            return True

        except Exception as e:
            logger.exception(f"Error in control data generation: {e}")
            return False

    def _process_articulation_analysis(self, job: ArticulationAnalysisJob) -> bool:
        """Process articulation analysis job"""
        logger.info("Processing articulation analysis")

        try:
            # Import here
            from .articulation_inference import ArticulationInference
            from .instrument_affinity import InstrumentAffinityMapper

            # Analyze articulation
            inference = ArticulationInference()

            if job.descriptive_text:
                articulation_profile = inference.parse_descriptive_text(job.descriptive_text)
            elif job.audio_path:
                articulation_profile = inference.analyze_audio(Path(job.audio_path))
            else:
                raise ValueError("Either descriptive_text or audio_path must be provided")

            # Map to instrument affinity
            mapper = InstrumentAffinityMapper()
            instrument_affinity = mapper.map_articulation(articulation_profile)

            # Save output
            output_dir = (
                Path(job.output_articulation_profile_path).parent
                if job.output_articulation_profile_path
                else Path.cwd()
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            if not job.output_articulation_profile_path:
                job.output_articulation_profile_path = str(output_dir / "articulation_profile.json")
            if not job.output_instrument_affinity_path:
                job.output_instrument_affinity_path = str(output_dir / "instrument_affinity.json")

            with open(job.output_articulation_profile_path, "w") as f:
                json.dump(articulation_profile.to_dict(), f, indent=2)

            with open(job.output_instrument_affinity_path, "w") as f:
                json.dump(instrument_affinity.to_dict(), f, indent=2)

            logger.info("Articulation analysis completed")
            return True

        except Exception as e:
            logger.exception(f"Error in articulation analysis: {e}")
            return False

    def _mark_failed(self, error_message: str) -> None:
        """Mark job as failed"""
        if self.job:
            self.job.status = JobStatus.FAILED.value
            self.job.error_message = error_message
            self.job.completed_at = datetime.now().isoformat()
            save_job_to_file(self.job, self.job_path)


@ensure_single_worker
def _run_worker(job_path: Path, external_ssd_path: Optional[Path]) -> bool:
    """Run worker with single-worker enforcement"""
    worker = PRROTWorker(job_path, external_ssd_path)
    return worker.process()


def main():
    """Main entry point for worker process"""
    parser = argparse.ArgumentParser(description="PRROT/PARROT Worker")
    parser.add_argument("job_path", type=Path, help="Path to job JSON file")
    parser.add_argument("--external-ssd", type=Path, help="External SSD base path")
    args = parser.parse_args()

    if not args.job_path.exists():
        logger.error(f"Job file not found: {args.job_path}")
        sys.exit(1)

    # Run with single-worker enforcement
    success = _run_worker(args.job_path, args.external_ssd)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
