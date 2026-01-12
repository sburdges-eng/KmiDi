import unittest
import os
from pathlib import Path
import json
from unittest.mock import MagicMock, patch

# Assuming the project root is accessible
import sys
sys.path.append(str(Path(__file__).parents[3])) # Adjust path to access KmiDi_PROJECT

from KmiDi_PROJECT.source.python.mcp_workstation.orchestrator import Orchestrator
from KmiDi_PROJECT.source.python.mcp_workstation.llm_reasoning_engine import StructuredIntent
from KmiDi_PROJECT.source.python.music_brain.session.intent_schema import CompleteSongIntent

class TestOrchestratorIntegration(unittest.TestCase):

    def setUp(self):
        self.llm_model_path = "./models/mistral-7b-q4_k_m.gguf"  # Placeholder, won't actually load
        self.output_dir = Path("./test_orchestrator_outputs")
        self.output_dir.mkdir(exist_ok=True)

        # Mock external dependencies for isolated testing
        self.mock_llm_instance = MagicMock()
        self.mock_image_engine_instance = MagicMock()
        self.mock_audio_engine_instance = MagicMock()
        self.mock_midi_pipeline_instance = MagicMock()

        # Configure mocks to return predictable values
        self.mock_llm_instance.create_completion.side_effect = self._mock_llm_completion

        self.mock_image_engine_instance.generate_image.return_value = {
            "status": "completed",
            "image_data_base64": "dummy_image_data",
            "details": "mock image"
        }
        self.mock_audio_engine_instance.generate_audio_texture.return_value = {
            "status": "completed",
            "audio_data_base64": "dummy_audio_data",
            "details": "mock audio"
        }
        self.mock_audio_engine_instance.acquire_lock.return_value = True
        self.mock_audio_engine_instance.release_lock.return_value = None

        self.mock_midi_pipeline_instance.generate_midi.return_value = {
            "status": "completed",
            "file_path": str(self.output_dir / "mock_midi.mid"),
            "midi_data_base64": "dummy_midi_data",
            "tempo": 120,
            "key": "C Major",
            "mood": "Joy",
            "duration_bars": 4,
            "details": "mock midi"
        }

        # Patch the dependencies within Orchestrator and LLMReasoningEngine
        patcher_llm_constructor = patch('KmiDi_PROJECT.source.python.mcp_workstation.llm_reasoning_engine.Llama', return_value=self.mock_llm_instance)
        patcher_image_engine = patch('KmiDi_PROJECT.source.python.mcp_workstation.orchestrator.ImageGenerationEngine', return_value=self.mock_image_engine_instance)
        patcher_audio_engine = patch('KmiDi_PROJECT.source.python.mcp_workstation.orchestrator.AudioGenerationEngine', return_value=self.mock_audio_engine_instance)
        patcher_midi_pipeline = patch('KmiDi_PROJECT.source.python.mcp_workstation.orchestrator.MIDIGenerationPipeline', return_value=self.mock_midi_pipeline_instance)

        self.mock_llm_llama = patcher_llm_constructor.start()
        self.mock_image_engine = patcher_image_engine.start()
        self.mock_audio_engine = patcher_audio_engine.start()
        self.mock_midi_pipeline = patcher_midi_pipeline.start()

        self.addCleanup(patcher_llm_constructor.stop)
        self.addCleanup(patcher_image_engine.stop)
        self.addCleanup(patcher_audio_engine.stop)
        self.addCleanup(patcher_midi_pipeline.stop)

        self.orchestrator = Orchestrator(llm_model_path=self.llm_model_path, output_dir=str(self.output_dir))

    def tearDown(self):
        # Clean up created files/directories
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)

    def _mock_llm_completion(self, prompt: str, **kwargs):
        if "Parse the following natural language input" in prompt:
            return {"choices": [{"text": json.dumps({
                "core_event": "a feeling of joy",
                "mood_primary": "Joy",
                "technical_genre": "Synthwave",
                "image_prompt": "a neon-lit cityscape",
                "image_style_constraints": "80s aesthetic, digital painting",
                "audio_texture_prompt": "upbeat synth arpeggio"
            })}]}
        elif "expand it into a detailed MIDI plan" in prompt:
            return {"choices": [{"text": json.dumps({"tempo": 120, "key": "C Major", "chords": ["C", "G", "Am", "F"]})}]}
        elif "generate an image prompt and style constraints" in prompt:
            return {"choices": [{"text": "Image Prompt: a futuristic car on a highway\nStyle Constraints: cyberpunk, digital art"}]}
        elif "generate an audio texture prompt" in prompt:
            return {"choices": [{"text": "Audio Texture Prompt: a shimmering, ethereal pad"}]}
        elif "Explain the reasoning behind the generated" in prompt:
            return {"choices": [{"text": "Mock explanation for the generated content."}]}
        return {"choices": [{"text": "default LLM response"}]}

    def test_end_to_end_workflow_image_enabled(self):
        user_prompt = "Generate a joyful synthwave track with a neon cityscape image."
        final_intent = self.orchestrator.execute_workflow(user_prompt, enable_image_gen=True, enable_audio_gen=False)

        self.assertIsNotNone(final_intent)
        self.assertEqual(final_intent.song_intent.mood_primary, "Joy")
        self.assertEqual(final_intent.technical_constraints.technical_genre, "Synthwave")
        self.assertIn("neon-lit cityscape", final_intent.image_prompt)
        self.assertIsNotNone(final_intent.midi_plan)
        self.assertEqual(final_intent.midi_plan["status"], "completed")
        self.assertIsNotNone(final_intent.generated_image_data)
        self.assertEqual(final_intent.generated_image_data["status"], "completed")
        self.assertIn("dummy_image_data", final_intent.generated_image_data["image_data_base64"])
        self.assertIsNone(final_intent.generated_audio_data)

    def test_end_to_end_workflow_audio_enabled(self):
        user_prompt = "Create a melancholic track with a shimmering pad audio texture."
        final_intent = self.orchestrator.execute_workflow(user_prompt, enable_image_gen=False, enable_audio_gen=True)

        self.assertIsNotNone(final_intent)
        # Note: The mock LLM completion for parsing intent doesn't fully reflect the melancholic prompt
        # as it's a fixed mock, but the audio generation part should be triggered.
        self.assertIsNotNone(final_intent.audio_texture_prompt)
        self.assertIsNotNone(final_intent.midi_plan)
        self.assertEqual(final_intent.midi_plan["status"], "completed")
        self.assertIsNone(final_intent.generated_image_data)
        self.assertIsNotNone(final_intent.generated_audio_data)
        self.assertEqual(final_intent.generated_audio_data["status"], "completed")
        self.assertIn("dummy_audio_data", final_intent.generated_audio_data["audio_data_base64"])

    def test_end_to_end_workflow_all_disabled(self):
        user_prompt = "Just a simple track."
        final_intent = self.orchestrator.execute_workflow(user_prompt, enable_image_gen=False, enable_audio_gen=False)

        self.assertIsNotNone(final_intent)
        self.assertIsNotNone(final_intent.midi_plan)
        self.assertEqual(final_intent.midi_plan["status"], "completed")
        self.assertIsNone(final_intent.generated_image_data)
        self.assertIsNone(final_intent.generated_audio_data)

if __name__ == '__main__':
    unittest.main()