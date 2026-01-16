import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from .audio_generation_engine import AudioGenerationEngine
from .image_generation_engine import ImageGenerationEngine
from .llm_reasoning_engine import LLMReasoningEngine
from .proposals import ProposalManager
from .phases import PhaseManager, format_phase_progress, get_next_actions
from .cpp_planner import CppTransitionPlanner, format_cpp_plan
from .ai_specializations import (
    get_capabilities,
    suggest_task_assignment,
    TaskType,
)
from .debug import get_debug, DebugCategory
from .models import AIAgent, PhaseStatus
from music_brain.session.intent_schema import CompleteSongIntent
from music_brain.tier1.midi_pipeline_wrapper import MIDIGenerationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)


class Orchestrator:
    def __init__(
        self,
        llm_model_path: str,
        output_dir: str = "./orchestrator_outputs",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_engine = ImageGenerationEngine(
            model_dir=str(self.output_dir / "stable_diffusion_v1_5")
        )
        self.audio_engine = AudioGenerationEngine(
            output_dir=str(self.output_dir / "audio_textures")
        )

        self.llm_engine = LLMReasoningEngine(
            model_path=llm_model_path,
            image_engine=self.image_engine,  # Pass image engine to LLM
            audio_engine=self.audio_engine,  # Pass audio engine to LLM
        )
        self.midi_pipeline = MIDIGenerationPipeline()

        self.resource_locks = {
            "llm": threading.Lock(),
            "midi_gen": threading.Lock(),
            "image_gen": threading.Lock(),
            "audio_gen": threading.Lock(),
        }

        logging.info("Orchestrator initialized.")

    def _acquire_resource(
        self,
        resource_name: str,
        timeout: float = 300,
    ) -> bool:
        logging.info(f"Attempting to acquire lock for {resource_name}...")
        if self.resource_locks[resource_name].acquire(timeout=timeout):
            logging.info(f"Lock acquired for {resource_name}.")
            return True
        logging.warning(
            "Failed to acquire lock for %s within %s seconds.",
            resource_name,
            timeout,
        )
        return False

    def _release_resource(self, resource_name: str):
        try:
            self.resource_locks[resource_name].release()
            logging.info("Lock released for %s.", resource_name)
        except RuntimeError:
            logging.warning(
                "Tried to release lock for %s but it was not held by this " "thread.",
                resource_name,
            )

    def execute_workflow(
        self,
        user_intent_text: str,
        enable_image_gen: bool = True,
        enable_audio_gen: bool = False,
    ) -> CompleteSongIntent:
        logging.info(
            "Starting workflow for intent: '%s'",
            user_intent_text,
        )
        start_time = time.time()

        # Phase 1: LLM Reasoning (Intent Parsing, Prompt Expansion)
        if not self._acquire_resource("llm"):
            raise RuntimeError("Could not acquire LLM resource.")
        try:
            structured_intent = self.llm_engine.parse_user_intent(
                user_intent_text,
            )
            structured_intent = self.llm_engine.generate_image_prompts(
                structured_intent,
            )
            if enable_audio_gen:
                structured_intent = self.llm_engine.generate_audio_texture_prompt(
                    structured_intent,
                )
            logging.info("LLM reasoning complete.")
        finally:
            self._release_resource("llm")

        # Convert StructuredIntent to CompleteSongIntent for MIDI pipeline
        # compatibility. Map all required fields with defaults.
        complete_intent = CompleteSongIntent(
            core_event=structured_intent.core_event or "",
            core_resistance=structured_intent.core_resistance or "",
            core_longing=structured_intent.core_longing or "",
            core_stakes=structured_intent.core_stakes or "",
            core_transformation=structured_intent.core_transformation or "",
            mood_primary=structured_intent.mood_primary or "",
            mood_secondary_tension=(structured_intent.mood_secondary_tension or "0.5"),
            imagery_texture=structured_intent.imagery_texture or "",
            vulnerability_scale=(structured_intent.vulnerability_scale or "Medium"),
            narrative_arc=structured_intent.narrative_arc or "",
            technical_genre=structured_intent.technical_genre or "",
            technical_tempo_range=(80, 120),  # Default tempo range
            technical_key=structured_intent.technical_key or "",
            technical_mode=structured_intent.technical_mode or "",
            technical_groove_feel=(structured_intent.technical_groove_feel or ""),
            technical_rule_to_break=(structured_intent.technical_rule_to_break or ""),
            rule_breaking_justification=(structured_intent.rule_breaking_justification or ""),
            output_target="",  # Not in StructuredIntent, use default
            output_feedback_loop="",  # Not in StructuredIntent, use default
            title="",  # Not in StructuredIntent, use default
            created="",  # Not in StructuredIntent, use default
            midi_plan=structured_intent.midi_plan,
            image_prompt=structured_intent.image_prompt,
            image_style_constraints=structured_intent.image_style_constraints,
            audio_texture_prompt=structured_intent.audio_texture_prompt,
            explanation=structured_intent.explanation,
            rule_breaking_logic=structured_intent.rule_breaking_logic,
        )

        # Phase 2: MIDI Generation
        # MIDI generation is fast and deterministic, so it might not need a
        # separate lock if it doesn't contend with LLM. For safety and
        # consistency in a multi-resource environment, include it.
        if not self._acquire_resource("midi_gen"):
            raise RuntimeError("Could not acquire MIDI resource.")
        try:
            midi_result = self.midi_pipeline.generate_midi(
                complete_intent,
                output_dir=str(self.output_dir / "midi_outputs"),
            )
            if midi_result and midi_result.get("status") == "completed":
                complete_intent.midi_plan = midi_result
                logging.info("MIDI generation complete.")
            else:
                error_msg = (
                    midi_result.get("details", "Unknown error")
                    if midi_result
                    else "No result returned"
                )
                logging.error("MIDI generation failed: %s", error_msg)
                complete_intent.midi_plan = {
                    "status": "failed",
                    "details": error_msg,
                }
        except Exception as e:
            logging.error(
                "MIDI generation exception: %s",
                e,
                exc_info=True,
            )
            complete_intent.midi_plan = {
                "status": "error",
                "details": f"MIDI generation exception: {e}",
            }
        finally:
            self._release_resource("midi_gen")

        # Phase 3: Image Generation (Optional)
        if enable_image_gen and structured_intent.image_prompt:
            # The LLMEngine already holds a reference to image_engine, so its
            # internal lock is used. Orchestrator handles top-level resource
            # scheduling.
            if not self._acquire_resource("image_gen"):
                logging.warning("Could not acquire Image Generation resource.")
                complete_intent.generated_image_data = {
                    "status": "skipped",
                    "details": "Could not acquire image generation resource.",
                }
            else:
                try:
                    structured_intent = self.llm_engine.generate_image_from_intent(
                        structured_intent,
                    )
                    complete_intent.generated_image_data = structured_intent.generated_image_data
                    if complete_intent.generated_image_data:
                        status = complete_intent.generated_image_data.get(
                            "status",
                            "unknown",
                        )
                        if status == "completed":
                            logging.info("Image generation complete.")
                        else:
                            logging.warning(
                                "Image generation status: %s",
                                status,
                            )
                except Exception as e:
                    logging.error(
                        "Image generation exception: %s",
                        e,
                        exc_info=True,
                    )
                    complete_intent.generated_image_data = {
                        "status": "error",
                        "details": f"Image generation exception: {e}",
                    }
                finally:
                    self._release_resource("image_gen")

        # Phase 4: Audio Generation (Optional)
        if enable_audio_gen and structured_intent.audio_texture_prompt:
            # The LLMEngine already holds a reference to audio_engine, so its
            # internal lock is used. Orchestrator handles top-level resource
            # scheduling. The orchestrator's resource_lock["audio_gen"]
            # ensures only one audio job runs at a time.
            if not self._acquire_resource("audio_gen"):
                logging.warning("Could not acquire Audio Generation resource.")
                complete_intent.generated_audio_data = {
                    "status": "skipped",
                    "details": "Could not acquire audio generation resource.",
                }
            else:
                try:
                    structured_intent = self.llm_engine.generate_audio_from_intent(
                        structured_intent,
                    )
                    complete_intent.generated_audio_data = structured_intent.generated_audio_data
                    if complete_intent.generated_audio_data:
                        status = complete_intent.generated_audio_data.get(
                            "status",
                            "unknown",
                        )
                        if status == "completed":
                            logging.info("Audio generation complete.")
                        else:
                            logging.warning(
                                "Audio generation status: %s",
                                status,
                            )
                except Exception as e:
                    logging.error(
                        "Audio generation exception: %s",
                        e,
                        exc_info=True,
                    )
                    complete_intent.generated_audio_data = {
                        "status": "error",
                        "details": f"Audio generation exception: {e}",
                    }
                finally:
                    self._release_resource("audio_gen")

        end_time = time.time()
        logging.info(
            "Workflow completed in %.2f seconds.",
            end_time - start_time,
        )
        return complete_intent


class Workstation:
    """Multi-AI workstation facade for CLI and MCP server usage."""

    def __init__(
        self,
        llm_model_path: Optional[str] = None,
        output_dir: str = "./orchestrator_outputs",
    ):
        self._debug = get_debug()
        self.proposals = ProposalManager()
        self.phases = PhaseManager()
        self.cpp_planner = CppTransitionPlanner()
        self.active_agents: List[AIAgent] = []
        self.orchestrator: Optional[Orchestrator] = None

        if llm_model_path:
            self.orchestrator = Orchestrator(
                llm_model_path=llm_model_path,
                output_dir=output_dir,
            )

    def register_agent(self, agent: AIAgent) -> None:
        if agent not in self.active_agents:
            self.active_agents.append(agent)
        self._debug.info(
            DebugCategory.AI_COMMUNICATION,
            f"Registered agent {agent.value}",
        )

    def get_agent_capabilities(self, agent: AIAgent) -> Dict[str, Any]:
        caps = get_capabilities(agent)
        return {
            "agent": agent.value,
            "display_name": caps.display_name,
            "description": caps.description,
            "strengths": {t.value: s for t, s in caps.strengths.items()},
            "proposal_categories": [c.value for c in caps.proposal_categories],
            "best_languages": caps.best_languages,
            "special_abilities": caps.special_abilities,
            "limitations": caps.limitations,
            "recommended_for": caps.recommended_for,
        }

    def submit_proposal(self, **kwargs) -> Optional[Dict[str, Any]]:
        proposal = self.proposals.submit_proposal(**kwargs)
        return proposal.to_dict() if proposal else None

    def vote_on_proposal(self, **kwargs) -> bool:
        return self.proposals.vote_on_proposal(**kwargs)

    def get_all_proposals(self) -> Dict[str, Any]:
        return {
            "proposals": [p.to_dict() for p in self.proposals.get_all_proposals()],
            "summary": self.proposals.get_proposal_summary(),
        }

    def get_proposals_for_agent(self, agent: AIAgent) -> Dict[str, Any]:
        pending = self.proposals.get_pending_votes(agent)
        slots = self.proposals.get_agent_proposal_slots()
        return {
            "pending_votes": [p.to_dict() for p in pending],
            "slots_remaining": slots.get(agent, 0),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_agents": [a.value for a in self.active_agents],
            "proposal_summary": self.proposals.get_proposal_summary(),
            "phase_summary": self.phases.get_phase_summary(),
            "cpp_summary": self.cpp_planner.get_progress_summary(),
            "next_actions": get_next_actions(self.phases),
        }

    def get_dashboard(self) -> str:
        sections = [
            format_phase_progress(self.phases),
            "",
            format_cpp_plan(self.cpp_planner),
        ]
        summary = self.proposals.get_proposal_summary()
        sections.append(
            "\n".join(
                [
                    "",
                    "PROPOSALS SUMMARY",
                    f"Total: {summary['total']}",
                    f"By Status: {summary['by_status']}",
                    f"By Agent: {summary['by_agent']}",
                ]
            )
        )
        return "\n".join(sections)

    def get_phase_progress(self) -> str:
        return format_phase_progress(self.phases)

    def get_current_phase(self) -> Dict[str, Any]:
        phase = self.phases.get_current_phase()
        return phase.to_dict() if phase else {}

    def update_task(
        self,
        phase_id: int,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        self.phases.update_task_status(
            phase_id=phase_id,
            task_id=task_id,
            status=PhaseStatus(status),
            progress=progress,
            notes=notes,
        )

    def assign_task_to_agent(
        self,
        phase_id: int,
        task_id: str,
        agent: AIAgent,
    ) -> None:
        self.phases.assign_task(
            phase_id=phase_id,
            task_id=task_id,
            agent=agent,
        )

    def get_cpp_plan(self) -> Dict[str, Any]:
        return self.cpp_planner.to_dict()

    def get_cpp_progress(self) -> str:
        return format_cpp_plan(self.cpp_planner)

    def start_cpp_module(
        self,
        module_id: str,
        agent: Optional[AIAgent] = None,
    ) -> None:
        self.cpp_planner.start_module(module_id, agent)

    def update_cpp_module(
        self,
        module_id: str,
        progress: float,
        status: Optional[str] = None,
    ) -> None:
        parsed_status = PhaseStatus(status) if status else None
        self.cpp_planner.update_module_progress(
            module_id=module_id,
            progress=progress,
            status=parsed_status,
        )

    def get_cmake_plan(self) -> str:
        return self.cpp_planner.get_build_plan()

    def suggest_assignments(
        self,
        tasks: List[Tuple[str, TaskType]],
    ) -> Dict[str, str]:
        assignments = suggest_task_assignment(tasks)
        return {name: agent.value for name, agent in assignments.items()}

    def get_agent_workload(self) -> Dict[str, int]:
        workload = {agent.value: 0 for agent in AIAgent}
        for phase in self.phases.phases:
            for task in phase.tasks:
                if task.assigned_to:
                    workload[task.assigned_to.value] += 1
        for cpp_task in self.cpp_planner.tasks.values():
            if cpp_task.assigned_to:
                workload[cpp_task.assigned_to.value] += 1
        return workload

    def get_debug_summary(self) -> Dict[str, Any]:
        errors = self._debug.get_errors(limit=25)
        return {
            "recent_errors": [
                {
                    "timestamp": e.timestamp,
                    "message": e.message,
                    "category": e.category.value,
                }
                for e in errors
            ],
            "performance": self._debug.get_performance_report(),
            "event_count": len(self._debug.events),
        }

    def reset(self) -> None:
        self.proposals = ProposalManager()
        self.phases = PhaseManager()
        self.cpp_planner = CppTransitionPlanner()
        self.active_agents = []
        self._debug.clear()


_workstation_instance: Optional[Workstation] = None
_workstation_init_args: Optional[tuple] = None
_workstation_init_kwargs: Optional[dict] = None


def get_workstation(*args, **kwargs) -> Workstation:
    """Singleton accessor for workstation state."""
    global _workstation_instance
    global _workstation_init_args
    global _workstation_init_kwargs
    if _workstation_instance is None:
        _workstation_instance = Workstation(*args, **kwargs)
        _workstation_init_args = args
        _workstation_init_kwargs = dict(kwargs)
        return _workstation_instance
    if args or kwargs:
        if args != (_workstation_init_args or ()) or dict(kwargs) != (
            _workstation_init_kwargs or {}
        ):
            _workstation_instance = Workstation(*args, **kwargs)
            _workstation_init_args = args
            _workstation_init_kwargs = dict(kwargs)
    return _workstation_instance


def shutdown_workstation() -> None:
    """Placeholder to mirror previous API."""
    global _workstation_instance
    global _workstation_init_args
    global _workstation_init_kwargs
    _workstation_instance = None
    _workstation_init_args = None
    _workstation_init_kwargs = None


def main():
    parser = argparse.ArgumentParser(
        description="KmiDi Local Metal AI Orchestrator CLI",
    )
    parser.add_argument(
        "--llm_model_path",
        type=str,
        required=True,
        help="Path to the Mistral 7B GGUF model file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Natural language user intent.",
    )
    parser.add_argument(
        "--no_image_gen",
        action="store_true",
        help="Disable image generation.",
    )
    parser.add_argument(
        "--enable_audio_gen",
        action="store_true",
        help="Enable optional audio texture generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./orchestrator_outputs",
        help="Directory for all generated outputs.",
    )

    args = parser.parse_args()

    orchestrator = Orchestrator(
        llm_model_path=args.llm_model_path,
        output_dir=args.output_dir,
    )

    try:
        final_intent = orchestrator.execute_workflow(
            user_intent_text=args.prompt,
            enable_image_gen=not args.no_image_gen,
            enable_audio_gen=args.enable_audio_gen,
        )
        print("\n--- Workflow Results ---")
        print(f"User Intent: {args.prompt}")
        midi_status = (
            final_intent.midi_plan.get("status", "N/A")
            if isinstance(final_intent.midi_plan, dict)
            else "N/A"
        )
        image_status = (
            final_intent.generated_image_data.get("status", "disabled")
            if isinstance(final_intent.generated_image_data, dict)
            else "disabled"
        )
        audio_status = (
            final_intent.generated_audio_data.get("status", "disabled")
            if isinstance(final_intent.generated_audio_data, dict)
            else "disabled"
        )
        print(f"Generated MIDI Plan Status: {midi_status}")
        print(f"Generated Image Status: {image_status}")
        print(f"Generated Audio Status: {audio_status}")
        print(f"Explanation: {final_intent.explanation}")

        # Optionally save the full intent object to JSON for inspection
        intent_output_path = Path(args.output_dir) / "final_intent.json"
        final_intent.save(str(intent_output_path))
        print(f"Full intent saved to: {intent_output_path}")

    except RuntimeError as e:
        logging.error(f"Workflow failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
