#!/usr/bin/env python3
"""
AI-Powered Training Orchestrator
=================================

Hybrid workflow where:
- OpenAI GPT handles: reasoning, planning, analysis, hyperparameter suggestions
- Local/Codespace handles: MIDI/audio model training execution

The LLM "thinks" about training strategy while we execute the actual training.

Usage:
    python scripts/ai_training_orchestrator.py --analyze-datasets
    python scripts/ai_training_orchestrator.py --plan-training
    python scripts/ai_training_orchestrator.py --train-with-guidance
    python scripts/ai_training_orchestrator.py --evaluate-results
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy imports for faster startup
def get_openai_client():
    """Get OpenAI client with API key from environment."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    return OpenAI(api_key=api_key)


# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = Path(os.environ.get("KELLY_AUDIO_DATA_ROOT", "/Volumes/sbdrive/audio/datasets"))
LOGS_DIR = ROOT / "logs" / "ai_orchestrator"

# Model registry for Kelly's 5-model architecture
MODEL_REGISTRY = {
    "emotion_recognizer": {
        "task": "Audio emotion classification",
        "architecture": "CNN",
        "input": "Mel spectrograms from audio",
        "output": "7 emotion classes",
        "dataset": "m4singer",
    },
    "melody_transformer": {
        "task": "Melodic sequence generation",
        "architecture": "LSTM",
        "input": "MIDI note sequences",
        "output": "Next note predictions",
        "dataset": "lakh_midi",
    },
    "harmony_predictor": {
        "task": "Chord progression prediction",
        "architecture": "MLP",
        "input": "Current harmonic context",
        "output": "48 chord types",
        "dataset": "lakh_midi",
    },
    "dynamics_engine": {
        "task": "Expression parameter mapping",
        "architecture": "MLP",
        "input": "Performance features",
        "output": "Velocity/expression values",
        "dataset": "m4singer",
    },
    "groove_predictor": {
        "task": "Timing/groove pattern prediction",
        "architecture": "LSTM",
        "input": "Rhythmic patterns",
        "output": "Micro-timing offsets",
        "dataset": "lakh_midi",
    },
}


@dataclass
class TrainingPlan:
    """AI-generated training plan."""
    model_name: str
    suggested_epochs: int
    suggested_batch_size: int
    suggested_lr: float
    reasoning: str
    priority: int
    estimated_time: str
    preprocessing_steps: List[str]
    augmentation_suggestions: List[str]
    evaluation_metrics: List[str]


@dataclass
class DatasetAnalysis:
    """AI analysis of dataset characteristics."""
    dataset_name: str
    total_samples: int
    analysis: str
    quality_issues: List[str]
    preprocessing_recommendations: List[str]
    augmentation_opportunities: List[str]


def analyze_datasets_with_ai() -> List[DatasetAnalysis]:
    """Use OpenAI to analyze available datasets and provide insights."""
    client = get_openai_client()

    print("\n" + "="*60)
    print("AI Dataset Analysis")
    print("="*60)

    # Gather dataset info
    datasets_info = {}

    if DATA_DIR.exists():
        for dataset_path in DATA_DIR.iterdir():
            if dataset_path.is_dir() and not dataset_path.name.startswith('.'):
                # Count files
                file_count = sum(1 for _ in dataset_path.rglob("*") if _.is_file())

                # Get size
                try:
                    import subprocess
                    result = subprocess.run(
                        ["du", "-sh", str(dataset_path)],
                        capture_output=True, text=True
                    )
                    size = result.stdout.split()[0] if result.returncode == 0 else "unknown"
                except (subprocess.SubprocessError, OSError, IndexError):
                    size = "unknown"

                # Sample file types
                extensions = {}
                for f in list(dataset_path.rglob("*"))[:1000]:
                    if f.is_file():
                        ext = f.suffix.lower()
                        extensions[ext] = extensions.get(ext, 0) + 1

                datasets_info[dataset_path.name] = {
                    "path": str(dataset_path),
                    "file_count": file_count,
                    "size": size,
                    "file_types": dict(sorted(extensions.items(), key=lambda x: -x[1])[:10])
                }

    if not datasets_info:
        print("No datasets found at", DATA_DIR)
        return []

    # Send to OpenAI for analysis
    prompt = f"""Analyze these music/audio datasets for ML training:

{json.dumps(datasets_info, indent=2)}

For each dataset, provide:
1. What type of data this appears to be (MIDI, audio, lyrics, etc.)
2. Suitability for training music AI models
3. Potential quality issues to watch for
4. Recommended preprocessing steps
5. Data augmentation opportunities

Focus on practical ML training considerations.
Respond in JSON format with this structure:
{{
    "analyses": [
        {{
            "dataset_name": "name",
            "data_type": "type",
            "suitability_score": 1-10,
            "analysis": "detailed analysis",
            "quality_issues": ["issue1", "issue2"],
            "preprocessing": ["step1", "step2"],
            "augmentation": ["technique1", "technique2"]
        }}
    ]
}}"""

    print("\nAsking GPT-4 to analyze datasets...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert ML engineer specializing in music/audio AI. Provide practical, actionable analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"Error parsing AI response: {e}")
        return []

    analyses = []
    for item in result.get("analyses", []):
        analysis = DatasetAnalysis(
            dataset_name=item.get("dataset_name", "unknown"),
            total_samples=datasets_info.get(item.get("dataset_name", ""), {}).get("file_count", 0),
            analysis=item.get("analysis", ""),
            quality_issues=item.get("quality_issues", []),
            preprocessing_recommendations=item.get("preprocessing", []),
            augmentation_opportunities=item.get("augmentation", [])
        )
        analyses.append(analysis)

        print(f"\n{'='*40}")
        print(f"Dataset: {analysis.dataset_name}")
        print(f"Samples: {analysis.total_samples:,}")
        print(f"\nAnalysis: {analysis.analysis}")
        print(f"\nQuality Issues:")
        for issue in analysis.quality_issues:
            print(f"  - {issue}")
        print(f"\nPreprocessing Recommendations:")
        for rec in analysis.preprocessing_recommendations:
            print(f"  - {rec}")

    return analyses


def plan_training_with_ai(model_name: Optional[str] = None) -> List[TrainingPlan]:
    """Use OpenAI to create optimized training plans."""
    client = get_openai_client()

    print("\n" + "="*60)
    print("AI Training Planner")
    print("="*60)

    # Get current model states
    models_to_plan = [model_name] if model_name else list(MODEL_REGISTRY.keys())

    model_states = {}
    for name in models_to_plan:
        model_file = MODELS_DIR / f"{name.replace('_', '')}.json"
        if model_file.exists():
            with open(model_file) as f:
                data = json.load(f)
                model_states[name] = {
                    "trained": data.get("trained", False),
                    "layers": len(data.get("layers", [])),
                    "run_id": data.get("run_id", "unknown")
                }
        else:
            model_states[name] = {"trained": False}

    # Check available compute
    import platform
    compute_info = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "has_mps": platform.system() == "Darwin",  # Apple Silicon
        "in_codespace": os.environ.get("CODESPACES") == "true"
    }

    prompt = f"""Create optimized training plans for these music AI models:

Models to train:
{json.dumps({k: MODEL_REGISTRY[k] for k in models_to_plan}, indent=2)}

Current model states:
{json.dumps(model_states, indent=2)}

Available compute:
{json.dumps(compute_info, indent=2)}

For each model, provide an optimized training plan considering:
1. Model architecture and task complexity
2. Dataset requirements (MIDI vs audio)
3. Available compute resources
4. Training order/priority
5. Hyperparameter suggestions with reasoning

Respond in JSON:
{{
    "plans": [
        {{
            "model_name": "name",
            "priority": 1-5,
            "suggested_epochs": int,
            "suggested_batch_size": int,
            "suggested_lr": float,
            "reasoning": "why these choices",
            "estimated_time": "X hours",
            "preprocessing_steps": ["step1"],
            "augmentation_suggestions": ["aug1"],
            "evaluation_metrics": ["metric1"]
        }}
    ],
    "overall_strategy": "high-level training strategy",
    "compute_recommendations": "how to best use available resources"
}}"""

    print("\nAsking GPT-4 to plan training strategy...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert ML engineer. Create practical, optimized training plans."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"Error parsing AI response: {e}")
        return []

    print(f"\n{'='*40}")
    print("OVERALL STRATEGY")
    print("="*40)
    print(result.get("overall_strategy", ""))
    print(f"\nCompute Recommendations: {result.get('compute_recommendations', '')}")

    plans = []
    for item in result.get("plans", []):
        plan = TrainingPlan(
            model_name=item.get("model_name", "unknown"),
            suggested_epochs=item.get("suggested_epochs", 50),
            suggested_batch_size=item.get("suggested_batch_size", 16),
            suggested_lr=item.get("suggested_lr", 0.001),
            reasoning=item.get("reasoning", ""),
            priority=item.get("priority", 3),
            estimated_time=item.get("estimated_time", "unknown"),
            preprocessing_steps=item.get("preprocessing_steps", []),
            augmentation_suggestions=item.get("augmentation_suggestions", []),
            evaluation_metrics=item.get("evaluation_metrics", [])
        )
        plans.append(plan)

        print(f"\n{'='*40}")
        print(f"Model: {plan.model_name} (Priority: {plan.priority})")
        print(f"{'='*40}")
        print(f"Epochs: {plan.suggested_epochs}, Batch: {plan.suggested_batch_size}, LR: {plan.suggested_lr}")
        print(f"Estimated time: {plan.estimated_time}")
        print(f"\nReasoning: {plan.reasoning}")
        print(f"\nPreprocessing: {', '.join(plan.preprocessing_steps)}")
        print(f"Augmentation: {', '.join(plan.augmentation_suggestions)}")
        print(f"Metrics: {', '.join(plan.evaluation_metrics)}")

    # Save plans
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    plans_file = LOGS_DIR / f"training_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(plans_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "plans": [asdict(p) for p in plans],
            "overall_strategy": result.get("overall_strategy", ""),
            "compute_recommendations": result.get("compute_recommendations", "")
        }, f, indent=2)
    print(f"\nPlan saved to: {plans_file}")

    return plans


def train_with_ai_guidance(model_name: str, use_plan: bool = True):
    """Train a model with AI-guided hyperparameters and monitoring."""
    client = get_openai_client()

    print("\n" + "="*60)
    print(f"AI-Guided Training: {model_name}")
    print("="*60)

    # Get AI training plan
    if use_plan:
        plans = plan_training_with_ai(model_name)
        if plans:
            plan = plans[0]
            epochs = plan.suggested_epochs
            batch_size = plan.suggested_batch_size
            lr = plan.suggested_lr
        else:
            epochs, batch_size, lr = 50, 16, 0.001
    else:
        epochs, batch_size, lr = 50, 16, 0.001

    print(f"\nStarting training with AI-suggested parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")

    # Import and run training
    import subprocess

    cmd = [
        sys.executable, "scripts/train.py",
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
    ]

    print(f"\nExecuting: {' '.join(cmd)}")
    print("="*60)

    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)

    if result.returncode == 0:
        print("\n" + "="*60)
        print("Training completed! Requesting AI evaluation...")
        evaluate_with_ai(model_name)
    else:
        print(f"\nTraining failed with code: {result.returncode}")

    return result.returncode == 0


def evaluate_with_ai(model_name: Optional[str] = None):
    """Use OpenAI to evaluate training results and suggest improvements."""
    client = get_openai_client()

    print("\n" + "="*60)
    print("AI Training Evaluation")
    print("="*60)

    # Gather training logs
    training_logs = {}
    log_dir = ROOT / "logs" / "training"

    if log_dir.exists():
        for log_file in sorted(log_dir.glob("*.json"))[-10:]:  # Last 10 runs
            if model_name and model_name.replace("_", "") not in log_file.name:
                continue
            with open(log_file) as f:
                training_logs[log_file.name] = json.load(f)

    if not training_logs:
        print("No training logs found.")
        return

    prompt = f"""Evaluate these music AI training runs and provide improvement suggestions:

Training Results:
{json.dumps(training_logs, indent=2, default=str)}

Analyze:
1. Training convergence and loss curves
2. Signs of overfitting or underfitting
3. Comparison across runs
4. Specific improvements for each model

Respond in JSON:
{{
    "evaluations": [
        {{
            "run_name": "name",
            "model": "model_name",
            "assessment": "good/needs_work/poor",
            "observations": ["obs1", "obs2"],
            "improvements": ["improvement1", "improvement2"],
            "next_steps": ["step1", "step2"]
        }}
    ],
    "overall_assessment": "summary of all training",
    "priority_improvements": ["most important changes"]
}}"""

    print("\nAsking GPT-4 to evaluate training results...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert ML engineer. Provide actionable feedback on training results."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"Error parsing AI response: {e}")
        return

    print(f"\n{'='*40}")
    print("OVERALL ASSESSMENT")
    print("="*40)
    print(result.get("overall_assessment", ""))

    print(f"\n{'='*40}")
    print("PRIORITY IMPROVEMENTS")
    print("="*40)
    for imp in result.get("priority_improvements", []):
        print(f"  - {imp}")

    for eval_item in result.get("evaluations", []):
        print(f"\n{'='*40}")
        print(f"Run: {eval_item.get('run_name', 'unknown')}")
        print(f"Assessment: {eval_item.get('assessment', 'unknown')}")
        print(f"{'='*40}")

        print("Observations:")
        for obs in eval_item.get("observations", []):
            print(f"  - {obs}")

        print("Suggested Improvements:")
        for imp in eval_item.get("improvements", []):
            print(f"  - {imp}")

    # Save evaluation
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    eval_file = LOGS_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(eval_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nEvaluation saved to: {eval_file}")


def interactive_training_session():
    """Run an interactive AI-guided training session."""
    client = get_openai_client()

    print("\n" + "="*60)
    print("Interactive AI Training Session")
    print("="*60)
    print("\nI'll help you train Kelly's music AI models.")
    print("Type 'quit' to exit.\n")

    conversation = [
        {"role": "system", "content": """You are an expert ML engineer helping train music AI models.

Available models:
- emotion_recognizer: CNN for audio emotion classification
- melody_transformer: LSTM for melodic sequence generation
- harmony_predictor: MLP for chord progression prediction
- dynamics_engine: MLP for expression parameter mapping
- groove_predictor: LSTM for timing/groove prediction

Available commands you can suggest:
- analyze: Analyze datasets
- plan [model]: Create training plan
- train [model]: Train a specific model
- evaluate: Evaluate training results
- status: Check model status

Be concise and practical. Focus on actionable ML advice."""}
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Check for direct commands
        if user_input.lower() == 'analyze':
            analyze_datasets_with_ai()
            continue
        elif user_input.lower().startswith('plan'):
            parts = user_input.split()
            model = parts[1] if len(parts) > 1 else None
            plan_training_with_ai(model)
            continue
        elif user_input.lower().startswith('train'):
            parts = user_input.split()
            if len(parts) > 1:
                train_with_ai_guidance(parts[1])
            else:
                print("Specify model: train emotion_recognizer")
            continue
        elif user_input.lower() == 'evaluate':
            evaluate_with_ai()
            continue
        elif user_input.lower() == 'status':
            for name in MODEL_REGISTRY:
                model_file = MODELS_DIR / f"{name.replace('_', '')}.json"
                status = "trained" if model_file.exists() else "not trained"
                print(f"  {name}: {status}")
            continue

        # Otherwise, chat with AI
        conversation.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation,
            temperature=0.7,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": assistant_message})

        print(f"\nAI: {assistant_message}")


def deep_research(topic: str):
    """Use OpenAI to conduct deep research on a music AI topic."""
    client = get_openai_client()

    print("\n" + "="*60)
    print(f"Deep Research: {topic}")
    print("="*60)

    prompt = f"""Conduct deep research on the following topic related to music AI and ML training:

Topic: {topic}

Provide comprehensive analysis including:
1. State-of-the-art approaches and recent papers
2. Best practices and common pitfalls
3. Specific implementation recommendations
4. Code patterns and architectures that work well
5. Hyperparameter guidelines
6. Dataset considerations
7. Evaluation metrics and benchmarks

Be thorough and technical. Include specific numbers, architectures, and actionable recommendations.
Format with clear sections and bullet points."""

    print(f"\nResearching: {topic}...")
    print("(This may take a moment for comprehensive analysis)\n")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a world-class ML researcher specializing in music AI, audio processing, and MIDI analysis. Provide deep, technical, and actionable research."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=4000
    )

    research = response.choices[0].message.content
    print(research)

    # Save research
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    safe_topic = topic.replace(" ", "_").replace("/", "-")[:50]
    research_file = LOGS_DIR / f"research_{safe_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(research_file, 'w') as f:
        f.write(f"# Deep Research: {topic}\n\n")
        f.write(f"*Generated: {datetime.now().isoformat()}*\n\n")
        f.write(research)
    print(f"\n\nResearch saved to: {research_file}")

    return research


def research_and_implement(topic: str, auto_implement: bool = False):
    """Research a topic and optionally implement recommendations."""
    client = get_openai_client()

    print("\n" + "="*60)
    print(f"Research & Implement: {topic}")
    print("="*60)

    # First, do the research
    research = deep_research(topic)

    if not auto_implement:
        return research

    # Ask AI to generate implementation code
    print("\n" + "="*60)
    print("Generating Implementation Code")
    print("="*60)

    impl_prompt = f"""Based on this research:

{research[:3000]}

Generate Python implementation code that applies these findings to our music AI training pipeline.

Our models:
- emotion_recognizer: CNN for audio emotion (7 classes)
- melody_transformer: LSTM for melody generation
- harmony_predictor: MLP for chord prediction (48 chords)
- dynamics_engine: MLP for expression mapping
- groove_predictor: LSTM for timing patterns

Provide:
1. A complete, runnable Python function or class
2. Clear docstrings explaining the approach
3. Comments referencing the research findings
4. Example usage

Focus on practical, immediately usable code."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert Python developer. Write clean, well-documented, production-ready code."},
            {"role": "user", "content": impl_prompt}
        ],
        temperature=0.2,
        max_tokens=3000
    )

    implementation = response.choices[0].message.content
    print(implementation)

    # Save implementation
    impl_file = LOGS_DIR / f"implementation_{topic.replace(' ', '_')[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    with open(impl_file, 'w') as f:
        f.write(f'"""\nGenerated Implementation: {topic}\nBased on deep research conducted {datetime.now().isoformat()}\n"""\n\n')
        # Extract code blocks
        import re
        code_blocks = re.findall(r'```python\n(.*?)```', implementation, re.DOTALL)
        if code_blocks:
            f.write('\n\n'.join(code_blocks))
        else:
            f.write(implementation)
    print(f"\nImplementation saved to: {impl_file}")

    return implementation


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ai_training_orchestrator.py --analyze-datasets
  python scripts/ai_training_orchestrator.py --plan-training
  python scripts/ai_training_orchestrator.py --plan-training --model emotion_recognizer
  python scripts/ai_training_orchestrator.py --train-with-guidance --model melody_transformer
  python scripts/ai_training_orchestrator.py --evaluate-results
  python scripts/ai_training_orchestrator.py --interactive
  python scripts/ai_training_orchestrator.py --research "MIDI melody generation with transformers"
  python scripts/ai_training_orchestrator.py --research "audio emotion recognition" --implement
        """
    )

    parser.add_argument("--analyze-datasets", action="store_true",
                       help="Analyze datasets with AI")
    parser.add_argument("--plan-training", action="store_true",
                       help="Create AI-optimized training plans")
    parser.add_argument("--train-with-guidance", action="store_true",
                       help="Train with AI-suggested hyperparameters")
    parser.add_argument("--evaluate-results", action="store_true",
                       help="Evaluate training results with AI")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive training session")
    parser.add_argument("--model", type=str,
                       help="Specific model to focus on")
    parser.add_argument("--research", type=str,
                       help="Deep research on a topic")
    parser.add_argument("--implement", action="store_true",
                       help="Generate implementation code from research")

    args = parser.parse_args()

    if args.analyze_datasets:
        analyze_datasets_with_ai()
    elif args.plan_training:
        plan_training_with_ai(args.model)
    elif args.train_with_guidance:
        if not args.model:
            print("Specify --model for training")
            sys.exit(1)
        train_with_ai_guidance(args.model)
    elif args.evaluate_results:
        evaluate_with_ai(args.model)
    elif args.interactive:
        interactive_training_session()
    elif args.research:
        if args.implement:
            research_and_implement(args.research, auto_implement=True)
        else:
            deep_research(args.research)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
