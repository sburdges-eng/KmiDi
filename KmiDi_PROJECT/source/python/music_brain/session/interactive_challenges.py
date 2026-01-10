"""
Interactive Rule-Breaking Challenges

Challenge-based tutorials that teach rule-breaking through hands-on exercises.
Part of the "New Features" implementation for Kelly MIDI Companion.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import random

from .teaching import RuleBreakingTeacher, LESSONS


class ChallengeDifficulty(Enum):
    """Challenge difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ChallengeStatus(Enum):
    """Challenge completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class Challenge:
    """A single rule-breaking challenge."""
    id: str
    title: str
    description: str
    difficulty: ChallengeDifficulty
    category: str  # "harmony", "rhythm", "production", etc.
    objective: str  # What the user should accomplish
    constraints: List[str] = field(default_factory=list)  # Rules/constraints for the challenge
    hints: List[str] = field(default_factory=list)  # Progressive hints
    example_solution: Optional[str] = None  # Example of a valid solution
    points: int = 10  # Points awarded for completion
    time_limit_minutes: Optional[int] = None  # Optional time limit

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "category": self.category,
            "objective": self.objective,
            "constraints": self.constraints,
            "hints": self.hints,
            "example_solution": self.example_solution,
            "points": self.points,
            "time_limit_minutes": self.time_limit_minutes,
        }


@dataclass
class ChallengeProgress:
    """Progress tracking for a challenge."""
    challenge_id: str
    status: ChallengeStatus = ChallengeStatus.NOT_STARTED
    attempts: int = 0
    completed_at: Optional[str] = None
    solution: Optional[str] = None  # User's solution
    feedback: Optional[str] = None  # Feedback on the solution
    hints_used: int = 0
    time_taken_minutes: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "challenge_id": self.challenge_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "completed_at": self.completed_at,
            "solution": self.solution,
            "feedback": self.feedback,
            "hints_used": self.hints_used,
            "time_taken_minutes": self.time_taken_minutes,
        }


# Challenge database
CHALLENGES = [
    Challenge(
        id="challenge_001",
        title="Create Tension with Borrowed Chords",
        description="Take a basic major key progression and introduce borrowed chords to create emotional tension.",
        difficulty=ChallengeDifficulty.BEGINNER,
        category="harmony",
        objective="Modify a I-IV-V-I progression to include at least one borrowed chord (iv, bVI, or bVII) that creates tension.",
        constraints=[
            "Start with a major key progression (e.g., F - Bb - C - F)",
            "Replace at least one chord with a borrowed chord",
            "The result should feel emotionally darker or more complex"
        ],
        hints=[
            "Try replacing the IV chord with iv (minor 4)",
            "The bVI chord creates an 'epic' feeling",
            "bVII gives a mixolydian, rock-like quality"
        ],
        example_solution="F - Bb - Bbm - F (iv substitution) or F - Bb - Db - F (bVI substitution)",
        points=10,
        time_limit_minutes=15
    ),

    Challenge(
        id="challenge_002",
        title="Avoid Tonic Resolution",
        description="Create a progression that deliberately avoids resolving to the tonic, maintaining tension.",
        difficulty=ChallengeDifficulty.INTERMEDIATE,
        category="harmony",
        objective="Write an 8-chord progression that never resolves to the I chord, ending on a different chord.",
        constraints=[
            "Use 8 chords total",
            "Cannot end on the I chord",
            "Progression should still feel musical, not random"
        ],
        hints=[
            "End on a borrowed chord like bVI or bVII",
            "Use modal interchange to find alternative endings",
            "Consider ending on a relative minor (vi)"
        ],
        example_solution="F - C - Dm - Bb - Am - Dm - G - C (ends on V, creating anticipation)",
        points=20,
        time_limit_minutes=20
    ),

    Challenge(
        id="challenge_003",
        title="Laid Back Groove",
        description="Take a perfectly quantized drum pattern and add timing variations to create a 'pocket' feel.",
        difficulty=ChallengeDifficulty.BEGINNER,
        category="rhythm",
        objective="Adjust a drum pattern so that the snare hits 10-20ms late to create a laid-back feel.",
        constraints=[
            "Start with a quantized pattern",
            "Snare should be noticeably behind the beat",
            "Maintain overall groove integrity"
        ],
        hints=[
            "Move the snare back by 10-20ms (not more, or it sounds sloppy)",
            "You can move hi-hats slightly forward to compensate",
            "Listen to hip-hop and neo-soul for reference"
        ],
        example_solution="Quantized snare on beat 2 → Move to 2.02 beats (20ms late at 120 BPM)",
        points=15,
    ),

    Challenge(
        id="challenge_004",
        title="Buried Vocals for Intimacy",
        description="Intentionally mix vocals lower than 'standard' to create an intimate or dissociative feeling.",
        difficulty=ChallengeDifficulty.INTERMEDIATE,
        category="production",
        objective="Create a mix where vocals are intentionally lower in the mix than typical, but still audible.",
        constraints=[
            "Vocals should be audible but not prominent",
            "Use reverb/delay to create space",
            "The result should feel intentional, not like a mistake"
        ],
        hints=[
            "Lower vocals 3-6dB below 'normal' level",
            "Add reverb to create distance",
            "Use high-pass filtering to make room for other elements"
        ],
        example_solution="Vocals at -12dB, reverb send at 30%, high-pass at 200Hz",
        points=25,
    ),

    Challenge(
        id="challenge_005",
        title="Pitch Imperfection as Emotion",
        description="Leave intentional pitch imperfections in a vocal or instrument performance to add emotional honesty.",
        difficulty=ChallengeDifficulty.ADVANCED,
        category="production",
        objective="Identify moments in a performance where pitch correction would be 'correct' but removing it serves the emotion.",
        constraints=[
            "Must be intentional, not just lazy",
            "The imperfection should serve the emotion",
            "Document why you're keeping the imperfection"
        ],
        hints=[
            "Look for moments of vulnerability in the performance",
            "Slight pitch drift can add human feeling",
            "Sometimes the 'crack' in the voice IS the emotion"
        ],
        example_solution="Keep slight flat note on emotional phrase like 'I miss you' - adds weight to the sadness",
        points=30,
    ),

    Challenge(
        id="challenge_006",
        title="Syncopated Melody Over Straight Beat",
        description="Write a melody that heavily syncopates against a straight 4/4 drum pattern.",
        difficulty=ChallengeDifficulty.INTERMEDIATE,
        category="rhythm",
        objective="Create melodic phrases that emphasize off-beats while drums stay on the grid.",
        constraints=[
            "Melody should emphasize beats 1.5, 2.5, 3.5, 4.5",
            "Drums stay perfectly on grid",
            "Create rhythmic tension through syncopation"
        ],
        hints=[
            "Start phrases on the 'and' of beats",
            "Use longer notes that cross bar lines",
            "Listen to funk and jazz for syncopation examples"
        ],
        example_solution="Melody starts on 'and of 1', emphasizes 'and of 2' and 'and of 4', while kick stays on 1 and 3",
        points=20,
    ),

    Challenge(
        id="challenge_007",
        title="Genre Blending with Rule Breaks",
        description="Combine two genres by breaking rules from both, creating something new.",
        difficulty=ChallengeDifficulty.ADVANCED,
        category="harmony",
        objective="Create a progression that blends jazz harmony with rock production, or similar genre combination.",
        constraints=[
            "Must clearly reference two different genres",
            "Rule-breaking should bridge the genres",
            "Result should feel cohesive, not random"
        ],
        hints=[
            "Jazz = extended chords (7ths, 9ths), Rock = power chords",
            "Try jazz voicings with rock distortion",
            "Use modal interchange to bridge harmonic languages"
        ],
        example_solution="Cmaj7 - Fmaj7 - Am - G (jazz) with distorted guitar (rock) = jazz-rock fusion",
        points=35,
    ),
]


class ChallengeSystem:
    """
    Interactive challenge system for learning rule-breaking.

    Usage:
        system = ChallengeSystem()
        system.start_challenge("challenge_001")
        system.submit_solution("challenge_001", "F - Bb - Bbm - F")
        system.get_progress()
    """

    def __init__(self, progress_file: Optional[str] = None):
        self.teacher = RuleBreakingTeacher()
        self.challenges = {ch.id: ch for ch in CHALLENGES}
        self.progress_file = progress_file or "challenge_progress.json"
        self.progress: Dict[str, ChallengeProgress] = {}
        self.load_progress()

    def load_progress(self):
        """Load progress from file."""
        if Path(self.progress_file).exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    for challenge_id, progress_data in data.items():
                        self.progress[challenge_id] = ChallengeProgress(
                            challenge_id=challenge_id,
                            status=ChallengeStatus(progress_data["status"]),
                            attempts=progress_data.get("attempts", 0),
                            completed_at=progress_data.get("completed_at"),
                            solution=progress_data.get("solution"),
                            feedback=progress_data.get("feedback"),
                            hints_used=progress_data.get("hints_used", 0),
                            time_taken_minutes=progress_data.get("time_taken_minutes"),
                        )
            except Exception as e:
                print(f"Warning: Could not load progress: {e}")

    def save_progress(self):
        """Save progress to file."""
        data = {
            challenge_id: progress.to_dict()
            for challenge_id, progress in self.progress.items()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)

    def list_challenges(
        self,
        difficulty: Optional[ChallengeDifficulty] = None,
        category: Optional[str] = None,
        status: Optional[ChallengeStatus] = None
    ) -> List[Challenge]:
        """List challenges with optional filters."""
        challenges = list(self.challenges.values())

        if difficulty:
            challenges = [ch for ch in challenges if ch.difficulty == difficulty]
        if category:
            challenges = [ch for ch in challenges if ch.category == category]
        if status:
            challenge_ids = {
                ch_id for ch_id, prog in self.progress.items()
                if prog.status == status
            }
            challenges = [ch for ch in challenges if ch.id in challenge_ids]

        return challenges

    def get_challenge(self, challenge_id: str) -> Optional[Challenge]:
        """Get a challenge by ID."""
        return self.challenges.get(challenge_id)

    def get_progress(self, challenge_id: str) -> Optional[ChallengeProgress]:
        """Get progress for a challenge."""
        return self.progress.get(challenge_id)

    def start_challenge(self, challenge_id: str) -> Tuple[Challenge, Optional[ChallengeProgress]]:
        """Start a challenge."""
        challenge = self.get_challenge(challenge_id)
        if not challenge:
            raise ValueError(f"Challenge not found: {challenge_id}")

        progress = self.progress.get(challenge_id)
        if not progress:
            progress = ChallengeProgress(challenge_id=challenge_id)
            self.progress[challenge_id] = progress

        progress.status = ChallengeStatus.IN_PROGRESS
        progress.attempts += 1
        self.save_progress()

        return challenge, progress

    def get_hint(self, challenge_id: str, hint_number: int = 0) -> Optional[str]:
        """Get a hint for a challenge (0-indexed)."""
        challenge = self.get_challenge(challenge_id)
        if not challenge:
            return None

        if 0 <= hint_number < len(challenge.hints):
            progress = self.progress.get(challenge_id)
            if progress:
                progress.hints_used = max(progress.hints_used, hint_number + 1)
                self.save_progress()
            return challenge.hints[hint_number]

        return None

    def submit_solution(
        self,
        challenge_id: str,
        solution: str,
        feedback: Optional[str] = None,
        auto_validate: bool = False
    ) -> Tuple[bool, str]:
        """
        Submit a solution for a challenge.

        Args:
            challenge_id: ID of the challenge
            solution: User's solution
            feedback: Optional feedback (if None, will auto-generate)
            auto_validate: Whether to automatically validate (basic check)

        Returns:
            Tuple of (is_valid, feedback_message)
        """
        challenge = self.get_challenge(challenge_id)
        if not challenge:
            return False, f"Challenge not found: {challenge_id}"

        progress = self.progress.get(challenge_id)
        if not progress:
            progress = ChallengeProgress(challenge_id=challenge_id)
            self.progress[challenge_id] = progress

        progress.solution = solution

        # Basic validation (can be enhanced with actual music theory checking)
        if auto_validate:
            is_valid = self._validate_solution(challenge, solution)
        else:
            is_valid = True  # Accept all solutions for now (user self-validates)
            feedback = feedback or "Solution submitted. Review your work against the objective and constraints."

        if is_valid:
            progress.status = ChallengeStatus.COMPLETED
            progress.feedback = feedback or "Challenge completed! Great work breaking the rules intentionally."
            # Note: In a real implementation, you'd set completed_at timestamp
        else:
            progress.feedback = feedback or "Solution doesn't meet all constraints. Try again!"

        self.save_progress()

        return is_valid, progress.feedback

    def _validate_solution(self, challenge: Challenge, solution: str) -> bool:
        """Basic validation of solution (simplified - can be enhanced)."""
        # This is a placeholder - real validation would parse music notation
        # and check against constraints
        solution_lower = solution.lower()

        # Basic checks based on constraints
        for constraint in challenge.constraints:
            constraint_lower = constraint.lower()
            # Very basic keyword matching (not production-ready)
            if "major key" in constraint_lower and "major" not in solution_lower:
                if "minor" not in constraint_lower:  # Allow if constraint mentions minor
                    return False

        return True

    def get_statistics(self) -> Dict:
        """Get overall challenge statistics."""
        total = len(self.challenges)
        completed = sum(1 for p in self.progress.values() if p.status == ChallengeStatus.COMPLETED)
        in_progress = sum(1 for p in self.progress.values() if p.status == ChallengeStatus.IN_PROGRESS)
        total_points = sum(p.points for p in self.progress.values() if p.status == ChallengeStatus.COMPLETED)

        return {
            "total_challenges": total,
            "completed": completed,
            "in_progress": in_progress,
            "not_started": total - completed - in_progress,
            "total_points": total_points,
            "completion_rate": (completed / total * 100) if total > 0 else 0,
        }

    def get_recommended_challenge(self) -> Optional[Challenge]:
        """Get a recommended challenge based on progress."""
        # Find challenges not yet completed
        not_completed = [
            ch for ch in self.challenges.values()
            if self.progress.get(ch.id, ChallengeProgress(challenge_id=ch.id)).status != ChallengeStatus.COMPLETED
        ]

        if not not_completed:
            return None

        # Prefer beginner challenges for new users
        beginner = [ch for ch in not_completed if ch.difficulty == ChallengeDifficulty.BEGINNER]
        if beginner:
            return random.choice(beginner)

        # Otherwise return random uncompleted challenge
        return random.choice(not_completed)

    def interactive_challenge_session(self, challenge_id: Optional[str] = None):
        """Run an interactive challenge session."""
        print("\n" + "=" * 60)
        print("🎯 RULE-BREAKING CHALLENGES")
        print("=" * 60)

        if challenge_id is None:
            challenge_id = self._select_challenge_interactive()
            if challenge_id is None:
                return

        challenge, progress = self.start_challenge(challenge_id)

        print(f"\n📋 Challenge: {challenge.title}")
        print(f"Difficulty: {challenge.difficulty.value.title()}")
        print(f"Category: {challenge.category.title()}")
        print(f"Points: {challenge.points}")
        if challenge.time_limit_minutes:
            print(f"Time Limit: {challenge.time_limit_minutes} minutes")

        print("\n" + "-" * 60)
        print("OBJECTIVE:")
        print("-" * 60)
        print(challenge.objective)

        print("\n" + "-" * 60)
        print("CONSTRAINTS:")
        print("-" * 60)
        for i, constraint in enumerate(challenge.constraints, 1):
            print(f"{i}. {constraint}")

        print("\n" + "-" * 60)
        print("DESCRIPTION:")
        print("-" * 60)
        print(challenge.description)

        # Hint system
        hint_count = 0
        while True:
            try:
                action = input("\n[W]ork on it | [H]int | [S]ubmit solution | [E]xit: ").strip().lower()

                if action == 'h':
                    if hint_count < len(challenge.hints):
                        hint = self.get_hint(challenge_id, hint_count)
                        print(f"\n💡 Hint {hint_count + 1}: {hint}")
                        hint_count += 1
                    else:
                        print("\nNo more hints available.")

                elif action == 's':
                    solution = input("\nEnter your solution: ").strip()
                    if solution:
                        is_valid, feedback = self.submit_solution(challenge_id, solution)
                        print(f"\n{'✅' if is_valid else '❌'} {feedback}")
                        if is_valid:
                            print(f"\n🎉 Challenge completed! You earned {challenge.points} points.")
                        break

                elif action == 'e':
                    print("\nChallenge saved. Come back to complete it later!")
                    break

                elif action == 'w':
                    print("\nWork on the challenge. Return when ready to submit!")
                    continue

            except (EOFError, KeyboardInterrupt):
                print("\n\nChallenge saved. Come back later!")
                break

    def _select_challenge_interactive(self) -> Optional[str]:
        """Interactive challenge selection."""
        stats = self.get_statistics()
        print(f"\nProgress: {stats['completed']}/{stats['total_challenges']} completed ({stats['completion_rate']:.1f}%)")
        print(f"Total Points: {stats['total_points']}")

        print("\nAvailable challenges:")
        print("  [R]ecommended | [L]ist all | [B]y difficulty | [C]ategory | [E]xit")

        try:
            choice = input("\nSelect option: ").strip().lower()

            if choice == 'r':
                rec = self.get_recommended_challenge()
                if rec:
                    return rec.id
                else:
                    print("All challenges completed!")
                    return None

            elif choice == 'l':
                challenges = self.list_challenges()
                for i, ch in enumerate(challenges, 1):
                    status = self.progress.get(ch.id, ChallengeProgress(challenge_id=ch.id)).status.value
                    print(f"  {i}. [{status[0].upper()}] {ch.title} ({ch.difficulty.value})")
                try:
                    num = int(input("\nSelect challenge number: ").strip())
                    if 1 <= num <= len(challenges):
                        return challenges[num - 1].id
                except (ValueError, IndexError):
                    pass

            elif choice == 'b':
                print("  [1] Beginner | [2] Intermediate | [3] Advanced | [4] Expert")
                try:
                    diff_choice = int(input("Select difficulty: ").strip())
                    difficulties = [
                        ChallengeDifficulty.BEGINNER,
                        ChallengeDifficulty.INTERMEDIATE,
                        ChallengeDifficulty.ADVANCED,
                        ChallengeDifficulty.EXPERT,
                    ]
                    if 1 <= diff_choice <= 4:
                        challenges = self.list_challenges(difficulty=difficulties[diff_choice - 1])
                        if challenges:
                            for i, ch in enumerate(challenges, 1):
                                print(f"  {i}. {ch.title}")
                            num = int(input("\nSelect challenge number: ").strip())
                            if 1 <= num <= len(challenges):
                                return challenges[num - 1].id
                except (ValueError, IndexError):
                    pass

            elif choice == 'c':
                categories = set(ch.category for ch in self.challenges.values())
                print("Categories:", ", ".join(categories))
                category = input("Enter category: ").strip().lower()
                challenges = self.list_challenges(category=category)
                if challenges:
                    for i, ch in enumerate(challenges, 1):
                        print(f"  {i}. {ch.title}")
                    num = int(input("\nSelect challenge number: ").strip())
                    if 1 <= num <= len(challenges):
                        return challenges[num - 1].id

            elif choice == 'e':
                return None

        except (EOFError, KeyboardInterrupt):
            return None

        return None


def main():
    """Run challenge system from command line."""
    system = ChallengeSystem()

    if len(sys.argv) > 1:
        challenge_id = sys.argv[1]
        system.interactive_challenge_session(challenge_id)
    else:
        system.interactive_challenge_session()


if __name__ == "__main__":
    import sys
    main()
