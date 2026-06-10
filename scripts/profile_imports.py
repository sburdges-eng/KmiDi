#!/usr/bin/env python3
"""Profile Python import performance for optimization."""

import cProfile
import pstats


def profile_imports():
    """Profile the music_brain import."""
    print("Profiling music_brain import...")
    print("=" * 80)

    cProfile.run("import music_brain", "import_stats")

    stats = pstats.Stats("import_stats")
    stats.sort_stats("cumulative")

    print("\nTop 20 functions by cumulative time:")
    print("-" * 80)
    stats.print_stats(20)

    print("\n" + "=" * 80)
    print("\nTop 20 functions by total time:")
    print("-" * 80)
    stats.sort_stats("tottime")
    stats.print_stats(20)

    print("\n" + "=" * 80)
    print("\nImport profiling complete. Stats saved to 'import_stats'")
    print("To view interactively: python3 -m pstats import_stats")


if __name__ == "__main__":
    profile_imports()
