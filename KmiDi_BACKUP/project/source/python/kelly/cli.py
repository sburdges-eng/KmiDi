"""Command-line interface for Kelly.

Kelly is a therapeutic iDAW (Intelligent Digital Audio Workstation) that
translates emotional wounds into musical expression through intentional
rule-breaking and authentic emotional mapping.
"""
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionCategory
from kelly.core.intent_processor import IntentProcessor, Wound
from kelly.core.midi_generator import MidiGenerator

app = typer.Typer(
    name="kelly",
    help="Kelly - Therapeutic iDAW translating emotions to music",
    add_completion=False
)
console = Console()


@app.command()
def list_emotions(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category")
) -> None:
    """List all available emotions in the thesaurus."""
    thesaurus = EmotionThesaurus()
    
    # Filter by category if specified
    nodes = list(thesaurus.nodes.values())
    if category:
        try:
            cat_enum = EmotionCategory[category.upper()]
            nodes = thesaurus.get_emotions_by_category(cat_enum)
        except KeyError:
            console.print(f"[red]Error:[/red] Invalid category '{category}'")
            console.print(f"Available categories: {', '.join(c.name.lower() for c in EmotionCategory)}")
            raise typer.Exit(1)
    
    if not nodes:
        console.print("[yellow]No emotions found.[/yellow]")
        return
    
    table = Table(title=f"Kelly Emotion Thesaurus ({len(nodes)} emotions)")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="magenta")
    table.add_column("Category", style="green")
    table.add_column("Intensity", style="yellow", justify="right")
    table.add_column("Valence", style="blue", justify="right")
    table.add_column("Arousal", style="red", justify="right")
    
    for node in sorted(nodes, key=lambda n: n.id):
        table.add_row(
            str(node.id),
            node.name,
            node.category.value,
            f"{node.intensity:.2f}",
            f"{node.valence:+.2f}",
            f"{node.arousal:.2f}"
        )
    
    console.print(table)


@app.command()
def show_emotion(
    name: str = typer.Argument(..., help="Name of the emotion to show")
) -> None:
    """Show detailed information about a specific emotion."""
    thesaurus = EmotionThesaurus()
    emotion = thesaurus.find_emotion_by_name(name)
    
    if not emotion:
        console.print(f"[red]Error:[/red] Emotion '{name}' not found.")
        console.print("Use 'kelly list-emotions' to see available emotions.")
        raise typer.Exit(1)
    
    # Create detailed panel
    content = f"""
[bold]Category:[/bold] {emotion.category.value}
[bold]ID:[/bold] {emotion.id}

[bold]VAD Dimensions:[/bold]
  Valence:   {emotion.valence:+.2f} ({'positive' if emotion.valence > 0 else 'negative'})
  Arousal:   {emotion.arousal:.2f} ({'calm' if emotion.arousal < 0.5 else 'excited'})
  Intensity: {emotion.intensity:.2f} ({'low' if emotion.intensity < 0.5 else 'high'})

[bold]Musical Attributes:[/bold]
"""
    for key, value in emotion.musical_attributes.items():
        content += f"  {key}: {value}\n"
    
    # Show nearby emotions
    nearby = thesaurus.get_nearby_emotions(emotion.id, threshold=0.8, max_results=5)
    if nearby:
        content += "\n[bold]Nearby Emotions:[/bold]\n"
        for node, distance in nearby:
            content += f"  • {node.name} (distance: {distance:.3f})\n"
    
    panel = Panel(content.strip(), title=f"Emotion: {emotion.name}", border_style="cyan")
    console.print(panel)


@app.command()
def process(
    wound: str = typer.Argument(..., help="Description of the wound/trigger"),
    intensity: float = typer.Option(0.7, "--intensity", "-i", help="Intensity of the wound (0.0-1.0)", min=0.0, max=1.0),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output MIDI file path"),
    tempo: int = typer.Option(120, "--tempo", "-t", help="Tempo in BPM", min=20, max=300),
    groove: str = typer.Option("straight", "--groove", "-g", help="Groove template"),
    source: str = typer.Option("user_input", "--source", "-s", help="Source of the wound")
) -> None:
    """
    Process a therapeutic wound and generate music.
    
    This command processes an emotional wound through three phases:
    1. Wound identification
    2. Emotional mapping
    3. Musical rule-breaking for expression
    
    Example:
        kelly process "feeling of loss" --intensity 0.8 --output output.mid
    """
    try:
        console.print(f"\n[bold cyan]Processing wound:[/bold cyan] {wound}")
        console.print(f"[cyan]Intensity:[/cyan] {intensity:.2f}")
        console.print(f"[cyan]Source:[/cyan] {source}\n")
        
        # Process intent
        processor = IntentProcessor()
        wound_obj = Wound(
            description=wound,
            intensity=intensity,
            source=source
        )
        
        result = processor.process_intent(wound_obj)
        
        # Display results
        emotion = result["emotion"]
        console.print(Panel(
            f"[bold]Mapped Emotion:[/bold] {emotion.name}\n"
            f"Category: {emotion.category.value}\n"
            f"Valence: {emotion.valence:+.2f}\n"
            f"Arousal: {emotion.arousal:.2f}\n"
            f"Intensity: {emotion.intensity:.2f}",
            title="Phase 1: Emotion Mapping",
            border_style="green"
        ))
        
        # Show rule breaks
        rule_breaks = result["rule_breaks"]
        if rule_breaks:
            console.print(f"\n[bold yellow]Phase 2-3: Rule Breaks ({len(rule_breaks)})[/bold yellow]")
            for rb in rule_breaks:
                console.print(f"  • [cyan]{rb.rule_type}[/cyan]: {rb.description}")
                console.print(f"    Severity: {rb.severity:.2f}")
                if rb.justification:
                    console.print(f"    [dim]{rb.justification}[/dim]")
        else:
            console.print("\n[yellow]No rule breaks generated for this emotion.[/yellow]")
        
        # Generate MIDI
        generator = MidiGenerator(tempo=tempo)
        params = result["musical_params"]
        
        mode = params.get("mode", "minor")
        allow_dissonance = params.get("allow_dissonance", False)
        
        chord_progression = generator.generate_chord_progression(
            mode=mode,
            allow_dissonance=allow_dissonance,
            length=4
        )
        
        # Save or display
        if output:
            midi_file = generator.create_midi_file(
                chord_progression, 
                groove=groove, 
                output_path=str(output),
                channel=0
            )
            console.print(f"\n[bold green]✓[/bold green] MIDI file saved to: {output}")
            console.print(f"  Tracks: {len(midi_file.tracks)}")
            console.print(f"  Duration: ~{len(chord_progression)} bars")
        else:
            console.print(f"\n[dim]No output file specified. Use --output to save MIDI.[/dim]")
        
        # Show musical parameters
        console.print(f"\n[bold]Musical Parameters:[/bold]")
        console.print(f"  Mode: {mode}")
        console.print(f"  Tempo: {tempo} BPM")
        console.print(f"  Groove: {groove}")
        console.print(f"  Dissonance: {'enabled' if allow_dissonance else 'disabled'}")
        console.print(f"  Tempo modifier: {params.get('tempo_modifier', 1.0):.2f}x")
        console.print(f"  Dynamics: {params.get('dynamics', 0.5):.2f}")
        
        # Show processing metadata
        metadata = result.get("processing_metadata", {})
        if metadata:
            console.print(f"\n[dim]Processing method: {metadata.get('emotion_match_method', 'unknown')}[/dim]")
            if metadata.get("wound_keywords"):
                console.print(f"[dim]Keywords: {', '.join(metadata['wound_keywords'])}[/dim]")
        
        console.print()  # Final newline
        
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def list_grooves() -> None:
    """List available groove templates."""
    generator = MidiGenerator()
    grooves = generator.get_available_grooves()
    
    table = Table(title="Available Groove Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Time Signature", style="green")
    table.add_column("Pattern Length", style="yellow")
    table.add_column("Swing", style="blue")
    table.add_column("Description", style="white")
    
    for groove_name in sorted(grooves):
        template = generator.groove_templates[groove_name]
        table.add_row(
            template.name,
            f"{template.time_signature[0]}/{template.time_signature[1]}",
            str(len(template.pattern)),
            f"{template.swing:.2f}" if template.swing > 0 else "0.00",
            template.description or "No description"
        )
    
    console.print(table)


@app.command()
def interpolate(
    emotion1: str = typer.Argument(..., help="First emotion name"),
    emotion2: str = typer.Argument(..., help="Second emotion name"),
    steps: int = typer.Option(5, "--steps", "-s", help="Number of interpolation steps", min=2, max=20)
) -> None:
    """Interpolate between two emotions."""
    thesaurus = EmotionThesaurus()
    
    em1 = thesaurus.find_emotion_by_name(emotion1)
    em2 = thesaurus.find_emotion_by_name(emotion2)
    
    if not em1:
        console.print(f"[red]Error:[/red] Emotion '{emotion1}' not found.")
        raise typer.Exit(1)
    if not em2:
        console.print(f"[red]Error:[/red] Emotion '{emotion2}' not found.")
        raise typer.Exit(1)
    
    table = Table(title=f"Interpolation: {emotion1} → {emotion2}")
    table.add_column("Step", style="cyan", justify="right")
    table.add_column("t", style="yellow", justify="right")
    table.add_column("Valence", style="blue", justify="right")
    table.add_column("Arousal", style="red", justify="right")
    table.add_column("Intensity", style="green", justify="right")
    table.add_column("Mode", style="magenta")
    
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0.0
        result = thesaurus.interpolate_emotions(em1.id, em2.id, t)
        
        if result:
            table.add_row(
                str(i + 1),
                f"{t:.2f}",
                f"{result['valence']:+.2f}",
                f"{result['arousal']:.2f}",
                f"{result['intensity']:.2f}",
                result['musical_attributes']['mode']
            )
    
    console.print(table)


@app.command()
def version() -> None:
    """Show Kelly version and information."""
    try:
        from kelly import __version__, __author__
        rprint(f"[bold cyan]Kelly[/bold cyan] version [green]{__version__}[/green]")
        rprint(f"Author: {__author__}")
    except ImportError:
        rprint("[bold cyan]Kelly[/bold cyan] (development version)")


if __name__ == "__main__":
    app()
