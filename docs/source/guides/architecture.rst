Architecture Guide
==================

System Overview
---------------

KmiDi-1 is built on a modular architecture with the following main components:

- **Intent Processing**: Converts emotions and user input into structured musical intents
- **Engines**: Generate musical content (bass, melody, harmony, etc.)
- **Orchestrator**: Coordinates multiple engines to create complete compositions
- **C++ Core**: High-performance audio processing and MIDI generation

Component Relationships
-----------------------

.. mermaid::
   :caption: KmiDi-1 Architecture

   graph TD
       A[User Input] --> B[Intent Processor]
       B --> C[Emotion Thesaurus]
       C --> D[Orchestrator]
       D --> E[Bass Engine]
       D --> F[Melody Engine]
       D --> G[Harmony Engine]
       E --> H[MIDI Output]
       F --> H
       G --> H

Data Flow
---------

1. User provides emotion/intent
2. IntentProcessor validates and structures the input
3. EmotionThesaurus maps emotions to musical parameters
4. Orchestrator coordinates engines
5. Engines generate MIDI content
6. Output is exported as MIDI files
