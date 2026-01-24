Onboarding Guide
================

Welcome to KmiDi-1! This guide will help you get started with the project.

Installation
------------

.. code-block:: bash

   git clone <repository-url>
   cd KmiDi-1
   pip install -r requirements.txt

Quick Start
-----------

.. code-block:: python

   from music_brain.session.intent_schema import CompleteSongIntent, SongRoot, SongIntent, TechnicalConstraints
   from music_brain.kelly_companion.engines import BassEngine, MelodyEngine
   
   # Create an intent
   intent = CompleteSongIntent(
       song_root=SongRoot(core_event="Creating music"),
       song_intent=SongIntent(mood_primary="joyful"),
       technical_constraints=TechnicalConstraints(technical_key="C")
   )
   
   # Generate music
   bass_engine = BassEngine()
   melody_engine = MelodyEngine()

Next Steps
----------

- Read the :doc:`architecture` guide
- Check out the :doc:`contributing` guide
- Review the :ref:`api-reference`
