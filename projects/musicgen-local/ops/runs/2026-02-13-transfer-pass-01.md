# Transfer Pass 01
Date: 2026-02-13
Mode: local transfer, no-overwrite (`rsync --ignore-existing`)

## Scope
- Priority 1 assets from transfer inventory
- Source roots:
  - /Volumes/KmiDi-external/ml-training-suiteEXTERNAL
  - /Volumes/KmiDi-external/_sortedEXTERNAL/CPP_JUCE/My Mac/Downloads/plugin-update
  - /Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/KmiDi_FINAL/ml/models/registry.schema.json

## Result Counts
- ML scaffold files present: 27
- Legacy import scripts present: 6
- Plugin reference files present: 6
- Model registry schema present: 1

## Destination Map
- /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite
- /Volumes/KmiDi-external/musicgen-local/scripts/legacy-import/ml-training-suite
- /Volumes/KmiDi-external/musicgen-local/apps/plugin-juce/legacy-reference/plugin-update
- /Volumes/KmiDi-external/musicgen-local/schemas/model-registry.legacy.schema.json


## Result Counts (Post-Cleanup)
- ML scaffold files present: 12
- Legacy import scripts present: 6
- Plugin reference files present: 6
- Model registry schema present: 1

## SHA-256 Checksums (Post-Cleanup)
### ML scaffold
f0a9df0ecc98aa1a703051dde6de97ba1b9271806a315754b475b682e8747de8  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/__init__.py
886ed73fd72ae6fbb6a1007e52ee2ce63b9ff2cc09e4bd5d7d2d2b7dcf86daa3  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/config.yaml
99cd8c50f8bf3763e9cf20a67007774d93ed8d4348c4adbb0e4d7728553dc2b0  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/data/__init__.py
f3527d91c7740cbb59ddaa7f702cbe8c7192fa0e8aec1d4ebfa870521313d042  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/data/dataset.py
5211bac3442ba630bf777150f2560ef0c54214611c8a9d8ba74aec3b3e07916c  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/models/__init__.py
e3843d8c603fe3a68676bb0ce5b79676076581dd3cd1e583fe8e69e1b71dc1a7  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/models/audio_classifier.py
751778e8c66003895f532b483a8984b774a94856a514e8636ce8bdd77c57f978  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/models/lora.py
ca9f07d4fbd3dc8dc2358f0a48f022fa786322a7aa2952732acdc8fb658f4f13  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/requirements.txt
d40d8c862f4136e0a8591fc8f86fbec0ea77da7ff23dd5004e0f5e676883c155  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/training/__init__.py
dda183f202a9881b2e457899f0c7c517d52edbacde63b3a360083e6585e6b760  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/training/trainer.py
51c57fed69be9a9e2c9e45c5fbdc8d4da4b52b6e871e9f312ba6c8dbe768644c  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/utils/__init__.py
53a007e2d57f0c195ada5804899cc602fb764f83286f023e6c009b26ccdd609f  /Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/utils/audio.py

### Legacy scripts
20e91fd14834e2262577426f64dce5973cf3c25fe5f4d4403ea7fde4c04494d9  /Volumes/KmiDi-external/musicgen-local/scripts/legacy-import/ml-training-suite/inference.py
952f98ef8ddfee164d4666ea6840ae835430589a8ac1b0e116c85532998c3735  /Volumes/KmiDi-external/musicgen-local/scripts/legacy-import/ml-training-suite/preprocess.py
dce85175b42e6d89d6b9e4542b144fc7fe19cc2a04dbf1e69614cb139c0ea6aa  /Volumes/KmiDi-external/musicgen-local/scripts/legacy-import/ml-training-suite/train.py
144a8b7faa0bc3d7aea5dd5c1f71c6d24f406dfde44ab3af1b97ce95459ce1b8  /Volumes/KmiDi-external/musicgen-local/scripts/legacy-import/ml-training-suite/train_emotion.py
a95924f1793685265053a0edaf65fafcc1ae9b59f88d346489cf5f6b963a3df9  /Volumes/KmiDi-external/musicgen-local/scripts/legacy-import/ml-training-suite/train_voice.py
5966fedeaae1b7f9ba08f559982ad02cc6bc14db532afeb02b5339e7de9f0feb  /Volumes/KmiDi-external/musicgen-local/scripts/legacy-import/ml-training-suite/train_with_lora.py

### Plugin reference
6d9963c5683ccbab1d9fe51dd52042a597b8f69fab189fb9f93b1084e478c70b  /Volumes/KmiDi-external/musicgen-local/apps/plugin-juce/legacy-reference/plugin-update/engine/IntentPipeline.cpp
350cfaa13021e19f147c63e4fe6172766f9dfcba38da32a48dce6170c78644cf  /Volumes/KmiDi-external/musicgen-local/apps/plugin-juce/legacy-reference/plugin-update/engine/IntentPipeline.h
357bc35cf3ee69d15aa681d06bea2748d9ba5ee912834b93c58f984b000fd9a3  /Volumes/KmiDi-external/musicgen-local/apps/plugin-juce/legacy-reference/plugin-update/plugin/PluginEditor.cpp
40e4987a3add12ad23e62b02a14f479d39882ac4b8ea51c003d547d0e27b7030  /Volumes/KmiDi-external/musicgen-local/apps/plugin-juce/legacy-reference/plugin-update/plugin/PluginEditor.h
d254182e0e8466d2b706efbdb5e4fbe9982de7c9b5299e1d80fe0284089902ab  /Volumes/KmiDi-external/musicgen-local/apps/plugin-juce/legacy-reference/plugin-update/plugin/PluginProcessor.cpp
75453d67545b082dbc6666734e3cf457b9936d17b3418bd615651e295f348dec  /Volumes/KmiDi-external/musicgen-local/apps/plugin-juce/legacy-reference/plugin-update/plugin/PluginProcessor.h

### Registry schema
d03f4e6358d6229baa255d3a4b8b750ffb5af6475daf99f8ec23c9e649e7b00d  /Volumes/KmiDi-external/musicgen-local/schemas/model-registry.legacy.schema.json
