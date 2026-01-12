# Consolidation Notes (KmiDi_PROJECT)

To work from this consolidated layout:

## Python
- Set PYTHONPATH to include `$(pwd)/source/python`
  - Example: `export PYTHONPATH=$(pwd)/source/python:$PYTHONPATH`
- Install from the consolidated root: `pip install -e .`
  - Uses `pyproject.toml` located at `KmiDi_PROJECT/pyproject.toml`

## C++
- Configure CMake from `KmiDi_PROJECT/source/cpp`:
  - `cd source/cpp`
  - `cmake -S . -B build && cmake --build build`

## Data
- Data schemas are in `KmiDi_PROJECT/data`
- Training data is separate in `KmiDi_TRAINING/`

## Config
- Additional configs are in `KmiDi_PROJECT/config/`
- Training configs are in `KmiDi_PROJECT/config/training/`

## Tests & Scripts
- Tests: `KmiDi_PROJECT/tests`
- Scripts: `KmiDi_PROJECT/scripts`

## Backup
- Full backup will be placed in `KmiDi_BACKUP/`
