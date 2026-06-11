#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
TARGET_TRIPLE="$(rustc -vV | awk '/host: /{print $2}')"

echo "==> 0. Syncing canonical entities (SSOT)"
python3 "${ROOT_DIR}/scripts/sync_entities.py"

echo "==> 1. Building C++ headless core and Python bindings"
PYBIND_DIR="$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")"
cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="${PYBIND_DIR}" \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_DESKTOP=OFF \
  -DBUILD_PLUGINS=OFF \
  -DKMIDI_BUILD_QT_UI=OFF \
  -DKMIDI_BUILD_JUCE_UI=OFF
cmake --build "${BUILD_DIR}" --parallel 4 --target penta_core penta_core_native

mkdir -p "${ROOT_DIR}/music_brain/native"
cp "${BUILD_DIR}"/python/penta_core_native* "${ROOT_DIR}/music_brain/native/" 2>/dev/null || true

echo "==> 2. Packaging Python engine sidecar"
python3 -m pip install -e "${ROOT_DIR}" --quiet
python3 -m pip install pyinstaller --quiet
pyinstaller --noconfirm --clean --name kmidi_brain --onefile "${ROOT_DIR}/music_brain/api.py"

mkdir -p "${ROOT_DIR}/engine/intent_ir/binaries"
cp "${ROOT_DIR}/dist/kmidi_brain" "${ROOT_DIR}/engine/intent_ir/binaries/kmidi_brain-${TARGET_TRIPLE}"
chmod +x "${ROOT_DIR}/engine/intent_ir/binaries/kmidi_brain-${TARGET_TRIPLE}"

if ! ROOT_DIR_ENV="${ROOT_DIR}" python3 - <<'PY'
import json, os, pathlib, sys
scripts = json.loads(pathlib.Path(os.environ['ROOT_DIR_ENV'], 'package.json').read_text()).get('scripts', {})
sys.exit(0 if 'tauri' in scripts else 1)
PY
then
  echo "==> 3. Historical Tauri UI shell step unavailable in current repo state"
  echo "    package.json does not define a tauri build script."
  echo "    Use npm run build for the frontend and root CMake targets for native/plugin builds."
  exit 1
fi

echo "==> 3. Building guarded historical Tauri UI shell"
cd "${ROOT_DIR}"
npm ci
npm run tauri build

echo "Build v1 pipeline complete."

