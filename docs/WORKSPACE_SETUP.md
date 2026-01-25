# Workspace Setup Guide

**Date:** January 18, 2026  
**Purpose:** Comprehensive guide for setting up and using the KmiDi development workspace

## Overview

This guide explains how to set up and use the VS Code workspace for KmiDi development. The workspace provides a unified development environment with configured settings, tasks, and debugging capabilities.

## Prerequisites

Before setting up the workspace, ensure you have:

- **VS Code** or **Cursor** installed
- **Node.js** 18+ and npm
- **Python** 3.9+ with pip
- **Rust** toolchain (for Tauri)
- **CMake** 3.27+ (for C++ builds)
- **Git** for version control

## Initial Setup

### 1. Open Workspace

Open the workspace file in VS Code/Cursor:

```bash
code KmiDi.code-workspace
# or
cursor KmiDi.code-workspace
```

The workspace will open with multiple folder roots:
- **KmiDi** - Main project root
- **Documentation** - Docs folder for focused editing
- **Source Code** - src/ folder
- **Tests** - tests/ folder

### 2. Install Recommended Extensions

VS Code will prompt you to install recommended extensions. Click "Install All" or install individually:

**Required Extensions:**
- Python (ms-python.python)
- Rust Analyzer (rust-lang.rust-analyzer)
- C/C++ (ms-vscode.cpptools)
- CMake Tools (ms-vscode.cmake-tools)
- Tailwind CSS IntelliSense (bradlc.vscode-tailwindcss)

**Recommended Extensions:**
- ESLint (dbaeumer.vscode-eslint)
- Prettier (esbenp.prettier-vscode)
- GitLens (eamodio.gitlens)
- Todo Tree (gruntfuggly.todo-tree)
- Tauri (tauri-apps.tauri-vscode)

### 3. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies (if using virtual environment)
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
pip install -e ".[dev]"
```

### 4. Configure Python Interpreter

1. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "Python: Select Interpreter"
3. Choose your Python interpreter (preferably `.venv/bin/python`)

## Workspace Features

### Folder Organization

The workspace organizes the project into logical folders:

- **KmiDi** - Main project with all source files
- **Documentation** - Quick access to docs/
- **Source Code** - Quick access to src/
- **Tests** - Quick access to tests/

### Settings

Workspace settings are configured in `.vscode/settings.json`:

- **Editor:** Format on save, trim whitespace, 100-char ruler
- **TypeScript:** Strict mode, path aliases, import organization
- **Python:** Ruff linting, Black formatting, pytest testing
- **Rust:** Clippy on save, rustfmt formatting
- **C++:** CMake integration, IntelliSense configuration
- **Tailwind:** IntelliSense for class names

### Tasks

Access tasks via `Cmd+Shift+P` → "Tasks: Run Task" or `Cmd+Shift+B` for build tasks.

**Build Tasks:**
- `Build: C++ Debug` - Build C++ in debug mode
- `Build: C++ Release` - Build C++ in release mode
- `Build: TypeScript` - Type check TypeScript
- `Build: Tauri` - Build Tauri app
- `Build: All` - Build everything

**Development Tasks:**
- `Dev: React` - Start Vite dev server (port 1420)
- `Dev: Python API` - Start Python API server (port 8000)
- `Dev: Tauri` - Start Tauri dev mode
- `Dev: All` - Start all dev servers

**Test Tasks:**
- `Test: C++` - Run C++ tests
- `Test: Rust` - Run Rust tests
- `Test: TypeScript` - Run TypeScript tests
- `Test: Python` - Run Python tests
- `Test: All` - Run all tests

**Maintenance Tasks:**
- `Lint: All` - Lint all code
- `Format: All` - Format all code
- `Clean` - Clean build artifacts
- `Clean: All` - Clean all artifacts

### Launch Configurations

Access launch configurations via the Debug panel (`Cmd+Shift+D`) or Run menu.

**React Development:**
- `Debug React App (Chrome)` - Launch Chrome with React DevTools
- `Attach to Chrome` - Attach to running Chrome instance

**Tauri Development:**
- `Debug Tauri App` - Launch Tauri in dev mode
- `Debug Rust (Tauri)` - Debug Rust backend with LLDB

**C++ Debugging:**
- `Debug C++ (LLDB)` - Debug C++ executable with LLDB
- `Debug C++ Tests` - Debug C++ test executable

**Python Debugging:**
- `Python: Current File` - Debug current Python file
- `Python: Music Brain API` - Debug Music Brain API server
- `Python: Pytest` - Debug pytest tests
- `Python: Attach` - Attach to running Python process

**Compound Configurations:**
- `Debug Full Stack` - Debug React and Python API simultaneously

## Development Workflows

### Starting Development

1. **Quick Start:**
   ```bash
   # Run task: "Dev: All"
   # Or from terminal:
   npm run dev:all
   ```

2. **Individual Services:**
   - React: Run task "Dev: React" or `npm run dev:react`
   - Python API: Run task "Dev: Python API" or `npm run dev:python`
   - Tauri: Run task "Dev: Tauri" or `npm run dev:tauri`

### Building

**Build Everything:**
```bash
# Run task: "Build: All"
# Or from terminal:
npm run build:all
```

**Build Individual Components:**
- C++ Debug: Task "Build: C++ Debug" or `npm run build:cpp`
- C++ Release: Task "Build: C++ Release" or `npm run build:cpp-release`
- TypeScript: Task "Build: TypeScript" or `npm run lint:ts`
- Tauri: Task "Build: Tauri" or `npm run tauri build`

### Testing

**Run All Tests:**
```bash
# Run task: "Test: All"
# Or from terminal:
npm run test:all
```

**Run Individual Test Suites:**
- C++: Task "Test: C++" or `npm run test:cpp`
- Rust: Task "Test: Rust" or `npm run test:rust`
- Python: Task "Test: Python" or `pytest tests/python`
- TypeScript: Task "Test: TypeScript" or `npm test`

### Debugging

**Debug React App:**
1. Start dev server (task "Dev: React")
2. Select "Debug React App (Chrome)" from launch configurations
3. Set breakpoints in TypeScript/React files
4. Debug in Chrome DevTools

**Debug Tauri App:**
1. Select "Debug Tauri App" from launch configurations
2. Set breakpoints in TypeScript or Rust files
3. Debug in integrated debugger

**Debug C++ Code:**
1. Build C++ in debug mode (task "Build: C++ Debug")
2. Select "Debug C++ (LLDB)" from launch configurations
3. Set breakpoints in C++ files
4. Debug with LLDB

**Debug Python:**
1. Select "Python: Current File" or "Python: Music Brain API"
2. Set breakpoints in Python files
3. Debug with debugpy

## Keyboard Shortcuts

### Tasks
- `Cmd+Shift+B` (macOS) / `Ctrl+Shift+B` (Windows/Linux) - Run build task
- `Cmd+Shift+P` → "Tasks: Run Task" - Run any task

### Debugging
- `F5` - Start debugging
- `Shift+F5` - Stop debugging
- `Cmd+Shift+F5` - Restart debugging
- `F9` - Toggle breakpoint
- `F10` - Step over
- `F11` - Step into
- `Shift+F11` - Step out

### Code Navigation
- `Cmd+P` - Quick file open
- `Cmd+Shift+O` - Go to symbol in file
- `Cmd+T` - Go to symbol in workspace
- `F12` - Go to definition
- `Shift+F12` - Find all references

## Troubleshooting

### Extensions Not Working

**Issue:** Extensions not activating

**Solution:**
1. Reload window: `Cmd+Shift+P` → "Developer: Reload Window"
2. Check extension is installed and enabled
3. Check workspace settings don't conflict

### Tasks Not Running

**Issue:** Task fails or doesn't start

**Solution:**
1. Check prerequisites are installed (Node.js, Python, Rust, CMake)
2. Verify npm scripts in `package.json`
3. Check terminal output for errors
4. Ensure you're in the workspace root

### Debugging Not Working

**Issue:** Breakpoints not hitting or debugger not starting

**Solution:**
1. **React:** Ensure dev server is running on port 1420
2. **C++:** Ensure code is built in debug mode with symbols
3. **Python:** Ensure debugpy is installed: `pip install debugpy`
4. **Rust:** Ensure rust-analyzer extension is active
5. Check launch configuration paths are correct

### TypeScript Errors

**Issue:** TypeScript showing errors

**Solution:**
1. Check `tsconfig.json` is valid
2. Run "TypeScript: Restart TS Server" from command palette
3. Verify TypeScript version: `npm list typescript`
4. Check workspace TypeScript SDK is selected

### Python Not Found

**Issue:** Python interpreter not detected

**Solution:**
1. Select interpreter: `Cmd+Shift+P` → "Python: Select Interpreter"
2. Create virtual environment if needed: `python -m venv .venv`
3. Install Python extension if missing
4. Check Python path in settings

### CMake Not Configuring

**Issue:** CMake Tools not working

**Solution:**
1. Install CMake Tools extension
2. Select CMake kit: `Cmd+Shift+P` → "CMake: Select a Kit"
3. Configure CMake: `Cmd+Shift+P` → "CMake: Configure"
4. Check `CMakeLists.txt` is valid

## Workspace Settings Customization

### Override Settings

You can override workspace settings by creating `.vscode/settings.json` in your user settings or by editing the workspace file directly.

### Add Custom Tasks

Add custom tasks to `.vscode/tasks.json`:

```json
{
  "label": "Custom Task",
  "type": "shell",
  "command": "your-command",
  "group": "build"
}
```

### Add Custom Launch Configs

Add custom launch configurations to `.vscode/launch.json`:

```json
{
  "name": "Custom Debug",
  "type": "node",
  "request": "launch",
  "program": "${workspaceFolder}/path/to/file"
}
```

## Integration with Dev Container

If using the dev container (`.devcontainer/devcontainer.json`):

1. Open folder in container: `Cmd+Shift+P` → "Dev Containers: Reopen in Container"
2. Workspace settings will apply in container
3. Extensions from devcontainer.json will be installed
4. All tasks and launch configs work in container

## Best Practices

1. **Use Tasks** - Prefer tasks over manual commands for consistency
2. **Format on Save** - Let editor format code automatically
3. **Use Debugger** - Use launch configs instead of console.log
4. **Check Extensions** - Keep recommended extensions installed
5. **Update Settings** - Keep workspace settings in sync with team

## Related Documentation

- `docs/DEVELOPMENT.md` - Development guide
- `docs/DEVELOPER_GUIDE.md` - Developer guide with patterns
- `.devcontainer/devcontainer.json` - Dev container configuration
- `package.json` - npm scripts reference

## Getting Help

- Check VS Code documentation: https://code.visualstudio.com/docs
- Review workspace settings in `.vscode/settings.json`
- Check task definitions in `.vscode/tasks.json`
- Review launch configurations in `.vscode/launch.json`
- See troubleshooting section above
