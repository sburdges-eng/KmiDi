# KmiDi Desktop Application GUI Manual

This manual provides a comprehensive guide to the KmiDi desktop application's Graphical User Interface (GUI). KmiDi is an emotion-driven music generation tool, and its GUI is designed to facilitate intuitive interaction with its core functionalities.

**Note on Screenshots**: Due to the limitations of the current documentation environment, actual screenshots are not included in this manual. Placeholder descriptions are provided for expected UI elements.

## Table of Contents
1.  [Application Layout](#1-application-layout)
2.  [Main Features](#2-main-features)
    *   [2.1 Emotion Input & Generation](#21-emotion-input--generation)
    *   [2.2 Music Playback & Export](#22-music-playback--export)
    *   [2.3 Interrogation Mode](#23-interrogation-mode)
    *   [2.4 Side A/Side B Toggle](#24-side-aside-b-toggle)
3.  [Workflows](#3-workflows)
    *   [3.1 Generating a New Song from Emotion](#31-generating-a-new-song-from-emotion)
    *   [3.2 Refining an Emotional Intent](#32-refining-an-emotional-intent)
4.  [Troubleshooting UI Issues](#4-troubleshooting-ui-issues)

---

## 1. Application Layout

The KmiDi desktop application features a clean, modern interface. The main window is divided into several logical sections:

*   **Header**: Contains the application title ("iDAW - Kelly Project") and global controls.
*   **Main Content Area**: The primary interactive space, dynamically changing based on the selected mode (e.g., Emotion Input, Interrogation).
*   **Side Panels (Conceptual)**: While not fully implemented in this version, the design anticipates left/right side panels for more advanced controls or information.

*(Placeholder: Diagram/Screenshot of main application window layout)*

---

## 2. Main Features

### 2.1 Emotion Input & Generation

This is the core interface for transforming emotional intent into music.

*   **Emotional Input Textbox**: A prominent input field where users can type their emotional state or intent (e.g., "I feel anxious but hopeful").
*   **"Load Emotions" Button**: (Expected) Triggers the loading of a list of predefined emotional presets or a thesaurus.
*   **"Generate Music" Button**: Initiates the music generation process based on the entered emotional intent. This communicates with the Music Brain API.

*(Placeholder: Screenshot of Emotion Input & Generation section)*

### 2.2 Music Playback & Export

After generation, users can preview and export their music.

*   **Playback Controls**: (Expected) Buttons for Play, Pause, Stop, and possibly a timeline scrubber.
*   **Export Options**: (Expected) Buttons or menu items to export the generated music in various formats (e.g., MIDI, WAV).

*(Placeholder: Screenshot of Music Playback & Export controls)*

### 2.3 Interrogation Mode

This mode allows for a conversational refinement of the emotional intent.

*   **Interrogation Textbox**: An input field for users to ask follow-up questions or provide more details about their desired music (e.g., "Make it feel more grounded").
*   **"Start Interrogation" Button**: Switches the UI to the interrogation interface.
*   **AI Response Area**: Displays the AI's replies and suggestions for refining the intent.

*(Placeholder: Screenshot of Interrogation Mode interface)*

### 2.4 Side A/Side B Toggle

A conceptual toggle for switching between different operational modes or perspectives of the DAW.

*   **"⏭ Side B" Button**: Toggles to the "Side B" view, often associated with AI generation and dynamic processes.
*   **"Side A: Professional DAW" Section**: A conceptual area for traditional DAW features like Mixer, Timeline, Transport control (currently noted as "coming soon").

*(Placeholder: Screenshot of Side A/Side B toggle and related sections)*

---

## 3. Workflows

### 3.1 Generating a New Song from Emotion

1.  **Launch the KMiDi app**.
2.  **Enter your emotional intent** into the "Emotional Input Textbox" (e.g., "I feel serene and hopeful").
3.  Click the **"Generate Music" button**.
4.  The application will communicate with the Music Brain API to generate music based on your input.
5.  *(Expected)* Once generated, the music will be available for **playback and export**.

*(Placeholder: Flowchart/Sequence Diagram: User Input → API Call → Music Generation → Playback/Export)*

### 3.2 Refining an Emotional Intent

1.  Follow steps 1-3 from "Generating a New Song from Emotion".
2.  Click the **"Start Interrogation" button** to enter the conversational refinement mode.
3.  Type your follow-up message into the "Interrogation Textbox" (e.g., "Make the rhythm more driving").
4.  The AI will respond with suggestions or clarifications.
5.  Repeat steps 3-4 until the desired musical direction is achieved, then click "Generate Music" again.

*(Placeholder: Flowchart/Sequence Diagram: Generate Music → Interrogate → Refine → Re-generate)*

---

## 4. Troubleshooting UI Issues

*   **"Music Brain API is offline" / No music generation**: Ensure the FastAPI Music Generation API is running (typically on `http://127.0.0.1:8000`). Refer to the [Deployment Guide](KMiDi_PROJECT/docs/deployment/DEPLOYMENT_GUIDE.md) for instructions on starting the API.
*   **UI elements not responding**: Restart the KMiDi desktop application. If the issue persists, ensure the Tauri backend is running correctly. Check the terminal where you launched the Tauri app for any error messages.
*   **Application crashes**: Report the issue with detailed steps to reproduce. Check system logs for crash reports.

For more general deployment and API troubleshooting, refer to the [Deployment Guide](KMiDi_PROJECT/docs/deployment/DEPLOYMENT_GUIDE.md) and [API Reference](KMiDi_PROJECT/docs/api/API_REFERENCE.md).
