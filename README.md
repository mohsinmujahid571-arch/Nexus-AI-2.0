# Nexus-AI-2.0
# NEXUS — Personal AI Assistant

### GitHub Repository Description

A voice-enabled, local-first Python AI assistant featuring a futuristic sci-fi interface, Gemini Live integration, persistent user memory, enhanced microphone audio processing, and intelligent desktop automation. Your personal AI butler, designed to assist, learn and adapt.

---

## About NEXUS

**NEXUS** is a personal AI assistant built in Python, designed to bring a more interactive and intelligent desktop assistant experience to your computer.

Powered by Google's Gemini Live API, NEXUS supports real-time voice conversations, natural language interactions, persistent memory and a collection of desktop automation tools — all wrapped in a cinematic, electric-blue neural interface.

The goal is simple: create an intelligent personal assistant that doesn't just respond to commands, but understands preferences, adapts over time and helps manage everyday digital tasks.

## ✨ Features

- **Gemini Live Integration:** Real-time AI conversations with voice input and audio responses.
- **British AI Butler Personality:** Calm, witty, sophisticated and conversational assistant persona.
- **Persistent Memory:** Learns useful preferences and remembers selected information across sessions.
- **Memory Management:** View, add, refresh and forget saved memories directly from Settings.
- **Enhanced Audio Processing:** Browser-based noise suppression, echo cancellation, automatic gain control, audio filtering, compression and adjustable voice gating.
- **Voice Interaction:** Natural speech-based interaction with configurable microphone processing.
- **Desktop Automation:** Open applications, manage files, control media and perform supported computer actions.
- **Futuristic Interface:** Animated neural sphere, holographic elements and interactive system dashboard.
- **System Monitoring:** Live CPU and RAM telemetry.
- **Web Search:** Search the web directly through the integrated interface.
- **Google Workspace Integration:** Optional Gmail, Google Calendar and Google Drive connectivity.
- **WhatsApp Integration:** Supported desktop messaging and calling actions with confirmation safeguards.
- **Privacy Controls:** Local configuration, user-controlled memory and configurable automation permissions.

## 🧠 Persistent Memory

NEXUS includes a dedicated memory system that stores useful, non-sensitive information locally.

Memories are organised into categories:

- Preferences
- Background
- Projects
- Learning
- Workflow

Users can review and manage stored information through the Settings panel. NEXUS can also use relevant saved preferences to personalise future interactions without constantly announcing that it remembers something.

## 🎙️ Audio Processing

NEXUS includes an enhanced microphone pipeline designed to improve voice input quality through:

- Noise suppression where supported by the browser and microphone.
- Echo cancellation.
- Automatic gain control.
- High-pass and low-pass audio filtering.
- Dynamic audio compression.
- Adjustable voice-gate sensitivity.

Audio recognition quality still depends on microphone hardware, room acoustics and the underlying speech recognition service.

## 🛠️ Technology Stack

- Python
- Google Gemini Live API
- Google Gen AI SDK
- Aiohttp
- WebSockets
- HTML, CSS and JavaScript
- Web Audio API
- Web Audio Worklets
- JSON-based local storage

## 🚀 Getting Started

1. Clone the repository.
2. Install the required Python dependencies.
3. Obtain a Google Gemini API key.
4. Launch NEXUS.
5. Enter your API key through the setup screen.
6. Open the local interface and initialise the voice connection.

## 🔒 Privacy & Security

NEXUS stores its configuration and persistent memories in a local `.nexus` directory.

The application includes configurable automation permissions and restrictions on the types of information saved in persistent memory.

**Important:** Saved memories remain on the local computer, but relevant memory context is sent to Google's Gemini service when establishing an AI session. Google Workspace integrations also require user-authorised access.

Users should keep API keys, OAuth credentials and other private configuration files out of public repositories.

## 📌 Project Status

NEXUS is an evolving personal AI assistant project. Features, integrations and audio processing capabilities may change as development continues.

---

**Built with Python, curiosity and a little futuristic ambition.**

*Your computer. Your assistant. Your NEXUS.*
