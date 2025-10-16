
# CodeHelp – AI Chat Assistant MVP

![CodeHelp Logo](https://via.placeholder.com/300x80?text=CodeHelp)

**CodeHelp** is an AI-powered chat assistant built using **LangGraph, FastAPI, and Chroma memory**. This project demonstrates an MVP (Minimum Viable Product) of a ChatGPT-like interface designed for experimentation with RAG (Retrieval-Augmented Generation) and conversational AI.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations & Known Issues](#limitations--known-issues)
- [Future Improvements](#future-improvements)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview
CodeHelp provides a **simple, elegant, and modern web-based interface** for interacting with an AI assistant. Users can:
- Enter queries in a chat panel
- Receive AI-generated responses
- Store session memory and persistent responses for RAG
- Save API keys securely for backend communication  

This first MVP focuses on functionality and UI simplicity while laying the foundation for more advanced software engineering practices.

---

## Features

- **Single-page chat interface** with HTML/CSS
- **API key management**: Users can input their OpenRouter API key securely
- **Conversational memory**: Stores the last 8 messages in memory
- **Persistent storage**: Saves responses to disk and Chroma vectorstore for RAG
- **Clean modern UI**: Minimalist and professional, inspired by ChatGPT
- **FastAPI backend** for handling messages and API interactions

---

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, LangGraph
- **Memory & RAG**: ChromaDB, LangChain
- **Frontend**: HTML, CSS, Jinja2 templates
- **Deployment-ready**: Docker (planned for future improvements)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<username>/CodeHelp.git
cd CodeHelp
````

2. Create a virtual environment (conda recommended):

```bash
conda create -n codehelp python=3.10
conda activate codehelp
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python main.py
```

5. Open your browser at `http://127.0.0.1:8000` to start chatting.

---

## Usage

1. Enter your **OpenRouter API key** in the popup form (first-time setup).
2. Start chatting in the **single-panel chat interface**.
3. Responses are generated via **LangGraph**, and session memory is saved in **ChromaDB** for retrieval-based generation.

---

## Limitations & Known Issues

* **Single-user session**: Chat history is shared globally; no multi-user support yet.
* **Limited memory**: Only the last 8 messages are stored in the session.
* **No authentication or security**: API key is stored in memory; not encrypted.
* **Basic UI**: Minimalist, no sidebar or advanced features.
* **Deployment**: Not production-ready; no Docker or cloud deployment yet.
* **Cost Management**: API limits may restrict usage; consider monitoring usage.

---

## Future Improvements

**Short-term (MVP+):**

* Multi-user support via database
* Encrypted API key storage
* Extended chat memory
* Advanced UI/UX improvements

**Medium-term:**

* Dockerize application for portable deployment
* Add feature for multiple chat windows / logs
* Integrate user authentication and sessions
* Handle feature-limited API calls gracefully

**Long-term:**

* Full production-ready web app hosted on cloud (AWS, GCP, or Azure)
* Scalable RAG memory for multiple users
* Advanced analytics on chat usage
* Real-time collaboration or multi-agent features

---

## Project Structure

```
CodeHelp/
│
├─ app/
│  ├─ memory/
│  │  ├─ session_memory.py
│  │  └─ chroma_memory.py
│  ├─ retrieval/
│  │  └─ embeddings.py
│  └─ utils/
│     └─ langgraph_setup.py
│
├─ config/
│  ├─ constants.py
│  └─ settings.py
│
├─ static/
│  └─ style.css
│
├─ templates/
│  └─ index.html
│
├─ main.py
└─ README.md
```

---

## License

MIT License © 

