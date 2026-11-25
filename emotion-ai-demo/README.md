# MozeAI - Intelligent AI Assistant

[![Streamlit App](https://img.shields.io/badge/🚀-Live_Demo-blue)](https://moze12432appio-a65t2vjq4b28fakzt6gzbf.streamlit.app)

## Features

- 🧠 **Visible thinking** - answers stream in live, with a "Thinking…" panel showing the model's reasoning
- 🔍 **Real-time web search** - the model searches when a question needs current information, and lists its sources
- 🐍 **Runs Python** - exact calculations and tested code; formulas rendered with LaTeX
- 💬 **Memory** - recent messages, a running summary of older ones, recall of relevant earlier exchanges, and facts you share (this browser session only)
- 📄 **File analysis** - PDF, DOCX, TXT, CSV, JSON; whole files are indexed and the relevant sections are used for each question
- 📊 **File creation** - PowerPoint decks with speaker notes, Word documents and Excel sheets with live formulas
- 🎨 **Image generation** - create and edit images with AI
- ⚙️ **Settings** - Deep thinking mode and a web-search switch
- 👍 **Learns from feedback** - ratings are saved to a private repository; a weekly GitHub Action turns them into rules and examples, which go live once the pull request is merged

## Tech Stack

- Streamlit for the UI
- Groq (gpt-oss-120b, falling back to gpt-oss-20b) with built-in web search and code execution, adaptive reasoning effort, and automatic fallbacks
- DuckDuckGo search and BeautifulSoup page reading when the built-in tools are unavailable
- Pollinations.ai for image generation; python-pptx, python-docx and openpyxl for files

## Reproducibility

The build evaluated in the IJASC paper is the `ijasc-r1` tag; `mozeai_bench.py` reproduces its benchmark.

## Author

**Mukiibi Moses** - Computer Engineering student and graduate researcher, BCML lab (Bio Computing and Machine Learning lab), Kwangwoon University, Seoul, South Korea

[GitHub](https://github.com/moze12432) | [Portfolio](https://moze12432.github.io)
