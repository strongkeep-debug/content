---
title: "Build an AI Transcription Tool in Daytona"
description: "Prototype a video transcription CLI in a Daytona workspace with Sapat, FFmpeg, Whisper APIs, and safe provider configuration."
date: 2026-05-10
author: "Strongkeep Debug"
tags: ["Daytona", "AI", "Whisper", "Transcription", "Python"]
---

# Build an AI Transcription Tool in Daytona

# Introduction

AI-enabled tools are easiest to build when the local setup is boring. The model call can be the interesting part, but the project still needs repeatable dependencies, safe secrets, a clear command-line interface, and a way to prove that the workflow works on a fresh machine.

This guide shows how to prototype an AI transcription tool inside a [Daytona workspace](../definitions/20240819_definition_daytona%20workspace.md) using [Sapat](https://github.com/nkkko/sapat) as the reference project.
Sapat is a compact Python CLI that turns video files into MP3 audio with FFmpeg, sends the audio to a [Whisper](../definitions/20260510_definition_whisper.md)-compatible provider, and writes the transcript beside the original video.

The goal is broader than "run this one repo." You will learn a pattern for building AI tools that can swap providers, keep API keys out of source control, run in a reproducible environment, and leave a useful output artifact.

## TL;DR

- Create a Daytona workspace from the Sapat repository so every dependency is installed in an isolated environment.
- Configure one transcription provider: OpenAI, Groq, or Azure OpenAI.
- Use FFmpeg to normalize video audio before it reaches the model.
- Run Sapat on one video or a directory of `.mp4` files and inspect the `.txt` output.
- Treat Sapat as a small reference architecture for future AI tools: CLI layer, provider layer, normalization layer, and output layer.

## What You Are Building

Sapat has a simple pipeline that is useful for understanding AI product prototypes:

```text
video file or folder
        |
        v
FFmpeg extracts MP3 audio
        |
        v
provider adapter: OpenAI, Groq, or Azure OpenAI
        |
        v
optional transcript correction with a chat model
        |
        v
plain text transcript written next to the video
```

That shape is common in AI developer tools. Deterministic code prepares the input, the provider adapter performs the model task, and the tool writes a durable file that can be reviewed, indexed, edited, or passed into another workflow.

## Whisper Details Worth Designing Around

Whisper-style transcription APIs look simple from the outside: send audio in, get text out.
The product behavior depends heavily on the audio you send and the hints you include with the request.
That is why Sapat is a useful demo for AI engineers.
It does not hide the non-AI work behind a web form; it shows the preparation steps that make the model call reliable.

Start by separating three related jobs:

| Job | What it means | Why it matters |
| --- | --- | --- |
| Transcription | Convert speech into text in the spoken language. | Best for meeting notes, captions, interviews, and searchable recordings. |
| Translation | Convert speech into text in another language, often English. | Useful for cross-language review, but it changes the expected output. |
| Correction | Clean up a transcript after speech recognition. | Helpful for punctuation and names, but it can alter wording if used carelessly. |

For a developer tool, the first product decision is usually whether users need a faithful transcript or a cleaned-up reading copy.
A faithful transcript should preserve the speaker's wording, even when it is rough.
A cleaned-up copy can improve punctuation, capitalization, and repeated domain terms, but should still be reviewed before it becomes customer-facing documentation.

The second decision is audio normalization.
Whisper models can handle noisy real-world recordings, but API requests still have limits.
Sapat converts video to MP3 before calling a provider, then exposes quality levels so you can trade file size against audio detail.
This is the boring engineering that makes the AI feature dependable: shorter files fail less often, lower bitrates cost less to move, and a repeatable FFmpeg command gives you the same input shape every time.

The third decision is provider behavior.
OpenAI, Groq, and Azure OpenAI expose similar transcription concepts, but their authentication, endpoint URLs, deployment names, rate limits, and model names differ.
Keeping that code inside provider adapters makes the CLI easier to trust.
When a new provider is added, you should not need to rewrite batching, output paths, quality settings, or prompt handling.

## Prerequisites

You need the following before starting:

- [Daytona](https://www.daytona.io/docs/getting-started) installed locally.
- [Docker](https://docs.docker.com/get-docker/) running, because Daytona workspaces use containers.
- Git and an editor such as VS Code.
- One short `.mp4` file for testing.
- One provider account and API key for OpenAI, Groq, or Azure OpenAI.
- Basic comfort with Python and terminal commands.

**Do not commit API keys.** Keep them in `.env` locally or in your workspace secret manager. A transcription tool usually handles customer audio, meeting recordings, or product demos, so secret handling and output hygiene matter from the first test run.

## Step 1: Create the Daytona Workspace

Start from the Sapat repository:

```bash
daytona create https://github.com/nkkko/sapat --code
```

When Daytona opens the workspace, inspect the project:

```bash
ls
```

You should see files and folders similar to this:

```text
.devcontainer/
src/
.env.example
pyproject.toml
README.md
requirements.txt
```

The important files are:

| File | Purpose |
| --- | --- |
| `src/sapat/script.py` | Defines the `sapat` CLI options with Click. |
| `src/sapat/transcription/base.py` | Converts input video to MP3 and writes the transcript file. |
| `src/sapat/transcription/openai.py` | Sends audio to the OpenAI transcription endpoint. |
| `src/sapat/transcription/groq.py` | Sends audio to Groq's OpenAI-compatible transcription endpoint. |
| `src/sapat/transcription/azure.py` | Sends audio to an Azure OpenAI audio endpoint. |
| `.env.example` | Lists the environment variables each provider needs. |

This structure is a good starting point for AI tools because it separates user input, provider-specific code, and shared file handling.

## Step 2: Install the Local Package

Inside the Daytona terminal, create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Check that the CLI is available:

```bash
sapat --help
```

If the command is not found, use the module path while debugging:

```bash
python -m sapat.script --help
```

Sapat depends on FFmpeg for the video-to-audio step. Confirm FFmpeg is installed:

```bash
ffmpeg -version
```

If the command is missing in your workspace image, install it in the container:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Step 3: Choose a Provider

Sapat supports three provider adapters. Pick one first. Trying to configure all three at once makes troubleshooting harder.

| Provider | Best for | Required key variables |
| --- | --- | --- |
| OpenAI | Simple default path for `whisper-1` style transcription | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_API_ENDPOINT` |
| Groq | Fast transcription experiments with Groq-hosted Whisper models | `GROQCLOUD_API_KEY`, `GROQCLOUD_MODEL`, `GROQCLOUD_API_ENDPOINT` |
| Azure OpenAI | Teams already running Azure OpenAI deployments | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, deployment and API version variables |

Copy the example file:

```bash
cp .env.example .env
```

Then edit only the provider you are using.

### OpenAI Example

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=whisper-1
OPENAI_API_ENDPOINT=https://api.openai.com/v1/audio/transcriptions
OPENAI_MODEL_NAME_CHAT=gpt-4o
```

### Groq Example

```bash
GROQCLOUD_API_KEY=gsk_...
GROQCLOUD_MODEL=whisper-large-v3-turbo
GROQCLOUD_API_ENDPOINT=https://api.groq.com/openai/v1/audio/transcriptions
GROQCLOUD_MODEL_NAME_CHAT=llama3-8b-8192
```

### Azure OpenAI Example

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME_WHISPER=whisper
AZURE_OPENAI_API_VERSION_WHISPER=2024-06-01
AZURE_OPENAI_DEPLOYMENT_NAME_CHAT=gpt-4o
AZURE_OPENAI_API_VERSION_CHAT=2023-03-15-preview
```

**Production note:** the current Azure adapter builds an audio `translations` URL.
That can be useful if you want English output from non-English speech, but it is not the same product behavior as same-language transcription.
Validate the endpoint path against your Azure deployment before using it for a production workflow.

## Step 4: Understand the CLI Options

Sapat exposes the workflow through one command:

```bash
sapat <video_file_or_directory> --api <provider> [options]
```

The useful options are:

| Option | What it controls |
| --- | --- |
| `--api` or `-a` | Provider adapter: `openai`, `groq`, or `azure`. Required. |
| `--language` or `-l` | Spoken language hint, defaulting to `en`. |
| `--prompt` or `-p` | Context that helps the model spell names and domain terms. |
| `--temperature` or `-t` | Sampling temperature, defaulting to `0.3` in the CLI. |
| `--quality` or `-q` | MP3 conversion quality: `L`, `M`, or `H`. |
| `--correct` | Runs a second chat-model pass to clean spelling, punctuation, and capitalization. |

The `--quality` flag matters because transcription APIs often impose file-size limits. Sapat's OpenAI and Groq adapters validate a 25 MB max audio file. Use `L` or `M` for longer files and `H` for short clips where stereo or higher bitrate matters.

| Quality | FFmpeg settings in Sapat | Use it when |
| --- | --- | --- |
| `L` | 22.05 kHz, mono, 96 kbps | Long talks, meetings, or cost-sensitive tests. |
| `M` | 44.1 kHz, mono, 96 kbps | General transcription with smaller files. |
| `H` | 44.1 kHz, stereo, 192 kbps | Short clips with music, multiple speakers, or higher audio detail. |

## Step 5: Run Your First Transcription

Create a folder for test media:

```bash
mkdir -p samples
```

Copy a short `.mp4` file into `samples/`. Then run one provider. For Groq:

```bash
sapat samples/demo.mp4 --api groq --language en --quality M --prompt "Product demo with Daytona, Sapat, FFmpeg, and Whisper"
```

For OpenAI:

```bash
sapat samples/demo.mp4 --api openai --language en --quality M --prompt "Technical walkthrough with Python CLI terminology"
```

For Azure:

```bash
sapat samples/demo.mp4 --api azure --language en --quality M --prompt "Developer tutorial about AI transcription"
```

When the command finishes, Sapat writes a transcript next to the input:

```text
samples/demo.txt
```

Open the file and check three things:

- Names and product terms are spelled correctly.
- Paragraph flow is readable enough for a first draft.
- The transcript matches the language behavior you expected from the provider.

## Step 6: Use Prompting and Correction Carefully

The `--prompt` option is best used as a vocabulary hint, not as a command to rewrite the audio. Keep it short and concrete:

```bash
sapat samples/demo.mp4 --api openai --language en --quality M --prompt "Names: Daytona, Sapat, Groq, Azure OpenAI, FFmpeg"
```

If the transcript is almost correct but messy, add `--correct`:

```bash
sapat samples/demo.mp4 --api openai --language en --quality M --prompt "Names: Daytona, Sapat, Groq" --correct
```

This runs a second chat completion pass after transcription. Use it for punctuation, capitalization, and recurring spelling errors. Do not use it when you need a legal or compliance-grade transcript where every hesitation and filler word must remain untouched.

## Step 7: Batch a Folder of Videos

Sapat accepts a directory path and processes every `.mp4` file in it:

```bash
sapat samples --api groq --language en --quality L --prompt "Internal engineering demo"
```

This is useful for:

- Creating searchable transcripts for product demos.
- Turning recorded standups into lightweight notes.
- Preparing video content for a documentation workflow.
- Building transcript datasets for later summarization.

Keep batch runs small while you are testing. It is easier to diagnose one bad file, one provider error, or one oversized audio export before you scale the workflow.

## Step 8: Extend the Tool With Another Provider

Sapat's provider layout is intentionally small. To add another transcription API, follow the pattern already used by OpenAI and Groq:

1. Create a new file in `src/sapat/transcription/`.
2. Subclass `TranscriptionBase`.
3. Implement `transcribe_audio`.
4. Read provider configuration from `.env`.
5. Add the provider name to the Click `--api` choices in `src/sapat/script.py`.
6. Instantiate the new provider in the CLI selection branch.

Here is a simplified sketch:

```python
from .base import TranscriptionBase


class ExampleTranscription(TranscriptionBase):
    def __init__(self, temperature: float):
        self.temperature = temperature
        self.max_file_size_mb = 25

    def transcribe_audio(self, audio_file: str, **kwargs):
        # 1. Validate the converted MP3.
        # 2. Send multipart form data to the provider.
        # 3. Return either a string or a dict with a "text" field.
        raise NotImplementedError
```

The important design rule is to keep the CLI stable. Users should still run:

```bash
sapat samples/demo.mp4 --api example --language en --quality M
```

Only the provider adapter should know the provider's authentication headers, endpoint format, and response shape.

## Troubleshooting

| Problem | Likely cause | Fix |
| --- | --- | --- |
| `ffmpeg: command not found` | FFmpeg is missing from the workspace image. | Install FFmpeg with `sudo apt-get install -y ffmpeg`. |
| `Transcription failed: ...` | Provider key, endpoint, deployment, model name, or account quota is wrong. | Recheck `.env`, then test with a short clip. |
| `File size exceeds the maximum limit of 25 MB` | The converted MP3 is too large for the adapter. | Use `--quality L`, shorten the video, or split the audio. |
| Output is English when you expected the source language | The Azure adapter currently uses a translations endpoint. | Confirm whether you need transcription or translation and adjust the endpoint. |
| Technical names are misspelled | The model did not have enough vocabulary context. | Add a short `--prompt` listing names and terms. |
| The transcript is readable but poorly punctuated | Raw transcription often prioritizes words over formatting. | Use `--correct`, then manually review the result. |

## Production Checklist

Before turning this prototype into a team tool, review the following:

- Store API keys as workspace or deployment secrets, not committed files.
- Log provider errors without printing secrets.
- Add tests for oversized files, missing keys, unsupported file types, and provider failures.
- Decide whether transcripts should be retained, encrypted, or deleted after downstream processing.
- Add cost controls for batch transcription.
- Make provider choice explicit in documentation so teammates know which account is billed.
- Keep a sample video and expected transcript for smoke testing.

## Conclusion

Sapat is a useful demo because it keeps the moving parts visible. A small CLI prepares media with FFmpeg, routes audio through a provider adapter, and writes a transcript file that developers can inspect immediately.

That same structure works for many AI-enabled tools.
Put deterministic preparation first, keep provider code isolated, make configuration explicit, and run the whole thing in a reproducible Daytona workspace.
Once that foundation is in place, experimenting with Whisper, correction passes, summaries, or new transcription providers becomes much less fragile.

## References

- [Sapat repository](https://github.com/nkkko/sapat)
- [Daytona documentation](https://www.daytona.io/docs)
- [FFmpeg documentation](https://ffmpeg.org/documentation.html)
- [Groq speech-to-text documentation](https://console.groq.com/docs/speech-to-text)
- [OpenAI Whisper processing guide](https://cookbook.openai.com/examples/whisper_processing_guide)
- [Azure OpenAI audio documentation](https://learn.microsoft.com/azure/ai-services/openai/whisper-quickstart)
