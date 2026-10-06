# HireMate

HireMate is a voice interview practice CLI. It connects to OpenAI's Realtime
API over WebSocket with the `gpt-realtime-2.1` model, captures microphone
audio, and plays the interviewer's spoken responses.

The project uses a `src/` package layout. Configuration, interview prompts,
Realtime API behavior, and terminal interaction live in separate modules so
they can be tested and extended independently.

If you already have a local `.env` file, the app loads it automatically.
`OPENAI_API_KEY` values already set in the shell take precedence.

## What is implemented

- A persistent Realtime conversation, so the interviewer can refer to earlier
  answers during the same interview.
- Interviewer instructions for a supportive, job-focused screening interview.
  The interviewer opens with a greeting and asks one question at a time.
- Live microphone input and spoken audio output. Both your recognized answer
  and the interviewer's speech transcript are shown in the terminal.
- Semantic voice activity detection with low eagerness gives you longer
  thinking pauses (up to 8 seconds) before the interviewer responds. You can
  interrupt the interviewer by speaking.
- Candidate transcripts are shown as a microphone sanity check. Transcription
  is a separate, approximate guide; the Realtime interviewer receives the
  original audio.
- An interrupted response is treated as normal turn-taking; it does not end
  the interview.
- Press `Ctrl+C` to end the interview.
- Configuration through environment variables:
  - `OPENAI_API_KEY` is required.
  - `OPENAI_REALTIME_MODEL` optionally overrides the default model.
  - `OPENAI_REALTIME_VOICE` optionally selects the generated voice (default:
    `marin`).
  - `INTERVIEWER_INSTRUCTIONS` optionally replaces the default interviewer
    instructions.
- Clear errors for a missing API key, API errors, and unsuccessful or
  prematurely closed responses.

## Requirements

- Python 3.10 or later.
- An OpenAI API key with access to the Realtime API and the configured model.
- Internet access.
- Working microphone and speaker devices. Headphones are recommended to
  prevent speaker audio from feeding back into the microphone.
- The `openai[realtime]`, `python-dotenv`, and `sounddevice` dependencies
  (installed with the project).

## Install and run

From this folder, create and activate a virtual environment, then install the
project and its development tools:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The app loads environment variables from a local `.env` file, if present, and
does not override variables already set in the shell. You can also set the
key in PowerShell and start the app:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
hiremate
```

You can also use `python main.py` as a compatibility launcher, or run the
package directly with `python -m hiremate`. Alternatively, add
`OPENAI_API_KEY=your-api-key` to `.env`. In Command Prompt, set the key with
`set OPENAI_API_KEY=your-api-key` first. Keep `.env` local and untracked; do
not paste the key into source files or commit it.

To customize the interviewer or model for the current PowerShell session:

```powershell
$env:INTERVIEWER_INSTRUCTIONS = "Interview me for a junior Python developer role. Ask one question at a time."
$env:OPENAI_REALTIME_MODEL = "gpt-realtime-2.1"
$env:OPENAI_REALTIME_VOICE = "marin"
hiremate
```

If `INTERVIEWER_INSTRUCTIONS` is set, it replaces (rather than appends to) the
default interviewer prompt.

## Project layout

```text
HireMate/
├── .env.example
├── pyproject.toml
├── README.md
├── main.py             # compatibility launcher
├── config.py           # compatibility for earlier root-level imports
├── src/
│   └── hiremate/
│       ├── cli.py       # terminal prompts and application entry point
│       ├── config.py    # environment-backed settings
│       ├── prompts.py   # default interviewer instructions
│       ├── realtime.py  # Realtime session and conversation behavior
│       └── audio.py     # microphone capture and speaker playback
└── tests/
    ├── test_audio.py
    ├── test_config.py
    └── test_realtime.py
```

The root `config.py` remains as a small compatibility shim for any earlier
imports; new application code should use `hiremate.config.Settings`.

## Design decisions

- **Configuration is isolated:** `Settings.from_env()` reads the API key and
  optional overrides once, validates them, and avoids including the secret in
  its representation.
- **Realtime behavior is isolated from the CLI:** `RealtimeInterviewer`
  manages audio turns and streaming, while the CLI handles startup and exit.
- **The official SDK owns the WebSocket connection:** this avoids maintaining
  protocol and connection-management code ourselves.
- **Session setup is acknowledged:** the CLI waits for the server's
  `session.updated` event before requesting the first interviewer response.
- **Voice activity detection handles turns:** the Realtime server detects
  when you finish speaking and can stop generated speech when you interrupt.
  Semantic VAD uses low eagerness so brief pauses are less likely to be treated
  as the end of an answer.
- **Audio is isolated:** device callbacks run outside the async network loop
  and communicate with it through queues. Microphone buffering is bounded to
  avoid unbounded memory growth if the network stalls.
- **Interrupts flush local playback:** speech-start and cancelled-response
  events clear queued audio so buffered interviewer speech does not continue
  after you begin answering.

## Test

Run the test suite and lint checks from the project root:

```powershell
python -m pytest
ruff check .
```
