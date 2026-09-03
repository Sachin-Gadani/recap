# recap

Turn a two-hour meeting or interview recording into an executive summary you can
actually hand someone — without the audio, the transcript, or a single word of it
leaving your machine.

```
$ recap run interview.m4a
==> Transcribing interview.m4a (1h 47m) with faster-whisper / small.en
    [########################] 100.0%  1:47:12/1:47:12
    1,914 segments, 18,204 words in 6m 11s
==> Summarising with ollama / llama3.1:8b
    notes 1/9  (00:00-12:40, ~2380 tokens)
    ...
    writing executive summary
./northwind-pilot-review.md
./northwind-pilot-review.json
./northwind-pilot-review.transcript.md
```

## Why this exists

The obvious way to summarise a recording is to upload it somewhere. That is fine
until the recording is a hiring debrief, a patient-adjacent research call, an
investor conversation, or a candid discussion your colleagues assumed was in the
room only. `recap` does the same job with a local Whisper model and a local LLM,
so the recording never touches a network.

It is not a HIPAA compliance story — no audit logs, no BAAs, no encryption at
rest beyond what your disk already does. It is the simpler guarantee: **the bytes
stay on this computer.** `recap` refuses to talk to a non-loopback model endpoint
unless you explicitly pass `--allow-remote-llm`.

## What you get

A Markdown report with a fixed structure, because the model is asked for JSON and
`recap` renders the Markdown itself:

- **Executive summary** — 3–6 specific bullets, not "the team discussed various topics"
- **Decisions** — what was decided, why, and the timestamp where it happened
- **Action items** — a table of task / owner / due date / timestamp
- **Discussion** — grouped into themes, in order of importance
- **Open questions, risks, next steps**
- An optional appendix of quotes and figures (`--notes`)

Every claim carries a `MM:SS` timestamp back into the recording, so a reader who
doubts a line can check it in ten seconds. Alongside the report you get the full
transcript as Markdown, JSON, SRT or VTT.

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/Sachin-Gadani/recap.git
cd recap
pip install -e ".[whisper]"
```

Two things need to exist on your machine:

**1. ffmpeg** — to decode mp3/m4a/mp4/etc.

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
winget install Gyan.FFmpeg   # Windows
```

**2. A local LLM.** [Ollama](https://ollama.com) is the easiest:

```bash
ollama pull llama3.1:8b      # ~5 GB, a good default on 16 GB of RAM
ollama serve                 # usually already running
```

Then check everything at once:

```bash
recap doctor
```

```
[ ok ] python: 3.12.4 on Darwin arm64
[ ok ] ffmpeg: /opt/homebrew/bin/ffmpeg
[ ok ] asr (faster-whisper): faster-whisper 1.0.3, model small.en
[ ok ] llm (ollama): http://127.0.0.1:11434 serving llama3.1:8b
[ ok ] privacy: LLM endpoint http://127.0.0.1:11434 is on this machine
[ ok ] work dir: /Users/you/.cache/recap (312.4 GB free)

recap is ready.
```

The Whisper model downloads on first use and is cached; after that the whole
pipeline works with the network off.

## Use

```bash
recap run meeting.m4a                        # the whole pipeline
recap run meeting.m4a -o notes/2026-03-11.md # choose where the report lands
recap run interview.wav --print              # also print to the terminal
recap run board.mp3 --focus "budget and hiring decisions"
recap run standup.m4a --audience "the engineers who missed it"
recap run call.m4a --notes                   # append quotes and figures

recap transcribe long.m4a -o long.srt        # transcription only
recap summarize long.srt -o exec.md          # summarise a transcript you already have

recap doctor                                 # is my toolchain ready?
recap models                                 # what has my LLM server got?
recap config --init                          # write a commented config file
```

Try it without any audio, using the example transcript in this repo:

```bash
recap summarize examples/northwind-pilot-checkin.vtt --print
```

`recap summarize` reads `.json`, `.srt`, `.vtt` and plain `.txt`, so a transcript
from Whisper, MacWhisper, Otter, or a colleague's copy-paste all work as input.

## How it works

```
audio ──► ffmpeg ──► Whisper ──► timestamped transcript
                                        │
                        chunk into ~2400-token overlapping windows
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              │  map: each chunk ──► JSON notes                   │
              │       topics · decisions · actions · questions    │
              │       risks · figures · quotes, all timestamped   │
              └─────────────────────────┬─────────────────────────┘
                                        │
                     merge + de-duplicate (and fold, if huge)
                                        │
                  reduce: all notes ──► one JSON summary
                                        │
                        rendered to Markdown by recap
```

The design choices that matter:

- **Chunks overlap.** A decision stated across a boundary is not lost.
- **Segments are never split.** A chunk boundary always falls between sentences.
- **The model returns JSON at every stage.** `recap` owns the Markdown, so the
  report has the same shape every time and a chatty model cannot reorganise it.
- **`num_ctx` is set explicitly.** Ollama's default context window is small
  enough to silently truncate a long chunk; `recap` never lets it.
- **Long recordings fold.** If the merged notes outgrow the context window, they
  are merged in batches first, recursively, rather than being truncated.
- **A failed chunk is not a failed run.** One bad JSON reply costs you that
  excerpt's notes and nothing else.
- **Everything is cached.** Transcription is keyed on the audio's SHA-256 and
  each map call on its content, so an interrupted run resumes for free and a
  re-run with a different `--focus` skips straight to summarising.

## Configuration

Flags beat environment variables beat the config file beat the defaults.

```bash
recap config --init          # writes ~/.config/recap/config.toml
recap config                 # show what is in effect right now
```

```toml
[asr]
model = "medium.en"          # tiny/base/small/medium/large-v3, .en variants are faster
device = "cuda"              # auto | cpu | cuda
language = "en"
initial_prompt = "Northwind, OIDC, SAML, Rahim"   # seed names and jargon

[llm]
model = "qwen2.5:14b-instruct"
num_ctx = 16384              # bigger context, fewer chunks, better cross-referencing

[summary]
audience = "the clinical operations team"
focus = "protocol deviations and follow-ups"
chunk_tokens = 3000
```

Every option also works as `RECAP_<SECTION>_<KEY>`, e.g. `RECAP_LLM_MODEL=llama3.1:70b`.

### Picking models

Size the LLM to your memory first, then the Whisper model to how messy the audio is.

| Machine | LLM | ASR model |
| --- | --- | --- |
| 8-16 GB | `llama3.2:3b` or `llama3.1:8b` | `small.en` |
| 32 GB | `qwen2.5:14b-instruct`, `num_ctx = 16384` | `medium.en` |
| 64 GB Apple Silicon | `qwen2.5:32b-instruct-q4_K_M`, `num_ctx = 32768` | `large-v3` |
| NVIDIA GPU | the largest model that fits **entirely** in VRAM | `large-v3` |

On a discrete GPU, check `ollama ps` — anything less than 100% GPU means the
model is spilling to system RAM, and the pipeline slows by an order of
magnitude. Drop a size rather than waiting it out.

Two things matter more than parameter count:

- **A bigger `num_ctx` and `chunk_tokens`.** Going from 8k/2400 to 32k/8000
  takes a two-hour meeting from ~10 chunks to ~3. Fewer seams where a decision
  gets split, fewer duplicates to merge, no folding pass, and each excerpt
  carries enough surrounding conversation to tell a firm decision from someone
  thinking out loud. It is also *faster* overall, despite the larger context,
  because there are fewer round trips.
- **A better ASR model.** Summary quality is capped by transcript quality, and
  no LLM recovers a misheard name. `small.en` to `large-v3` is a bigger win than
  any LLM upgrade.

For non-English recordings, use `large-v3`, pick an LLM that speaks the
language, and set `[summary] language`.

### Two models: fast extraction, careful synthesis

`recap` makes one extraction call per chunk and exactly one synthesis call.
Extraction is the easy half — pulling decisions and owners out of an excerpt
into JSON barely troubles a 14B — but you pay its speed once per chunk.
Synthesis is where a larger model earns its keep, and it happens once.

So point them at different models:

```bash
recap run meeting.m4a --llm-model qwen2.5:14b --reduce-model llama3.3:70b
```

```toml
[llm]
model = "qwen2.5:14b-instruct"        # N extraction calls
reduce_model = "llama3.3:70b"         # one synthesis call
```

You get the big model's judgment exactly where it matters, for one call's worth
of waiting. Folding, on the rare runs that need it, deliberately stays on the
fast model — it can run many times. Both models are checked for existence before
transcription starts, so a typo fails in seconds rather than after an hour.

`recap` also speaks the OpenAI chat API, so LM Studio, `llama.cpp --server`, Jan
and vLLM work as well:

```bash
recap run meeting.m4a --llm-backend openai --llm-url http://127.0.0.1:1234 --llm-model local-model
```

Non-loopback URLs are refused unless you pass `--allow-remote-llm`. That is the
one switch that lets transcript text leave the machine, and it is never implied.

## Speaker labels

`recap` does not diarise. Whisper alone cannot tell you who spoke, and bolting on
`pyannote` means a Hugging Face token, a licence click-through, and a large model
download — a lot of moving parts for a guess.

What works well in practice: transcribe first, then say who is who.

```bash
recap transcribe meeting.m4a -o meeting.srt
# open meeting.srt, prefix cues with "Priya: " where it matters
recap summarize meeting.srt -o exec.md
```

`recap` parses `Speaker: ` prefixes from SRT/VTT/TXT input, carries them into the
prompts, and lists the speakers in the report. If you already run a diarising
tool, export its SRT and feed that in.

## What it will not do

- **It will not catch what was never said.** Unspoken context is invisible to it.
- **It is only as good as the transcript.** Whisper mishears names and acronyms;
  `initial_prompt` helps a lot with recurring jargon.
- **A small local model paraphrases loosely.** The timestamps are there so you
  can check anything that matters before you forward it.
- **It is not a compliance control.** Local ≠ compliant. If your recordings are
  regulated, the disk they sit on and the people who can read it are still your
  problem.

Treat the report as a well-organised pointer into the recording, not as evidence.

## Development

```bash
pip install -e ".[dev,whisper]"
pytest                 # 136 tests, no network, no models needed
ruff check .
```

The test suite runs the whole pipeline against a fake Ollama server
(`tests/fake_ollama.py`), so `pytest` needs neither a GPU nor a downloaded model
and finishes in about fifteen seconds.

```
recap/
  config.py      defaults, TOML, env overrides, the loopback check
  audio.py       ffprobe/ffmpeg wrangling
  asr.py         faster-whisper and whisper.cpp backends
  transcript.py  the data model plus json/srt/vtt/txt readers and writers
  chunking.py    overlapping, segment-aligned windows
  llm.py         Ollama and OpenAI-compatible clients, JSON repair, retries
  prompts.py     the map / fold / reduce prompts
  summarize.py   orchestration, normalisation, de-duplication, caching
  render.py      JSON summary -> Markdown
  doctor.py      environment checks with fixes
  cli.py         the terminal interface
```

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
