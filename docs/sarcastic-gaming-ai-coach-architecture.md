# Sarcastic Gaming AI Coach v1 Architecture

## Summary

- The repository is currently greenfield, so this plan assumes a clean Python project from scratch.
- Use a strict two-layer design:
  - Local capture and detectors watch gameplay and emit structured events.
  - Grok receives only selected events, memory summaries, and occasional snapshots to generate commentary.
- Ship the first real MVP as `valorant` plus `gta_like`, with a reusable `fps_common` detector layer so CS-like support becomes a pack addition later.
- Keep v1 intentionally narrow:
  - no live tactical advice
  - no enemy callouts
  - no hidden-information inference
  - no process memory reading
  - no automation or gameplay assistance
- Two areas are intentionally deferred:
  - full duplex voice conversation
  - broad multi-game parity

## Product Boundaries

### Allowed

- roast the player after visible outcomes
- summarize rounds after they end
- notice recurring mistakes
- make callbacks to earlier moments in the same session
- provide reflective coaching after the moment has passed

### Explicitly Not Allowed

- enemy detection for live callouts
- hidden-info overlays
- aim or recoil assistance
- timing or rotate instructions
- mouse or keyboard automation
- process memory reading
- injection into games

## Runtime Architecture

- Platform: Windows-first desktop application
- Language: Python 3.11+
- App shape: one local Python service with a thin desktop shell
- Concurrency model: `asyncio`
- UI: `PySide6`
- Capture default: `dxcam`
- Capture fallback: `mss`
- CV stack: `opencv-python`, `numpy`
- Models later: `onnxruntime`, `torch`
- Persistence:
  - SQLite as the canonical session store
  - JSONL as a debug mirror
  - JPG or MP4 evidence artifacts only for selected events
- Commentary output:
  - Grok text generation
  - xAI TTS playback for the desired voice personalities
- Overlay strategy:
  - detached subtitle window or desktop shell panel
  - do not rely on an injected in-game overlay for v1

## High-Level System Layers

### 1. Local Event Intelligence

Responsible for:

- screen capture
- optional microphone capture
- ROI extraction
- frame differencing
- template matching
- OCR where justified
- lightweight temporal heuristics
- future local ML detectors

Output:

- structured signals and fused gameplay events

### 2. Grok Commentary Intelligence

Responsible for:

- prompt construction
- commentary generation
- callback usage based on session memory
- optional speech output through xAI TTS

Non-responsibilities:

- continuous frame watching
- primary event detection
- tactical gameplay guidance

## Proposed Folder Structure

```text
sarcastic_gaming_ai_coach/
  pyproject.toml
  README.md
  .env.example
  src/sg_coach/
    main.py
    bootstrap.py
    shared/
      types.py
      logging.py
      clocks.py
      settings.py
    capture/
      base.py
      dxcam_backend.py
      mss_backend.py
      window_locator.py
      roi.py
      replay_buffer.py
      mic.py
    detectors/
      base.py
      signal.py
      registry.py
      scheduler.py
      fusion.py
      cooldowns.py
      packs/
        fps_common/
          killfeed.py
          round_end.py
          scoreboard.py
        valorant/
          death.py
          kill.py
          round_end.py
        gta_like/
          fail_state.py
          wanted_level.py
          chaos.py
      ml/
        base.py
        onnx_sequence.py
        image_classifier.py
    memory/
      models.py
      store.py
      counters.py
      retrieval.py
      summarizer.py
      callbacks.py
    orchestrator/
      bus.py
      session.py
      pipeline.py
      policy.py
      state_machine.py
    grok/
      client.py
      prompts.py
      payloads.py
      tts_xai.py
      safety_filter.py
    ui/
      app.py
      main_window.py
      subtitle_window.py
      hotkeys.py
      viewmodels.py
    config/
      defaults.yaml
      personas.yaml
      games/
        valorant.yaml
        fps_common.yaml
        gta_like.yaml
  data/
    sessions/
    debug_frames/
    debug_clips/
    templates/
  tools/
    replay_runner.py
    labeler.py
    benchmark_detectors.py
    export_dataset.py
  tests/
    unit/
    integration/
    detector_fixtures/
      valorant/
      gta_like/
```

## Module Responsibilities

### `capture/`

- find the target monitor or window
- capture frames locally at a controlled cadence
- expose ROI crops
- maintain a rolling replay buffer
- optionally capture microphone audio later

### `detectors/`

- define detector interfaces
- run heuristic detectors
- handle per-game detector packs
- fuse weak signals into canonical events
- apply confidence scoring, dedupe, and cooldowns
- host future ML-backed detectors under the same interface

### `memory/`

- store raw events
- maintain counters and recurring pattern trackers
- build rolling summaries
- generate callback candidates
- retrieve relevant prior events for Grok prompts

### `orchestrator/`

- own the event bus
- schedule detector execution
- run the pipeline
- decide when to call Grok
- decide when to speak versus stay silent
- manage pause, mute, and session lifecycle

### `grok/`

- build prompt payloads
- call xAI APIs
- enforce anti-cheat and non-tactical prompt rules
- synthesize speech with xAI TTS

### `ui/`

- simple desktop controls
- status and detector health
- subtitles or commentary panel
- mute, pause, and hotkeys

### `config/`

- ROI definitions
- thresholds
- personas
- commentary rate limits
- per-game templates and rules

## Core Event Pipeline

Canonical runtime flow:

```text
FramePacket
  -> DetectionSignal
  -> fused GameEvent
  -> MemoryUpdate
  -> CommentaryDecision
  -> GrokRequest
  -> SpeechPlayback / SubtitleDisplay
```

### Event Bus Topics

- `frame.raw`
- `signal.detector`
- `event.game`
- `memory.updated`
- `commentary.request`
- `commentary.ready`
- `speech.play`
- `ui.state`

### Session State Machine

- `IDLE`
- `ATTACHING`
- `CALIBRATING`
- `RUNNING`
- `PAUSED`
- `ERROR`
- `SHUTDOWN`

### Gameplay Context State

- `UNKNOWN`
- `ROUND_ACTIVE`
- `ROUND_TRANSITION`
- `POST_ROUND`
- `FREE_ROAM`
- `FAIL_SCREEN`

## Core Schemas

```python
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class FramePacket(BaseModel):
    frame_id: str
    timestamp: datetime
    game: str | None = None
    monitor_id: int
    width: int
    height: int
    image_bgr: Any
    roi_images: dict[str, Any] = Field(default_factory=dict)


class DetectionSignal(BaseModel):
    signal_id: str
    timestamp: datetime
    game: str
    detector_name: str
    signal_type: str
    confidence: float
    roi_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    frame_ref: str | None = None
    ttl_ms: int = 750
    dedupe_key: str | None = None
    cooldown_key: str | None = None


class GameEvent(BaseModel):
    event_id: str
    timestamp: datetime
    session_id: str
    game: str
    event_type: str
    confidence: float
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    frame_path: str | None = None
    clip_path: str | None = None
    source_signal_ids: list[str] = Field(default_factory=list)
    dedupe_key: str | None = None
    cooldown_key: str | None = None


class CommentaryRequest(BaseModel):
    request_id: str
    persona: Literal["light_sarcasm", "unhinged", "coach_roast", "commentary_only"]
    latest_event: GameEvent
    recent_events: list[GameEvent]
    counters: dict[str, int]
    callback_candidates: list[dict[str, Any]]
    memory_summary: str
    include_frame: bool = False
    frame_path: str | None = None


class Detector(Protocol):
    name: str
    game: str
    cadence_hz: float

    async def detect(self, frame: FramePacket, context: dict[str, Any]) -> list[DetectionSignal]:
        ...
```

### Pipeline Pseudocode

```python
async def run_session():
    async for frame in capture_backend.frames():
        await bus.publish("frame.raw", frame)

        for detector in registry.for_game(session.game):
            if scheduler.should_run(detector, frame.timestamp):
                signals = await detector.detect(frame, session.context)
                await bus.publish_many("signal.detector", signals)

        fused_events = await fusion.consume_ready_signals()
        for event in fused_events:
            if cooldowns.allow(event) and deduper.allow(event):
                await memory.store_event(event)
                decision = policy.decide(event, memory.snapshot())
                if decision.should_call_grok:
                    await bus.publish("commentary.request", decision.request)
```

## Event Deduplication, Cooldowns, And Fusion

### Deduplication

- suppress duplicate events generated from the same moment
- key off event family, actor role, and a short time bucket

### Cooldowns

- suppress repetitive commentary for repeated moments
- keep dedupe and cooldown separate
- dedupe handles duplicate detection
- cooldown handles pacing

### Fusion Rules

- collect weak signals in short temporal windows
- emit a canonical `GameEvent` only once rule thresholds pass
- require stability across multiple frames when possible

### MVP Fusion Examples

- `death`
  - killfeed self-name as victim
  - or death-state visual transition plus HUD-loss confirmation
- `kill`
  - killfeed self-name as killer
  - stable across 2 to 3 sampled frames
- `round_end`
  - win or loss banner plus phase transition
- `wanted_level_change`
  - star-count delta stable for 3 consecutive frames
- `wasted_busted_mission_failed`
  - fail banner template or OCR stable across multiple frames
- `chaos_moment`
  - motion spike plus brightness or color spike
  - keep this low-confidence at first

## Memory Design

### What To Store

- raw event log
- notable event log
- per-session counters
- recurring mistake counters
- callback notes
- rolling summaries

### Storage Format

SQLite tables:

- `sessions`
- `events`
- `counter_state`
- `memory_notes`
- `summaries`

Debug mirror:

- append-only JSONL event stream

Artifacts:

- still frames for commentary-triggering events
- short clips only for higher-salience events

### Retrieval Strategy

For each commentary request:

- always include the latest event
- include the last 3 to 5 notable events
- include active counters
- include at most one callback candidate from the same pattern family
- prefer explicit counters and notes over embeddings in v1

### Rolling Summaries

- rebuild every 2 minutes or every 8 notable events
- maintain:
  - `short_summary` for prompt budget
  - `long_summary` for logs and later analysis

### Callback Rules

- only callback on the second occurrence onward
- minimum 90 seconds between callbacks
- maximum one callback in every 3 spoken lines
- expire callbacks after 20 minutes or at session boundary

## MVP Scope

### Valorant / FPS MVP

Implement:

- frame capture
- death detection
- kill detection from killfeed changes
- round-end detection
- session logging
- memory counters
- Grok commentary on selected events
- subtitle output and xAI TTS output

Postpone:

- self-flash detection
- whiff classification
- utility misuse classification
- scoreboard analytics
- spoken player input
- CS-like pack generalization until Valorant is stable

### GTA-like MVP

Implement:

- `wasted`
- `busted`
- `mission_failed`
- wanted level changes
- generic `chaos_moment`

Postpone:

- precise crash severity
- precise explosion taxonomy
- mission-type understanding
- long-range open-world scene analysis

### Important MVP Simplification

Do not force a perfect `major_crash_or_explosion` detector in v1. Start with a broader `chaos_moment` event until replay benchmarks justify splitting it into finer classes.

## Detector Strategy

### Phase 1: Heuristics First

Use:

- template matching
- ROI image checks
- simple OCR only where necessary
- frame differencing
- rule-based temporal state machines

### Recommended Initial Detectors

#### `fps_common`

- killfeed region differencer
- round-end banner detector
- scoreboard-open detector

#### `valorant`

- death detector
- kill detector
- round-end detector

#### `gta_like`

- fail-state banner detector
- wanted-level star counter
- chaos detector

### Detectors That Should Likely Stay Heuristic

- static fail-state banners
- wanted star counting
- scoreboard-open detection

### Detectors Worth Upgrading To ML

- crash severity
- explosion or chaos classification
- whiff sequences
- self-sabotage patterns
- utility misuse patterns

### Swap-Friendly Detector Interface

- every detector returns `DetectionSignal`
- fusion creates canonical `GameEvent`
- ML detectors and heuristic detectors share the same contract
- the rest of the system never depends on detector internals

## Grok Commentary Strategy

### Prompt Inputs

- latest event
- recent notable events
- running counters
- callback candidate
- personality mode
- optional still frame

### System Prompt Rules

Grok should be instructed to:

- act as a sarcastic commentator, not a tactical coach
- keep responses short
- avoid repetitive catchphrases
- avoid slurs or abusive content
- never provide enemy callouts
- never give hidden-information advice
- never suggest immediate tactical actions such as:
  - peek now
  - rotate now
  - swing now
  - hold this angle
  - aim here

### Suggested Output Contract

Require structured JSON:

```json
{
  "commentary_text": "short line here",
  "should_speak": true,
  "priority": "normal",
  "callback_event_id": "optional",
  "reason": "why this line was selected"
}
```

### Anti-Repetition Rules

- cap spoken lines by cooldown
- disallow callback-heavy responses too frequently
- track recent phrasing and avoid repeated constructions
- do not speak on every event
- prefer strong or funny moments over routine events

### TTS Strategy

- use xAI TTS for the desired Grok-style voices
- keep TTS separate from text generation so retries and muting stay simple
- reserve full voice-agent interaction for a later phase

## Development Roadmap

### Phase 0: Project Scaffold

- initialize Python project
- add settings and environment loading
- add logging
- add SQLite schema
- add JSONL writer
- add replay-runner skeleton
- add a minimal desktop shell

### Phase 1: Capture And Pipeline

- implement `dxcam` capture
- add `mss` fallback
- add ROI calibration
- add replay ring buffer
- implement event bus and scheduler
- add a simple debug frame viewer

### Phase 2: MVP Detector Packs

- implement Valorant detector pack
- implement GTA-like detector pack
- wire event fusion, dedupe, and cooldowns

### Phase 3: Memory And Commentary

- implement counters
- implement callback engine
- implement rolling summaries
- add Grok prompt builder
- add xAI TTS playback
- enforce commentary pacing policy

### Phase 4: Offline Validation

- build replay-driven detector tests
- measure false positives and false negatives
- tune thresholds
- add artifact browser or debug inspector

### Phase 5: Hybrid And ML Expansion

- add optional ONNX-backed detectors
- compare model-only, heuristic-only, and hybrid pipelines
- expand to CS-like games after the common FPS layer is stable

## Testing And Developer Experience

### Unit Tests

- fusion rules
- dedupe rules
- cooldown behavior
- memory retrieval
- callback rules
- prompt construction
- tactical-output safety filtering

### Detector Fixture Tests

- run prerecorded clips through the same runtime pipeline
- assert emitted event sequences against expected labels

### Integration Tests

- replay clip
- emit events
- update memory
- produce commentary request
- queue subtitle or speech output

### Benchmark Metrics

Track:

- precision
- recall
- false positives per hour
- median latency
- p95 latency

### Debugging Tools

- save confidence traces per detector
- save ROI snapshots for triggered events
- build a replay viewer with event markers and thumbnails
- keep per-game fixtures in `tests/detector_fixtures/<game>`

## Dataset And ML Expansion Plan

### Data Collection

Collect gameplay clips for:

- `chaos_moment`
- `major_crash`
- `explosion`
- `whiff`
- `self_sabotage`
- `utility_mistake`

Use heuristics to auto-slice candidate windows, then correct them manually.

### Labeling Strategy

- label 2 to 5 second clips
- store start and end timestamps
- store one primary class
- include explicit negative examples
- preserve source game, map, resolution, and UI variant metadata

### Model Choices

- static ROI image classification for banners and some HUD states
- short video or frame-sequence classification for temporal events
- avoid object detection unless localization is truly required

### Local Inference Stack

- train in PyTorch
- export to ONNX
- deploy with `onnxruntime`
- prefer small local models for low-latency inference

### Model Packaging

Each model package should contain:

- `model.onnx`
- `labels.json`
- `manifest.yaml`
- `metrics.json`

### Model Versioning And Comparison

- version every model independently
- keep benchmark reports beside model artifacts
- compare heuristic-only, model-only, and hybrid pipelines on the same replay fixture set before promotion

## Practical Defaults

- Python 3.11+
- `uv` for environment and dependency management
- `opencv-python`, `numpy`, `pydantic`, `httpx`, `websockets`, `PySide6`, `dxcam`, `sounddevice`, `onnxruntime`
- SQLite as the source of truth
- JSONL only for debugging and inspection
- initial ROI presets for `1920x1080` and `2560x1440`
- manual ROI calibration UI required
- microphone capture exists in the architecture, but spoken player input is out of MVP

## Likely Failure Points

- GPU or driver sensitivity in capture backends
- per-resolution ROI drift
- OCR instability on stylized game UIs
- noisy chaos detection in GTA-like footage
- over-talking if cooldown policy is too loose
- prompt drift into tactical advice if the safety layer is weak

## Recommended First Implementation Order

1. Scaffold the project and config system.
2. Build capture, ROI calibration, and replay buffering.
3. Build the event bus and memory store.
4. Implement one reliable detector at a time.
5. Add Grok text commentary.
6. Add xAI TTS.
7. Add replay-driven evaluation before expanding detector scope.

## Reference Note

The audio boundary in this plan assumes xAI-provided TTS voices are used through xAI audio APIs rather than local offline TTS. Relevant docs:

- xAI Text-to-Speech: <https://docs.x.ai/developers/model-capabilities/audio/text-to-speech>
- xAI Voice Agent: <https://docs.x.ai/developers/model-capabilities/audio/voice-agent>
