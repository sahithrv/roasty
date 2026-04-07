# GTA Wasted Pipeline Walkthrough

This document explains the current end-to-end flow for the GTA live pipeline.

The goal is to make the runtime easy to visualize:

- what starts when the program runs
- which components subscribe to which bus topics
- what data shape exists at each stage
- what gets added or removed as the data moves forward
- where files are written locally
- where the Grok request is built and where the Grok response comes back

This is the current "real" path in the project, not an abstract future architecture.

## Entry Point

The current live GTA test starts here:

- [gta_wasted_pipeline.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/gta_wasted_pipeline.py)

When you run:

```powershell
python -m sg_coach.orchestrator.gta_wasted_pipeline
```

the program does this:

1. loads settings from `.env`
2. creates a new session runtime
3. creates the live screen capture source
4. creates the GTA `wasted` detector
5. creates the GTA `wanted stars` HUD detector
6. creates the replay buffer
7. subscribes workers to bus topics
8. starts all workers concurrently with `asyncio.gather(...)`

## High-Level Diagram

```text
Program Start
  ->
Load Settings
  ->
Create SessionRuntime
  ->
Create Workers + Topic Queues
  ->
Live Frame Capture (dxcam)
  ->
FRAME_RAW topic
  -> replay_buffer_worker
  -> detector_worker (wasted)
  -> detector_worker (wanted stars)
  ->
GtaWastedDetector.detect(...) and GtaWantedStarsDetector.detect(...)
  ->
DetectionSignal
  ->
SIGNAL_DETECTOR topic
  ->
fusion_worker
  ->
GameEvent
  ->
EVENT_GAME topic
  -> event_sink
  -> memory_worker
  -> commentary_request_worker
  -> speech_cue_worker
  ->
CommentaryRequest
  ->
COMMENTARY_REQUEST topic
  ->
commentary_model_worker
  ->
Grok API call
  ->
CommentaryResult
  ->
COMMENTARY_READY topic
  ->
commentary_result_sink
  ->
text + raw response written to disk

SPEECH_PLAY topic
  ->
speech_cue_sink
  ->
cheap local text cues written to disk
```

## Core Runtime Object

The session-scoped container is:

- [session.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/session.py)

`SessionRuntime` currently owns:

- `session_id`
- `settings`
- `bus`
- `memory_store`

Why it exists:

- the runtime needs one place to hold session-wide state
- workers should depend on one runtime object instead of a growing list of arguments
- later, more shared components can be added without rewriting every function signature

## The Event Bus

The bus is implemented in:

- [bus.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/bus.py)

This is a **broadcast bus**, not a worker queue.

That means:

- each subscriber gets its own `asyncio.Queue`
- publishing one message to a topic sends a copy of that message to every subscriber queue

Why that matters:

- the same `GameEvent` can go to logging, memory, and commentary at the same time
- one consumer does not steal the event from another

Topic names live in:

- [topics.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/topics.py)

Current important topics:

- `frame.raw`
- `signal.detector`
- `event.game`
- `memory.updated`
- `commentary.request`
- `commentary.ready`
- `speech.play`

Stream completion sentinels live in:

- [streaming.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/shared/streaming.py)

These are used because real detectors do not emit one signal per frame. Most frames emit nothing, so the pipeline must know when a stream is actually finished.

## Data Types By Stage

The main data models live in:

- [events.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/shared/events.py)

### 1. `FramePacket`

Produced by capture.

Contains:

- frame id
- timestamp
- game name
- monitor id
- width / height
- raw image array (`image_bgr`)

Meaning:

- this is still low-level capture data
- no semantic claim has been made yet

### 2. `ReplayFrame`

Produced only inside the replay buffer.

Defined in:

- [replay_buffer.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/capture/replay_buffer.py)

Contains:

- frame id
- timestamp
- resized image
- width / height

Meaning:

- this is a compact historical copy of recent frames
- used later to explain what happened before the event

### 3. `DetectionSignal`

Produced by the detector.

Contains:

- game
- detector name
- signal type, for example `wasted` or `wanted_level_changed`
- confidence score
- ROI name
- metadata about the detection
- `frame_ref`
- dedupe / cooldown keys

Meaning:

- "the detector thinks something happened"
- this is a claim, not yet a canonical event

### 4. `GameEvent`

Produced by the fusion layer.

Contains:

- session id
- game
- event type
- confidence
- tags
- metadata
- optional `frame_path`
- optional `clip_path`
- source signal ids
- dedupe / cooldown keys

Meaning:

- this is the canonical app-level event
- both important events like `wasted` and HUD/state events like `wanted_level_changed` use the same event shape
- memory, cheap speech cues, and Grok commentary all act on this object rather than on raw detector claims

### 5. `SpeechCue`

Produced by the cheap local speech path.

Contains:

- session id
- source event id
- cue type
- one text line
- light metadata copied from the event

Meaning:

- this is a lightweight text seed for a future realtime/voice layer
- it is intentionally cheap and local, so it does not spend multimodal API tokens

### 5. `MemorySnapshot`

Produced by the memory layer.

Defined in:

- [store.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/memory/store.py)

Contains:

- total events
- counters
- latest event
- recent events
- notable events
- recurring patterns
- callback candidates
- summary text

Meaning:

- compact read model of the session so far
- built for downstream consumers like commentary

### 6. `CommentaryRequest`

Produced by the commentary request worker.

Contains:

- persona
- latest event
- recent events
- counters
- callback candidates
- memory summary
- optional main frame path
- selected replay-context frame paths

Meaning:

- this is the exact semantic package that is meant to be sent to Grok

### 7. `CommentaryResult`

Produced by the model worker after calling Grok.

Contains:

- request id
- event id
- model name
- final text
- raw response JSON

Meaning:

- this is the actual model output stage

## Stage-By-Stage Flow

### Stage A: Capture

Files:

- [dxcam_backend.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/capture/dxcam_backend.py)
- [gta_wasted_pipeline.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/gta_wasted_pipeline.py)

`DxcamFrameSource.frames()` captures a real monitor frame and wraps it as `FramePacket`.

`live_frame_producer(...)` publishes that `FramePacket` to `FRAME_RAW`.

What is added here:

- timestamp
- frame id
- raw image array

What is not known yet:

- whether anything meaningful happened

### Stage B: Replay Buffer

File:

- [replay_buffer.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/capture/replay_buffer.py)

`replay_buffer_worker(...)` subscribes to `FRAME_RAW`.

It stores a downscaled subset of recent frames in `ReplayFrameBuffer`.

Why:

- we want the previous ~10 seconds when `wasted` happens
- we do not want to keep full-resolution raw frames for the whole window

What is added here:

- historical local context

What is not added yet:

- no semantic event label

### Stage C: Detection

Files:

- [worker.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/detectors/worker.py)
- [wasted.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/detectors/gta/wasted.py)

`detector_worker(...)` reads `FramePacket`s from `FRAME_RAW` and calls:

- `GtaWastedDetector.detect(frame)`

Inside the detector:

1. crop a center ROI
2. preprocess ROI into edges
3. preprocess the template into edges
4. run template matching
5. track confidence and confirmation count
6. emit `DetectionSignal` only when a real threshold hit occurs

What is added here:

- the semantic claim: `signal_type="wasted"`
- confidence score
- detector metadata

What is removed here:

- nothing is literally removed, but downstream stages stop depending on the full frame contents

### Stage D: Fusion

Files:

- [worker.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/fusion/worker.py)
- [demo.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/fusion/demo.py)

`fusion_worker(...)` reads `DetectionSignal` from `SIGNAL_DETECTOR`.

The current fuser is a passthrough:

- one signal in
- one event out

Later, this is where multiple weak signals could be combined into one stronger event.

What is added here:

- `session_id`
- canonical `event_type`
- `source_signal_ids`

Why this matters:

- the rest of the app should think in terms of `GameEvent`, not detector-specific claims

### Stage E: Memory

Files:

- [worker.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/memory/worker.py)
- [store.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/memory/store.py)

`memory_worker(...)` subscribes to `EVENT_GAME`.

It stores:

- raw event log
- recent events
- counters
- notable events
- recurring patterns
- callback candidates

This produces the session memory that later helps Grok make callbacks like:

- "first death"
- "third wasted in the session"
- "same mistake again"

What is added here:

- session-level meaning across time

### Stage F: Commentary Request Building

File:

- [commentary.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/commentary.py)

`commentary_request_worker(...)` also subscribes to `EVENT_GAME`.

For `wasted` events it does:

1. export replay-buffer frames to disk
2. choose a smaller subset of context frames
3. read the latest session memory snapshot
4. enrich the event metadata with replay context
5. build `CommentaryRequest`
6. publish it on `COMMENTARY_REQUEST`

This is the stage where the event stops being "just a detected death" and becomes "a model-ready commentary prompt."

What is added here:

- `context_frame_paths`
- memory summary
- recent events
- counters
- callback candidates

### Stage G: Grok Payload Construction

File:

- [payloads.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/grok/payloads.py)

This stage converts `CommentaryRequest` into a chat-style payload:

- one `system` message
- one `user` message with a structured JSON blob

The system prompt contains the behavioral rules:

- be sarcastic / funny
- no tactical advice
- no hidden-information reasoning
- keep it short

The user payload contains:

- latest event
- recent event history
- counters
- callback candidates
- summary text
- selected context frame paths

### Stage H: Grok API Call

Files:

- [client.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/grok/client.py)
- [commentary.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/commentary.py)

`commentary_model_worker(...)` subscribes to `COMMENTARY_REQUEST`.

It does:

1. dump the outgoing request JSON to disk
2. call xAI chat completions API
3. extract the assistant text
4. build `CommentaryResult`
5. publish it on `COMMENTARY_READY`

If the API call fails:

- an `_error.json` file is written instead

### Stage I: Final Result

Still in:

- [commentary.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/commentary.py)

`commentary_result_sink(...)` subscribes to `COMMENTARY_READY`.

It writes:

- raw response JSON
- plain text commentary

At this stage, the Grok output exists locally and can later be:

- printed
- shown in UI
- spoken with TTS

## What Gets Written To Disk

### Detector debug

Location:

- `data/debug_frames/detectors/gta_wasted/`

Contains:

- sample ROI crops
- best-match annotated frames
- threshold-hit match images

### Commentary context frames

Location:

- `data/debug_frames/commentary_context/<session_id>/<event_id>/`

Contains:

- exported replay frames from before the `wasted` event

### Commentary request / response

Location:

- `data/debug_commentary/<session_id>/`

Contains:

- `<request_id>.json`
  - exact Grok request payload
- `<request_id>_response.json`
  - raw model response
- `<request_id>_text.txt`
  - final commentary text
- `<request_id>_error.json`
  - model-call failure details, if the API request fails

## Why Duplicate `wasted` Events Can Happen

This is the issue you noticed in testing.

Current reason:

- the banner stays on screen across several frames
- the detector can match multiple nearby frames
- the current detector cooldown is simple and local

Where to improve it later:

- [wasted.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/detectors/gta/wasted.py)
- possibly a later policy/dedup layer between `GameEvent` and commentary

Good future fixes:

- slightly longer detector cooldown
- event dedup window at the `GameEvent` stage
- commentary cooldown so the same death cannot produce two model calls

## What To Visualize Mentally

The easiest mental model is:

```text
Raw pixels
  -> FramePacket
  -> DetectionSignal
  -> GameEvent
  -> MemorySnapshot + replay context
  -> CommentaryRequest
  -> CommentaryResult
```

Each stage adds meaning:

- capture knows what pixels were seen
- detection knows what might have happened
- fusion knows what event the system accepts
- memory knows how this event fits into the session
- commentary knows what to tell the model
- Grok returns the actual line

## Recommended Reading Order In Code

If you want to understand the system from first principles, read these files in this order:

1. [gta_wasted_pipeline.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/gta_wasted_pipeline.py)
2. [events.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/shared/events.py)
3. [bus.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/bus.py)
4. [replay_buffer.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/capture/replay_buffer.py)
5. [wasted.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/detectors/gta/wasted.py)
6. [store.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/memory/store.py)
7. [commentary.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/orchestrator/commentary.py)
8. [payloads.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/grok/payloads.py)
9. [client.py](/c:/Users/sahit/Desktop/roasty_ai/src/sg_coach/grok/client.py)

That order mirrors the actual runtime data flow.
