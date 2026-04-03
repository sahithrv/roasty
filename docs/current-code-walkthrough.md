# Current Code Walkthrough

This document explains the code that exists right now in the repository and how the pieces connect.

It is intentionally about the current implementation, not the full long-term architecture.

## Big Picture

The project currently has four real layers:

1. package bootstrap and settings
2. shared event models
3. an in-process async event bus
4. a demo pipeline that proves the event flow
5. a first in-memory session memory store

Right now, the system is not doing real game capture yet. It is proving the architecture shape first.

## Current File Map

```text
src/sg_coach/
  main.py
  bootstrap.py
  shared/
    logging.py
    settings.py
    types.py
    events.py
  orchestrator/
    bus.py
    topics.py
    demo_pipeline.py
  memory/
    store.py
```

## What Each File Does

### `src/sg_coach/main.py`

The top-level entrypoint.

Its current job is:

- call bootstrap
- get a logger
- confirm the app started with the resolved settings

Later, this will likely start a real runtime session instead of just logging.

### `src/sg_coach/bootstrap.py`

This is the startup preparation layer.

Its current job is:

- load environment-based settings
- configure logging
- create runtime data directories

This file exists so startup logic is centralized instead of scattered.

### `src/sg_coach/shared/settings.py`

This defines the typed application configuration object.

Important idea:

- settings are loaded once
- the result is cached
- the rest of the app consumes a `Settings` object, not raw environment variables

### `src/sg_coach/shared/logging.py`

This provides:

- one global logging format
- a helper to configure logging
- a helper to get loggers

### `src/sg_coach/shared/events.py`

This defines the message types that move through the system.

The main models are:

- `FramePacket`
- `DetectionSignal`
- `GameEvent`
- `CommentaryRequest`

These are the core contracts of the architecture.

### `src/sg_coach/orchestrator/topics.py`

This is the central list of event-bus topic names.

Instead of repeating raw strings everywhere like `"event.game"`, the project now imports named constants such as:

- `FRAME_RAW`
- `SIGNAL_DETECTOR`
- `EVENT_GAME`

This reduces typo risk and makes the event flow easier to trace.

### `src/sg_coach/orchestrator/bus.py`

This is the in-process async event bus.

Its job is:

- let code subscribe to a topic
- give each subscriber its own queue
- broadcast published items to every subscriber queue on that topic

This file is the key decoupling mechanism in the current architecture.

### `src/sg_coach/orchestrator/demo_pipeline.py`

This is not the real runtime.

It is a teaching and validation scaffold that proves:

- frames can be published
- detectors can consume frames and emit signals
- fusion can consume signals and emit events
- multiple sinks can receive the same final event

### `src/sg_coach/memory/store.py`

This is the first real memory module.

Its job is:

- store session events
- count event types
- keep a rolling recent-event window
- track notable events
- track simple recurring patterns
- produce a memory snapshot for future commentary/policy layers

It is still in-memory only. It is not yet persistent.

## The Current Data Flow

The current system flow is:

```text
FramePacket
  -> DetectionSignal
  -> GameEvent
  -> MemorySnapshot / sinks
```

In the demo pipeline, that becomes:

```text
frame_producer
  -> topic: frame.raw
  -> fake_detector_worker
  -> topic: signal.detector
  -> fake_fusion_worker
  -> topic: event.game
  -> event sinks
```

## Core Concepts: Queue, Buffer, Sink

These are easy to mix up, so here is the exact meaning in this codebase.

### Queue

A queue is an inbox for one subscriber.

In `bus.py`, each topic can have multiple subscriber queues.

Example:

- one queue for the fusion worker
- one queue for the memory worker
- one queue for the UI

Each queue receives the same published event if they subscribe to the same topic.

### Buffer

A buffer is stored history, usually bounded.

Current examples:

- `deque(maxlen=recent_window_size)` in `memory/store.py` for recent events
- `deque(maxlen=max(recent_window_size * 2, 16))` for notable events

Those are rolling buffers because old items fall off automatically when the max size is reached.

### Sink

A sink is a consumer at the end of some stage of the pipeline.

In `demo_pipeline.py`, `event_sink()` is a simple sink because it consumes `GameEvent` objects and logs them.

Later, real sinks could be:

- memory update worker
- commentary request builder
- subtitle renderer
- TTS player

## How The Event Bus Works

The most important internal object in `bus.py` is:

```python
self._subscribers: dict[str, list[asyncio.Queue[Any]]]
```

That means:

- dictionary key: topic name
- dictionary value: list of subscriber queues for that topic

Conceptually it can look like this:

```python
{
    "frame.raw": [frame_detector_queue],
    "signal.detector": [fusion_queue],
    "event.game": [memory_queue, ui_queue, commentary_queue],
}
```

This is a broadcast model, not a worker-pool model.

That means:

- publishing one `GameEvent` to `event.game`
- sends that same event to every queue subscribed to `event.game`

It does not mean:

- one consumer gets it and the others miss it

That distinction is the whole reason each subscriber gets its own queue.

## What `events.py` Models Mean

### `FramePacket`

Represents one captured frame plus metadata.

Current important fields:

- `frame_id`
- `timestamp`
- `game`
- `monitor_id`
- `width`
- `height`
- `image_bgr`
- `roi_images`

Right now the demo uses `image_bgr=None` because there is no real capture backend yet.

### `DetectionSignal`

Represents a detector claim.

This is intentionally lower-trust than a `GameEvent`.

Examples:

- "I think this looks like a kill"
- "I think the round ended"

Important idea:

- signals are detector output
- they are not yet canonical gameplay truth

### `GameEvent`

Represents a fused event that downstream systems can act on.

Examples:

- `kill`
- `death`
- `round_end`

This is the type that memory and commentary will mostly care about.

### `CommentaryRequest`

Represents what the commentary layer will eventually consume.

This is not used in the demo yet, but it is already defined so the system has a stable future contract.

## How `demo_pipeline.py` Works

This file is a miniature version of the real architecture.

### `build_demo_frame()`

Creates fake frame input.

This stands in for a real capture backend.

### `frame_producer()`

Publishes a finite number of fake frames to `FRAME_RAW`.

This simulates a capture loop.

### `fake_detector_worker()`

Consumes frame messages from a queue and emits `DetectionSignal` objects.

This simulates the detector layer.

Important architectural point:

- the detector does not call fusion directly
- it only publishes to the bus

That keeps the layers decoupled.

### `fake_fusion_worker()`

Consumes signals and emits `GameEvent` objects.

This simulates the later real fusion stage where multiple weak signals may be combined into one stronger event.

The demo keeps it simple:

- one signal in
- one event out

### `event_sink()`

Consumes final `GameEvent` objects and logs them.

The demo uses two sinks to prove the bus is broadcasting correctly.

### `run_demo()`

This is the wiring function.

It:

- creates the bus
- creates a synthetic `session_id`
- subscribes queues to topics
- starts all workers concurrently with `asyncio.gather`

This is the closest thing the current codebase has to a real runtime wiring layer.

## How `memory/store.py` Works

`SessionMemoryStore` is the first real stateful domain module in the project.

It exists so later systems can ask:

- what has happened recently
- how many times did some event happen
- are there recurring patterns
- is there something worth calling back to

### Internal State

The core internal fields are:

- `_event_log`
  - full event history for the current session
- `_recent_events`
  - rolling buffer of the most recent events
- `_counters`
  - count by event type
- `_notable_events`
  - rolling buffer of events that are considered especially important
- `_recurring_patterns`
  - count by pattern family
- `_callback_candidates`
  - possible future callback opportunities for commentary

### `store_event()`

This is the main write method.

Current behavior:

1. reject an event if its `session_id` does not match the memory store's session
2. copy the event if it needs a `"notable"` tag so shared event objects are not mutated
3. append it to the full event log
4. append it to the recent-event buffer
5. increment the event-type counter
6. optionally mark it as notable
7. prune expired callback candidates
8. update recurring-pattern state
9. decide whether a summary refresh is needed

This method is the future hook point for richer memory intelligence.

### Why `deque` Is Used

`deque(maxlen=...)` is used for rolling windows.

That gives you:

- append performance
- automatic dropping of old entries
- simple bounded memory usage

This is exactly the right tool for "recent events" and "recent notable events".

### `MemorySnapshot`

This is the read model returned to other layers.

That is useful because other modules should not reach into private memory internals like `_event_log`.

Instead, they consume a stable object that contains:

- counters
- latest event
- recent events
- notable events
- recurring patterns
- callback candidates
- summary state

### Summary Logic

Right now `summary_text()` is intentionally primitive.

That is not a bug. It is a staging decision.

The goal is:

- define the method name and call path now
- improve the internal summarization later without changing callers

## Current Important Boundaries

These boundaries already exist and should be preserved:

### Producers do not know consumers

`frame_producer()` does not know who uses frames.

It just publishes frames.

### Detectors do not know fusion

`fake_detector_worker()` does not call the fusion worker directly.

It publishes signals.

### Fusion does not know memory or UI

`fake_fusion_worker()` does not know who consumes events.

It publishes events.

### Memory is a separate domain layer

The memory store should not own the bus or perform commentary generation.

It should store, summarize, and expose memory state.

## Current Limitations

The current codebase does not yet have:

- real screen capture
- real detector logic
- real event fusion rules
- a runtime object that wires memory into the event bus
- disk persistence for memory
- commentary policy
- Grok API integration
- UI rendering

That is normal. The project is still building the internal spine.

## Current Review Notes

The current `SessionMemoryStore` is in good shape for this phase, but there are a few important design realities:

- it is synchronous, which is fine because it is only in-memory work
- it is process-local, so all state disappears on restart
- callback cooldown state here is only memory-side bookkeeping, not final commentary enforcement
- the summary text is a simple local summary, not a model-generated session summary

## Recommended Next Step

The next implementation step should be to connect `SessionMemoryStore` to the demo pipeline.

That means replacing one of the demo sinks with a real memory consumer:

```text
event.game
  -> memory worker
  -> SessionMemoryStore.store_event(...)
  -> MemorySnapshot
```

Once that exists, the project will have its first true end-to-end path:

```text
fake frame
  -> signal
  -> event
  -> memory
  -> snapshot
```

That is the right point to add commentary policy after memory is working.
