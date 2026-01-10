# recorder-transcriber-server

FastAPI server that records short audio sessions, runs Whisper-based transcription, and enhances the resulting text with a language model.

## API

| Type | Path | Request Body | Response | Notes |
| --- | --- | --- | --- | --- |
| POST | `/start_recording` | _None_ | `{ status:"recording", started_at, max_duration_seconds }` | 409 if a session is already running. |
| POST | `/stop_recording` | _None_ | `{ recording_id, path, captured_at }` | 409 if no session is recording. |
| POST | `/transcribe` | `{ recording_id }` | `{ recording_id, text, generated_at }` | 404 if `recording_id` is unknown. |
| POST | `/enhance` | `{ text, recording_id? }` | `{ title, body, tags[], created_at, recording_id? }` | Generates a summarized/narrative note from raw text. |
| WS | `/listen/ws` | — | — | WebSocket for voice-activated listening (see below). |

All timestamps are ISO 8601 strings. `recording_id` doubles as the file path produced by `/stop_recording` and is required for `/transcribe` unless `/enhance` is invoked directly with free-form text.

### WebSocket `/listen/ws`

Voice-activated listening with wake-word detection and automatic transcription.

**Lifecycle:**
1. Connect to `/listen/ws`
2. Receive `connected` event
3. Send `{ "action": "start" }` to begin listening
4. Receive `state_change` events as state transitions (`IDLE` → `ARMED` → `LISTENING`)
5. Receive `result` event when utterance is captured and transcribed
6. Send `{ "action": "stop" }` to halt, or disconnect to end session

**Incoming commands:**
| Command | Description |
| --- | --- |
| `{ "action": "start" }` | Start listening for wake-words |
| `{ "action": "stop" }` | Stop listening |

**Outgoing events:**
| Event | Fields | Description |
| --- | --- | --- |
| `connected` | `type, message` | Sent on successful connection |
| `state_change` | `type, state, timestamp` | State: `IDLE`, `ARMED`, `LISTENING`, `STOPPED` |
| `result` | `type, recording_id, path, text, captured_at, transcribed_at` | Transcription result |
| `error` | `type, message, timestamp` | Error message |

# Notes  
- Uses pulse as audio backend (at least looks for pulse devices)
- Requires a local openai-compatible llm server for enhancement
- Container is compatible with older cuda host versions to run whisper on Nvidia Pascal GPUs -> CUDA version 19.x -> cudnn 9.x for gpu accelerated transcription
- Can set cpu in config section 
