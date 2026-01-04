# recorder-transcriber-server

FastAPI server that records short audio sessions, runs Whisper-based transcription, and enhances the resulting text with a language model.

## API (all `POST`)

| Path | Request Body | Response | Notes |
| --- | --- | --- | --- |
| `/start_recording` | _None_ | `{ status:"recording", started_at, max_duration_seconds }` | 409 if a session is already running. |
| `/stop_recording` | _None_ | `{ recording_id, path, captured_at }` | 409 if no session is recording. |
| `/transcribe` | `{ recording_id }` | `{ recording_id, text, generated_at }` | 404 if `recording_id` is unknown. |
| `/enhance` | `{ text, recording_id? }` | `{ title, body, tags[], created_at, recording_id? }` | Generates a summarized/narrative note from raw text. |

All timestamps are ISO 8601 strings. `recording_id` doubles as the file path produced by `/stop_recording` and is required for `/transcribe` unless `/enhance` is invoked directly with free-form text.

# Notes  
- Uses pulse as audio backend (at least looks for pulse devices)
- Requires a local openai-compatible llm server for enhancement
- Container is compatible with older cuda versions to run whisper on Nvidia Pascal GPUs (Image is work in progress, currently incompatibilities are raging)
