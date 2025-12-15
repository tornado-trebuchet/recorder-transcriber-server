# recorder-transcriber-server


## Endpoints

/record
/transcribe
/record_transcribe
/enhance
/transcribe_enhance
/record_transcribe_enhance


## Performance 

CPU only for local models

## Flow

### Linux

PulseAudio -> Transcription Model -> Enhancement Model -> Ruturn formatted text at the endpoint 