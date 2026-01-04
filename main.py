import uvicorn

from recorder_transcriber.api import app

def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=1643)

if __name__ == "__main__":
    main()
