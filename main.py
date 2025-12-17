import uvicorn

from recorder_transcriber.api import app
from recorder_transcriber.config import config


def main() -> None:
    host, port = config.server_addr, config.server_port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
