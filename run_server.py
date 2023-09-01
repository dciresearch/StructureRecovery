import toml
from app.server_fastapi import app
import uvicorn

FASTAPI_ADDRESS = "0.0.0.0"
FASTAPI_PORT = 80


if __name__ == "__main__":
    uvicorn.run(
        app, host=FASTAPI_ADDRESS, port=FASTAPI_PORT)
