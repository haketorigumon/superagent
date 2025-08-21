import modal
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = modal.App("llm-web-ui")

@app.function()
@modal.web_server()
def serve():
    web = FastAPI()
    web.mount("/", StaticFiles(directory="static", html=True), name="static")
    return web
