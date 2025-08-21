import modal
from pathlib import Path

app = modal.App("llm-web-ui")

# 静态文件路径
html_path = Path(__file__).parent / "index.html"

@app.function()
@modal.web_server()
def serve():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    web = FastAPI()

    @web.get("/")
    def index():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    return web
