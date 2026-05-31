"""Entry point — run with: python main.py"""
import uvicorn
import threading
import webbrowser
import time

def open_browser():
    time.sleep(2)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
