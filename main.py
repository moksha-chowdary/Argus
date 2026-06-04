"""
ARGUS — AI Visual Market Analyst
Entry point: launches Streamlit UI or FastAPI backend
"""
import sys
import subprocess
from pathlib import Path


def launch_ui():
    """Launch the Streamlit chat interface."""
    ui_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run(["streamlit", "run", str(ui_path), "--server.port", "8501"])


def launch_api():
    """Launch the FastAPI backend server."""
    subprocess.run(["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ui"
    if mode == "api":
        launch_api()
    else:
        launch_ui()
