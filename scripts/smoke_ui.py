"""Brief local HTTP/UI startup verification, no training or generated examples."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from app import build_app

demo = build_app()
try:
    _, url, _ = demo.launch(server_name="127.0.0.1", server_port=7866, prevent_thread_lock=True, quiet=True)
    with httpx.Client(trust_env=False) as client:
        home = client.get(url)
        config = client.get(url + "config")
    assert home.status_code == 200 and config.status_code == 200
    print(json.dumps({"http": home.status_code, "config": config.status_code, "components": len(config.json()["components"])}))
finally:
    demo.close()
