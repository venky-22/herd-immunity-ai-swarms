"""Tiny client for Adaption's MCP server (JSON-RPC over HTTP). Usage: python finetune/adaption.py <tool> '<json args>'"""
import json, os, sys
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
URL = "https://api.prod.adaptionlabs.ai/api/v1/mcp"
HDR = {"Authorization": f"Bearer {os.environ['ADAPTION_API_KEY']}", "Content-Type": "application/json",
       "Accept": "application/json, text/event-stream"}


def call(tool: str, args: dict | None = None):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args or {}}}
    r = httpx.post(URL, headers=HDR, json=body, timeout=120)
    r.raise_for_status()
    data = next((l[6:] for l in r.text.splitlines() if l.startswith("data: ")), r.text)
    res = json.loads(data)
    if "error" in res:
        raise RuntimeError(res["error"])
    out = res["result"]
    texts = [c["text"] for c in out.get("content", []) if c.get("type") == "text"]
    try:
        return json.loads(texts[0]) if len(texts) == 1 else texts
    except json.JSONDecodeError:
        return texts


if __name__ == "__main__":
    print(json.dumps(call(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}), indent=1)[:6000])


REST = "https://api.prod.adaptionlabs.ai/api/v1"


def upload_csv(path: str, name: str, fmt: str = "csv") -> dict:
    """Presigned-S3 upload: initiate -> PUT bytes -> complete. Returns the complete() response (has dataset_id)."""
    auth = {"Authorization": HDR["Authorization"]}
    p = Path(path)
    init = httpx.post(f"{REST}/datasets/upload/initiate", headers=auth, timeout=60,
                      json={"name": name, "file_format": fmt}).json()
    url = init["upload_url"]
    put = httpx.put(url, content=p.read_bytes(), timeout=600)
    put.raise_for_status()
    s3_key = init.get("s3_key") or url.split(".amazonaws.com/", 1)[1].split("?", 1)[0]
    done = httpx.post(f"{REST}/datasets/upload/complete", headers=auth, timeout=60,
                      json={"s3_key": s3_key, "name": name, "file_format": fmt,
                            "file_size_bytes": p.stat().st_size})
    if done.status_code >= 400:
        raise RuntimeError(f"complete failed {done.status_code}: {done.text[:500]}")
    return done.json()
