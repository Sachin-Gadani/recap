"""A stand-in Ollama server so the pipeline can be tested without a real model."""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def _notes_for(chunk_text: str) -> dict:
    """Deterministic 'extraction': turn the excerpt into plausible notes."""
    stamps = re.findall(r"\[((?:\d+:)?\d{2}:\d{2})\]", chunk_text) or ["00:00"]
    words = [w for w in re.findall(r"[A-Za-z]{5,}", chunk_text)][:3] or ["discussion"]
    return {
        "excerpt_summary": f"Excerpt covering {stamps[0]} to {stamps[-1]} about {words[0]}.",
        "topics": [{"title": f"{words[0]} at {stamps[0]}", "detail": "Discussed at length.", "timestamp": stamps[0]}],
        "decisions": [{"decision": f"Proceed with {words[0]} as of {stamps[0]}", "rationale": "Team agreed.", "timestamp": stamps[0]}],
        "action_items": [{"task": f"Follow up on {words[0]} from {stamps[-1]}", "owner": "Dana", "due": "Friday", "timestamp": stamps[-1]}],
        "open_questions": [{"question": f"Who owns {words[0]} raised at {stamps[0]}?", "timestamp": stamps[0]}],
        "risks": [{"risk": f"Timeline is tight around {stamps[0]}.", "timestamp": stamps[0]}],
        "facts": [{"fact": f"Budget is 40k, noted at {stamps[0]}.", "timestamp": stamps[0]}],
        "quotes": [{"quote": f"We should ship it, said at {stamps[0]}.", "speaker": "Dana", "timestamp": stamps[0]}],
    }


def _summary_for(notes_text: str) -> dict:
    return {
        "title": "Quarterly Planning Review",
        "one_liner": "The team committed to shipping the pilot before the end of the quarter.",
        "executive_summary": [
            "The pilot ships by 30 September with a 40k budget.",
            "Dana owns follow-up on the outstanding integration work.",
            "Hiring for the second engineer is deferred to next quarter.",
        ],
        "narrative": "The meeting opened with budget.\n\nIt closed with owners assigned.",
        "themes": [
            {"title": "Budget", "bullets": ["40k approved", "No contingency"], "timestamp": "00:00"},
            {"title": "Staffing", "bullets": ["Hiring deferred"], "timestamp": "10:00"},
        ],
        "decisions": [{"decision": "Ship the pilot in September", "rationale": "Customer commitment", "timestamp": "05:00"}],
        "action_items": [
            {"task": "Draft the rollout plan", "owner": "Dana", "due": "Friday", "timestamp": "12:00"},
            {"task": "Confirm the vendor quote | with tax", "owner": "", "due": "", "timestamp": "18:00"},
        ],
        "open_questions": ["Who signs off on the vendor contract?"],
        "risks": ["The September date leaves no slack."],
        "next_steps": ["Reconvene on Monday."],
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence the default stderr spam
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": m} for m in self.server.models]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(request)

        if self.server.fail_next:
            self.server.fail_next -= 1
            self._send({"error": "model is loading"}, 500)
            return

        prompt = request["messages"][-1]["content"]
        if self.server.bad_json_once:
            self.server.bad_json_once = False
            content = "Sure! Here are the notes: {oops not json"
        elif "Executive summary JSON:" in prompt:
            content = json.dumps(_summary_for(prompt))
        elif "Merged JSON:" in prompt:
            content = json.dumps(_notes_for(prompt))
        else:
            content = json.dumps(_notes_for(prompt))
        self._send({"model": request.get("model", "fake"), "message": {"role": "assistant", "content": content}})


class FakeOllama:
    """Context manager yielding a base_url that behaves like Ollama."""

    def __init__(self, models=("llama3.1:8b",)):
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.server.models = list(models)
        self.server.requests = []
        self.server.fail_next = 0
        self.server.bad_json_once = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    @property
    def requests(self) -> list:
        return self.server.requests

    def fail_next(self, count: int = 1) -> None:
        self.server.fail_next = count

    def send_bad_json_once(self) -> None:
        self.server.bad_json_once = True

    def __enter__(self) -> FakeOllama:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
