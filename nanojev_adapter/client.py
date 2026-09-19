import json
import urllib.error
import urllib.request


class NanoJevClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(self) -> dict:
        with self.opener.open(f"{self.base_url}/api/health", timeout=min(self.timeout, 10)) as response:
            result = json.load(response)
        if result.get("ready") is not True:
            raise RuntimeError("NanoJev service is not ready")
        return result

    def evaluate(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/api/evaluate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        last_error = None
        for _ in range(2):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    return json.load(response)
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
        raise RuntimeError("NanoJev local inference failed after one retry") from last_error
