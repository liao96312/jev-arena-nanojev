import json
import os
import urllib.error
import urllib.request


class TypeSafeJevClient:
    def __init__(self, api_key: str | None = None,
                 endpoint: str = "https://api.typesafe.ai/v1/systemone",
                 model: str = "jev-latest", timeout: float = 30):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.endpoint, self.model, self.timeout = endpoint, model, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(self) -> dict:
        return {"ready": bool(self.api_key), "model": self.model}

    def evaluate(self, payload: dict) -> dict:
        if not self.api_key:
            raise RuntimeError("未设置 TYPESAFE_API_KEY")
        states = []
        for state in payload["states"]:
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps({"model": self.model, "state": state["state"],
                                 "questions": state["questions"]}, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
            )
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"Jev API 请求失败（HTTP {exc.code}）") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                raise RuntimeError("Jev API 连接失败") from exc
            states.append({"id": state["id"], "answers": result["answers"]})
        return {"states": states,
                "execution": {"forward_passes": len(states), "network_model_calls": len(states)}}
