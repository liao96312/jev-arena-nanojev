import io
import json
import unittest

from nanojev_adapter.typesafe_client import TypeSafeJevClient


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FakeOpener:
    def __init__(self):
        self.request = None

    def open(self, request, timeout):
        self.request = request
        return FakeResponse(json.dumps({"answers": {"action": {
            "type": "choice", "choice": "wait", "confidence": .8,
            "probabilities": {"wait": .8, "move_n": .2}}}}).encode())


class TypeSafeJevClientTests(unittest.TestCase):
    def test_translates_typesafe_response_to_existing_agent_contract(self):
        client = TypeSafeJevClient(api_key="test-key")
        client.opener = FakeOpener()
        response = client.evaluate({"states": [{"id": "s1", "state": "HP=100",
                                                  "questions": {"action": {"type": "choice",
                                                  "instructions": "Choose", "criteria": {
                                                      "wait": "Wait", "move_n": "Move N"}}}}]})
        self.assertEqual(response["states"][0]["id"], "s1")
        self.assertEqual(response["states"][0]["answers"]["action"]["probabilities"]["wait"], .8)
        body = json.loads(client.opener.request.data)
        self.assertEqual(body["model"], "jev-latest")
        self.assertNotIn("states", body)


if __name__ == "__main__":
    unittest.main()
