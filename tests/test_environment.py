import asyncio
import json
import unittest
from types import SimpleNamespace
from tempfile import TemporaryFile

import httpx
from openai import AsyncOpenAI

from luma_gym import load_environment
from luma_gym.documents import CaseDocuments
from luma_gym.eval import RecordingClient
from luma_gym.tasks import generate_tasks


class EnvironmentTests(unittest.TestCase):
    def test_rollout_and_tool_isolation(self):
        tasks = generate_tasks(3, 17)
        case = CaseDocuments(tasks[1])
        other = CaseDocuments(tasks[2])
        with self.assertRaises(ValueError):
            case.read_document(other.list_documents()[0]["id"])
        for mode in ("single", "tools"):
            env = load_environment(mode=mode, n=3)
            row = env.get_dataset()[1]
            self.assertEqual(row["answer"], "")
            if mode == "tools":
                visible = json.dumps(row["prompt"])
                self.assertNotIn(tasks[1].record["gross_pay"], visible)
                self.assertNotIn("0.06", visible)
            counter = 0

            def respond(request):
                nonlocal counter
                counter += 1
                payload = json.loads(request.content)
                if mode == "tools" and counter == 1:
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call-list", "type": "function", "function": {"name": "list_documents", "arguments": "{}"}},
                    ]}
                    finish = "tool_calls"
                elif mode == "tools" and counter == 2:
                    documents = json.loads(payload["messages"][-1]["content"])
                    calls = []
                    for index, document in enumerate(documents):
                        calls.append({"id": f"call-{index}", "type": "function", "function": {
                            "name": "read_document", "arguments": json.dumps({"document_id": document["id"]}),
                        }})
                    calls.append({"id": "call-rates", "type": "function", "function": {
                        "name": "lookup_reference", "arguments": '{"reference":"first-half"}',
                    }})
                    message = {"role": "assistant", "content": None, "tool_calls": calls}
                    finish = "tool_calls"
                else:
                    message = {"role": "assistant", "content": json.dumps(tasks[1].ground_truth)}
                    finish = "stop"
                return httpx.Response(200, json={
                    "id": "mock", "object": "chat.completion", "created": 0, "model": "mock",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                })

            async def run():
                async with AsyncOpenAI(api_key="offline-test", http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(respond),
                )) as client:
                    with TemporaryFile(mode="w+") as log:
                        recording = RecordingClient(client, log, {"input": 0.000001, "output": 0.000002})
                        output = await env.run_rollout(row, recording, "mock", {"temperature": 0.7},
                                                       state_columns=["payroll_score"])
                        self.assertTrue(recording.known_cost)
                        self.assertGreater(recording.cost, 0)
                        return output

            output = asyncio.run(run())
            self.assertEqual(output["reward"], 1.0)
            self.assertEqual(output["payroll_score"]["score"], 1.0)
            self.assertEqual(counter, 3 if mode == "tools" else 1)
            if mode == "tools":
                state = {"info": row["info"]}
                message = SimpleNamespace(tool_calls=[SimpleNamespace(
                    id="bad-call", name="read_document", arguments='{"document_id":"not-this-case"}',
                )])
                reply = asyncio.run(env.env_response([message], state))
                self.assertEqual(json.loads(reply[0].content), {"error": "Invalid tool request"})
                self.assertEqual(state["payroll_tool_errors"], 1)


if __name__ == "__main__":
    unittest.main()
