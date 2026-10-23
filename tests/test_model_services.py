"""Check model configs through the real SDK without contacting model services."""

import json
import os
from copy import deepcopy
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from memcore.config import ConfigError, PROJECT_ROOT, load_config
from memcore.runner import validate_inputs
from memcore.versions.v1.agent.memory_layer import LLMController, EmbeddingController


ANSWER_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer", "strict": True,
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                   "required": ["answer"], "additionalProperties": False},
    },
}


class ModelServicesTests(unittest.TestCase):
    def request_with_config(self, filename, response_format=ANSWER_FORMAT):
        environment = {
            "OPENAI_API_KEY": "offline-test-key",
            "MEMCORE_LLM_API_KEY": "offline-test-key",
            "MEMCORE_LLM_BASE_URL": "http://chat.test/v1",
            "MEMCORE_EMBEDDING_API_KEY": "offline-test-key",
            "MEMCORE_EMBEDDING_BASE_URL": "http://embedding.test/v1",
        }
        with patch.dict(os.environ, environment):
            config = load_config(overrides={
                "dataset": str(PROJECT_ROOT / "examples/smoke_locomo.json"),
                "llm_config_path": str(PROJECT_ROOT / "configs" / filename),
            })
            services = validate_inputs(config)["llm"]

        calls = []

        def respond(request):
            body = json.loads(request.content)
            calls.append((request.url, body))
            if request.url.path == "/v1/embeddings":
                return httpx.Response(200, json={
                    "object": "list", "model": body["model"],
                    "data": [{"object": "embedding", "index": 0, "embedding": [1.0] * 1024}],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                })
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"answer":"Paris"}'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })

        def client(**kwargs):
            result = OpenAI(http_client=httpx.Client(transport=httpx.MockTransport(respond)), **kwargs)
            self.addCleanup(result.close)
            return result

        with patch("openai.OpenAI", side_effect=client):
            chat = LLMController(**services["llm"]).llm
            embedding = EmbeddingController(**services["embedding"])
        answer = chat.get_completion("Where?", response_format=response_format, max_tokens=512)
        vector = embedding.get_embedding("Paris")
        self.assertEqual(json.loads(answer), {"answer": "Paris"})
        self.assertEqual(len(vector), 1024)
        self.assertEqual(calls[0][1]["seed"], 42)
        self.assertEqual(calls[0][1]["max_tokens"], 512)
        self.assertEqual(calls[1][0].host, "embedding.test")
        self.assertEqual(calls[1][1]["model"], "Qwen/Qwen3-Embedding-0.6B")
        return calls[0]

    def test_qwen_configs_keep_non_thinking_requests(self):
        for filename, model in (("llm.yaml", "Qwen/Qwen3-8B"),
                                ("llm_qwen3_14b.yaml", "Qwen/Qwen3-14B")):
            with self.subTest(model=model):
                url, body = self.request_with_config(filename)
                self.assertEqual(url.host, "chat.test")
                self.assertEqual(body["model"], model)
                self.assertEqual(body["response_format"], ANSWER_FORMAT)
                self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
                self.assertTrue(body["messages"][0]["content"].endswith("/no_think"))

    def test_gpt4o_uses_official_endpoint_without_qwen_parameters(self):
        url, body = self.request_with_config("llm_gpt4o.yaml")
        self.assertEqual(url.host, "api.openai.com")
        self.assertEqual(body["model"], "gpt-4o")
        self.assertEqual(body["response_format"], ANSWER_FORMAT)
        self.assertNotIn("chat_template_kwargs", body)
        self.assertNotIn("/no_think", body["messages"][0]["content"])

    def test_gpt4o_requires_nested_link_reasons_without_changing_qwen_schema(self):
        response_format = deepcopy(ANSWER_FORMAT)
        link = {
            "type": "object", "additionalProperties": False,
            "properties": {"target_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["target_id"],
        }
        schema = response_format["json_schema"]["schema"]
        schema["properties"]["new_links"] = {"type": "array", "items": link}
        schema["required"].append("new_links")
        original = deepcopy(response_format)
        _, gpt_body = self.request_with_config("llm_gpt4o.yaml", response_format)
        _, qwen_body = self.request_with_config("llm.yaml", response_format)
        sent_link = gpt_body["response_format"]["json_schema"]["schema"]["properties"]["new_links"]["items"]
        self.assertEqual(sent_link["required"], ["target_id", "reason"])
        self.assertEqual(qwen_body["response_format"], original)
        self.assertEqual(response_format, original)

    def test_full_metrics_cannot_be_selected_in_release_config(self):
        with self.assertRaisesRegex(ConfigError, "metrics_mode"):
            load_config(overrides={"metrics_mode": "full"})


if __name__ == "__main__":
    unittest.main()
