"""Offline tests for bounded query-intent decoding and cost metadata."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from memcore.versions.v1.agent import memory_layer
from memcore.versions.v1.agent.key_graph_memory_v3 import KeyGraphMemoryV3
from memcore.versions.v1.storage.graph_store import InMemoryGraphStore


class FakeTracker:
    def __init__(self):
        self.records = []

    def record(self, **kwargs):
        self.records.append(kwargs)


class FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish_reason = self.outputs.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                     finish_reason=finish_reason)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))


def make_system(outputs):
    completions = FakeCompletions(outputs)
    tracker = FakeTracker()
    controller = memory_layer.OpenAIController.__new__(memory_layer.OpenAIController)
    controller.model = 'fake'
    controller.extra_body = None
    controller.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    controller._token_tracker = tracker
    controller._current_caller = 'previous_caller'
    controller._json_schema_supported = True
    controller._last_completion_metadata = None
    system = KeyGraphMemoryV3.__new__(KeyGraphMemoryV3)
    system.prompt_loader = SimpleNamespace(get_query_intent_prompt=lambda: lambda query: 'Original prompt: ' + query)
    system.llm_controller = SimpleNamespace(llm=controller)
    system.store = InMemoryGraphStore()
    system.memories = {}
    system.last_query_intent_trace = None
    system.last_retrieval_trace = None
    return system, controller, completions, tracker


class QueryIntentTests(unittest.TestCase):
    def test_first_valid_response_keeps_original_prompt_schema_and_temperature(self):
        content = json.dumps({'topic_desc': 'Writing class', 'keywords': ['Maria', 'writing']})
        system, controller, completions, tracker = make_system([(content, 'stop')])
        result = system._analyze_query_intent('What writing classes?', allow_empty_keywords=True)
        self.assertEqual(result['time_sorting'], 'none')
        self.assertEqual(result['keywords'], ['Maria', 'writing'])
        self.assertEqual(len(completions.calls), 1)
        call = completions.calls[0]
        self.assertEqual(call['messages'][-1]['content'], 'Original prompt: What writing classes?')
        self.assertEqual(call['response_format']['type'], 'json_schema')
        self.assertEqual(call['temperature'], 0.7)
        self.assertEqual(call['max_tokens'], 512)
        self.assertEqual(controller._current_caller, 'previous_caller')
        self.assertEqual(tracker.records[0]['caller'], 'query_intent')
        self.assertEqual(controller.last_completion_metadata, {
            'output_characters': len(content), 'finish_reason': 'stop', 'usage_recorded': True})
        metadata = controller.last_completion_metadata
        metadata['output_characters'] = -1
        self.assertEqual(controller.last_completion_metadata['output_characters'], len(content))
        with self.assertRaises(AttributeError):
            controller.last_completion_metadata = {}
        self.assertEqual(system.last_query_intent_trace[0]['retry_reason'], None)

    def test_whitespace_and_truncated_json_retry_with_bounded_budget(self):
        valid = '{"topic_desc":"Screenplay","keywords":["Joanna","screenplay"]}'
        system, controller, completions, tracker = make_system([
            ('\n' * 400, 'length'), ('{"topic_desc":', 'length'), (valid, 'stop')])
        result = system._analyze_query_intent('When was the first screenplay?')
        self.assertEqual(result['topic_desc'], 'Screenplay')
        self.assertEqual([call['max_tokens'] for call in completions.calls], [512] * 3)
        self.assertEqual([call['response_format']['type'] for call in completions.calls],
                         ['json_schema', 'json_schema', 'json_object'])
        self.assertIn('compact JSON object', completions.calls[1]['messages'][-1]['content'])
        self.assertIn('"required"', completions.calls[2]['messages'][-1]['content'])
        self.assertEqual([entry['retry_reason'] for entry in system.last_query_intent_trace],
                         ['empty_response', 'json_decode_error', None])
        self.assertEqual([entry['finish_reason'] for entry in system.last_query_intent_trace],
                         ['length', 'length', 'stop'])
        self.assertEqual([entry['output_characters'] for entry in system.last_query_intent_trace],
                         [400, 14, len(valid)])
        self.assertEqual(len(tracker.records), 3)
        self.assertTrue(all(record['caller'] == 'query_intent' for record in tracker.records))
        self.assertEqual(controller._current_caller, 'previous_caller')

    def test_invalid_json_shape_retries_and_never_fabricates_intent(self):
        system, controller, completions, tracker = make_system([
            ('[]', 'stop'), ('{"topic_desc":" ","keywords":["x"]}', 'stop'),
            ('{"topic_desc":"Topic","keywords":"not a list"}', 'stop')])
        with self.assertRaisesRegex(RuntimeError, 'after 3 bounded attempts'):
            system._analyze_query_intent('Q')
        self.assertEqual(len(completions.calls), 3)
        self.assertEqual(len(tracker.records), 3)
        self.assertEqual(controller._current_caller, 'previous_caller')
        self.assertEqual([entry['retry_reason'] for entry in system.last_query_intent_trace],
                         ['invalid_object', 'invalid_topic_desc', 'invalid_keywords'])
        self.assertFalse(any('response' in entry for entry in system.last_query_intent_trace))

    def test_empty_keywords_only_allowed_when_explicit_and_trace_reaches_retrieval(self):
        valid = '{"topic_desc":"General subject","keywords":[]}'
        system, _, _, _ = make_system([(valid, 'stop')])
        self.assertEqual(system.retrieve_paper_structured(
            'Q', enable_topic_path=False, enable_keyword_path=False), [])
        self.assertEqual(system.last_retrieval_trace['query_intent_attempts'],
                         system.last_query_intent_trace)
        system, _, _, _ = make_system([(valid, 'stop')] * 3)
        with self.assertRaises(RuntimeError):
            system._analyze_query_intent('Q', allow_empty_keywords=False)

    def test_ollama_missing_finish_reason_is_supported(self):
        controller = memory_layer.OllamaController.__new__(memory_layer.OllamaController)
        controller.model = 'fake'
        controller._last_completion_metadata = None
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"topic_desc":"x","keywords":[]}'))])
        with patch.object(memory_layer, 'completion', return_value=response) as completion:
            content = controller.get_completion('Q', response_format={'type': 'json_object'},
                                                max_tokens=512)
        self.assertTrue(content.startswith('{'))
        self.assertEqual(completion.call_args.kwargs['max_tokens'], 512)
        self.assertIsNone(controller.last_completion_metadata['finish_reason'])


if __name__ == '__main__':
    unittest.main()
