"""Query evolution response parsing without model or network calls."""

import ast
from pathlib import Path
import re
from typing import Optional
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "src/memcore/versions/v1/agent/key_agent_v3.py"


def agent_type():
    """Load the production method without importing graph/model dependencies."""
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "KeyToGAgentV3")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == "_evolve_query"]
    namespace = {"Optional": Optional, "re": re}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["KeyToGAgentV3"]


KeyToGAgentV3 = agent_type()


class QueryEvolutionTests(unittest.TestCase):
    def setUp(self):
        self.agent = object.__new__(KeyToGAgentV3)
        self.agent.logger = Mock()
        self.agent.prompt_loader = Mock()
        self.agent.prompt_loader.get_subquery_prompt.return_value = (
            lambda query_str, context_str, prev_reasoning, missing_info:
            f"{query_str}|{context_str}|{prev_reasoning}|{missing_info}")
        self.completion = Mock()
        self.agent.llm = Mock()
        self.agent.llm.llm.get_completion = self.completion

    def evolve(self, response):
        self.completion.return_value = response
        result = self.agent._evolve_query(
            question="What did Gina find?", missing_info="Find the object",
            prev_reasoning="None", current_context="Gina visited her store")
        self.completion.assert_called_once_with(
            "What did Gina find?|Gina visited her store|None|Find the object",
            response_format=None, temperature=0.5)
        self.assertEqual(self.agent.llm.llm._current_caller, "query_evolution")
        self.completion.reset_mock()
        return result

    def test_labeled_stop_tokens_terminate(self):
        for response in (
            "Sub-question: None", "Output: None", " sub-query : 'None' ",
            'NEXT QUESTION: "none"', "Question: none.", "Subquery: None",
            "Output: Question: None", "Next question: Output: 'None'",
            "Sub-question:   ", "None", "'None'", "None.", "  ",
        ):
            with self.subTest(response=response):
                self.assertIsNone(self.evolve(response))

    def test_labeled_query_keeps_its_meaning(self):
        query = "What specific place did Gina find for her store?"
        for response in (f"Sub-question: {query}", f"  sub-QUERY :  {query}  ",
                         f"Next question: {query}",
                         f"Output: Question: {query}",
                         f"Next question: Output: {query}", query):
            with self.subTest(response=response):
                self.assertEqual(self.evolve(response), query)


if __name__ == "__main__":
    unittest.main()
