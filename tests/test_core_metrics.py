"""Behavioral checks for the lightweight, versioned primary metrics."""

import builtins
import unittest
from unittest.mock import patch

from memcore.versions.v1.utils import core_metrics


class CoreMetricsTests(unittest.TestCase):
    def test_historical_and_normalized_f1_are_distinct(self):
        historical = core_metrics._set_f1('"The cat"', "cat", core_metrics.HISTORICAL_V1)
        normalized = core_metrics._set_f1('"The cat"', "cat", core_metrics.NORMALIZED_V1)
        self.assertEqual(historical, 0.0)  # historical quote stays on cat
        self.assertEqual(normalized, 1.0)  # punctuation and article removed
        self.assertEqual(core_metrics._set_f1("the cat", "cat", core_metrics.HISTORICAL_V1), 2 / 3)
        self.assertEqual(core_metrics._set_f1("the cat", "cat", core_metrics.NORMALIZED_V1), 1.0)

    def test_set_f1_does_not_count_duplicate_words(self):
        for version in core_metrics.F1_VERSIONS:
            self.assertEqual(core_metrics._set_f1("cat cat dog", "cat", version), 2 / 3)

    def test_empty_answers_and_invalid_version(self):
        self.assertEqual(core_metrics.calculate_core_metrics("", ""),
                         {"exact_match": 0, "f1": 0.0, "bleu1": 0.0})
        self.assertEqual(core_metrics.calculate_core_metrics("  ", "cat"),
                         {"exact_match": 0, "f1": 0.0, "bleu1": 0.0})
        with self.assertRaisesRegex(ValueError, "Unknown F1 version"):
            core_metrics.calculate_core_metrics("cat", "cat", "unknown")

    def test_bleu_uses_historical_nltk_settings(self):
        with patch.object(core_metrics, "word_tokenize", side_effect=lambda s: s.split()) as tokenize:
            with patch.object(core_metrics, "sentence_bleu", return_value=0.75) as bleu:
                result = core_metrics.calculate_core_metrics("A Cat", "A Cat")
        self.assertEqual(result, {"exact_match": 1, "f1": 1.0, "bleu1": 0.75})
        self.assertEqual([call.args[0] for call in tokenize.call_args_list], ["a cat", "a cat"])
        self.assertEqual(bleu.call_args.args[:2], ([["a", "cat"]], ["a", "cat"]))
        self.assertEqual(bleu.call_args.kwargs["weights"], (1, 0, 0, 0))
        self.assertTrue(callable(bleu.call_args.kwargs["smoothing_function"]))

    def test_missing_nltk_data_is_clear_and_never_downloaded(self):
        with patch.object(core_metrics, "word_tokenize", side_effect=LookupError("punkt_tab missing")):
            with patch("nltk.download") as download:
                with self.assertRaisesRegex(RuntimeError, "punkt.*punkt_tab"):
                    core_metrics.calculate_core_metrics("cat", "cat")
                download.assert_not_called()

    def test_import_has_no_model_metric_dependencies(self):
        # Re-import in a fresh module namespace with heavyweight imports blocked.
        import importlib.util
        from pathlib import Path

        source = Path(core_metrics.__file__)
        real_import = builtins.__import__
        blocked = ("torch", "bert_score", "sentence_transformers")

        def guarded_import(name, *args, **kwargs):
            if name.startswith(blocked):
                raise AssertionError(f"unexpected model import: {name}")
            return real_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location("isolated_core_metrics", source)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=guarded_import):
            spec.loader.exec_module(module)
        self.assertEqual(module.F1_VERSIONS, core_metrics.F1_VERSIONS)


if __name__ == "__main__":
    unittest.main()
