"""Synthetic regression checks: no benchmark data, checkpoints, or model downloads."""

import ast
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'canonical70k'))
sys.path.insert(0, str(ROOT / 'feature'))

from common import (  # noqa: E402
    assert_disjoint,
    exact_code_hash,
    normalize_code,
    normalized_code_hash,
    refuse_overwrite,
    sha256_file,
)
from code_transforms import (  # noqa: E402
    ast_normalize_code,
    black_format_code,
    remove_comments_only,
)
from extract_handcraft_features import FEATURE_FIELDS, features_for_code  # noqa: E402

spec = importlib.util.spec_from_file_location(
    'canonical_stage4', ROOT / 'scripts/canonical70k/05_stage4_train_evaluate.py'
)
stage4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage4)

SAMPLE = '''def add_values(first_value, second_value):
    """Return a sum with a # literal."""
    # A standalone comment.
    return first_value + second_value  # An inline comment.
'''


class FeatureTests(unittest.TestCase):
    def test_exact_feature_schema_and_finite_values(self):
        self.assertEqual(len(FEATURE_FIELDS), 25)
        self.assertEqual(len(set(FEATURE_FIELDS)), 25)
        for code in (SAMPLE, 'x = 1\n', '', 'def incomplete('):
            with self.subTest(code=code):
                features = features_for_code(code)
                self.assertEqual(set(features), set(FEATURE_FIELDS))
                self.assertTrue(all(math.isfinite(features[k]) for k in FEATURE_FIELDS))

    def test_basic_feature_meanings(self):
        features = features_for_code('x = 1\n')
        self.assertEqual(features['n_lines'], 1)
        self.assertEqual(features['n_chars'], 6)
        self.assertEqual(features['avg_line_len'], 6)
        self.assertEqual(features['n_identifiers'], 1)
        self.assertEqual(features['n_funcs'], 0)
        self.assertEqual(features_for_code(SAMPLE)['n_funcs'], 1)


class IntegrityTests(unittest.TestCase):
    def test_normalization_is_deterministic_and_idempotent(self):
        raw = '\r\nx = 1  \r\n    y = 2\t\r\n'
        expected = 'x = 1\n    y = 2'
        self.assertEqual(normalize_code(raw), expected)
        self.assertEqual(normalize_code(raw), normalize_code(raw))
        self.assertEqual(normalize_code(expected), expected)

    def test_hashes_distinguish_raw_and_normalized_code(self):
        self.assertEqual(
            exact_code_hash('abc'),
            'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
        )
        self.assertEqual(exact_code_hash(SAMPLE), exact_code_hash(SAMPLE))
        self.assertNotEqual(exact_code_hash('x=1\n'), exact_code_hash('x=1  \r\n'))
        self.assertEqual(normalized_code_hash('x=1\n'), normalized_code_hash('x=1  \r\n'))

    def test_file_hash_matches_text_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.txt'
            path.write_bytes(b'abc')
            self.assertEqual(sha256_file(path), exact_code_hash('abc'))

    def test_disjoint_helper_rejects_overlap(self):
        assert_disjoint({'train': {'a'}, 'test': {'b'}}, 'IDs')
        with self.assertRaisesRegex(AssertionError, 'IDs not disjoint'):
            assert_disjoint({'train': {'a'}, 'test': {'a'}}, 'IDs')

    def test_overwrite_guard_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frozen.txt'
            refuse_overwrite(path, False)
            path.write_text('original')
            with self.assertRaises(FileExistsError):
                refuse_overwrite(path, False)
            self.assertEqual(path.read_text(), 'original')
            refuse_overwrite(path, True)
            self.assertEqual(path.read_text(), 'original')


class TransformTests(unittest.TestCase):
    def test_black_output_is_parseable_and_deterministic(self):
        source = 'def add(a,b):\n return a+b\n'
        result = black_format_code(source)
        self.assertIsNotNone(result)
        ast.parse(result)
        self.assertEqual(result, black_format_code(source))
        self.assertEqual(result, black_format_code(result))
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(result)))

    def test_ast_output_is_parseable_and_deterministic(self):
        result = ast_normalize_code(SAMPLE)
        self.assertIsNotNone(result)
        ast.parse(result)
        self.assertEqual(result, ast_normalize_code(SAMPLE))
        self.assertEqual(result, ast_normalize_code(result))
        self.assertEqual(ast.dump(ast.parse(SAMPLE)), ast.dump(ast.parse(result)))

    def test_comment_removal_preserves_strings_and_syntax(self):
        result = remove_comments_only(SAMPLE)
        self.assertIsNotNone(result)
        tree = ast.parse(result)
        self.assertEqual(ast.get_docstring(tree.body[0]), 'Return a sum with a # literal.')
        self.assertNotIn('standalone comment', result)
        self.assertNotIn('inline comment', result)
        self.assertEqual(ast.dump(ast.parse(SAMPLE)), ast.dump(tree))
        self.assertEqual(result, remove_comments_only(SAMPLE))

    def test_invalid_syntax_is_reported(self):
        for transform in (black_format_code, ast_normalize_code, remove_comments_only):
            with self.subTest(transform=transform.__name__):
                self.assertIsNone(transform('def broken('))


class AlignmentTests(unittest.TestCase):
    """Exercise the real Stage-4 loader using tiny fixed-width synthetic arrays."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.features = base / 'features'
        self.embeddings = base / 'embeddings'
        self.manifests = base / 'manifests'
        for directory in (self.features, self.embeddings, self.manifests):
            directory.mkdir()
        self.ids = np.array(['synthetic-a', 'synthetic-b', 'synthetic-c'])
        self.labels = np.array([0, 1, 0])
        self.xf = np.zeros((3, 25), dtype=np.float32)
        self.xe = np.zeros((3, 768), dtype=np.float32)
        np.save(self.features / 'X_test.npy', self.xf)
        np.save(self.features / 'y_test.npy', self.labels)
        np.save(self.features / 'ids_test.npy', self.ids)
        np.save(self.embeddings / 'X_test_codebert.npy', self.xe)
        np.save(self.embeddings / 'ids_test.npy', self.ids)
        pd.DataFrame({'id': self.ids, 'label': self.labels}).to_csv(
            self.manifests / 'test.csv', index=False
        )

    def load(self):
        return stage4.load_split_arrays(
            self.features, self.embeddings, self.manifests, 'test', 'test', 3
        )

    def test_aligned_arrays_load_in_order(self):
        xf, xe, labels, ids = self.load()
        np.testing.assert_array_equal(xf, self.xf)
        np.testing.assert_array_equal(xe, self.xe)
        np.testing.assert_array_equal(labels, self.labels)
        np.testing.assert_array_equal(ids, self.ids)

    def test_embedding_id_reordering_is_rejected(self):
        np.save(self.embeddings / 'ids_test.npy', self.ids[::-1])
        with self.assertRaisesRegex(AssertionError, 'feature IDs != embedding IDs'):
            self.load()

    def test_manifest_id_mismatch_is_rejected(self):
        pd.DataFrame({'id': ['other-a', 'other-b', 'other-c'], 'label': self.labels}).to_csv(
            self.manifests / 'test.csv', index=False
        )
        with self.assertRaisesRegex(AssertionError, 'feature IDs != manifest IDs'):
            self.load()

    def test_label_mismatch_is_rejected(self):
        np.save(self.features / 'y_test.npy', 1 - self.labels)
        with self.assertRaisesRegex(AssertionError, 'feature labels != manifest labels'):
            self.load()

    def test_wrong_feature_width_is_rejected(self):
        np.save(self.features / 'X_test.npy', np.zeros((3, 24)))
        with self.assertRaisesRegex(AssertionError, 'features shape'):
            self.load()


if __name__ == '__main__':
    unittest.main()
