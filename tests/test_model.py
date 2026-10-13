"""Small deterministic model/control checks; no external dataset or webcam required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

import brain
import circuit
from swat import shuffled


class SimulatorTests(unittest.TestCase):
    def test_rest_is_silent(self):
        model = brain.LIF(sp.csr_matrix((3, 3)), np.array([0]))
        for _ in range(30):
            self.assertEqual(model.step(np.array([False])).size, 0)
        np.testing.assert_allclose(model.v, brain.V_REST)

    def test_input_spike_propagates_only_to_connected_target(self):
        weights = sp.csr_matrix(([100.0], ([0], [1])), shape=(3, 3))
        model = brain.LIF(weights, np.array([0]))
        events = []
        for step in range(50):
            events.extend((step, int(neuron)) for neuron in model.step(np.array([step == 0])))
        self.assertIn((1, 0), events)
        target_steps = [step for step, neuron in events if neuron == 1]
        self.assertTrue(target_steps)
        self.assertGreaterEqual(min(target_steps), 1 + model.delay)
        self.assertFalse(any(neuron == 2 for _, neuron in events))

    def test_shuffle_preserves_each_sources_weights(self):
        original = sp.csr_matrix([[0, 2, -3], [4, 0, 0], [0, 5, 0]])
        changed = shuffled(original, seed=3)
        np.testing.assert_array_equal(np.diff(original.indptr), np.diff(changed.indptr))
        for i in range(original.shape[0]):
            np.testing.assert_array_equal(np.sort(original.getrow(i).data),
                                          np.sort(changed.getrow(i).data))
        np.testing.assert_array_equal(changed.toarray(), shuffled(original, seed=3).toarray())

    def test_path_search_honors_direction_and_cutoff(self):
        adjacency = sp.csr_matrix((np.ones(5), (np.arange(1, 6), np.arange(5))), shape=(6, 6))
        np.testing.assert_array_equal(circuit.steps(adjacency, [0])[:5], np.arange(5))
        self.assertTrue(np.isinf(circuit.steps(adjacency, [0])[5]))
        self.assertTrue(np.isinf(circuit.steps(adjacency, [5])[0]))


class CacheTests(unittest.TestCase):
    def test_bundled_cache_works_without_annotations(self):
        weights, ids = circuit.escape_circuit()
        with patch.object(brain, 'cell_types', side_effect=AssertionError('external data read')):
            types = circuit.cell_types(ids)
        self.assertEqual(weights.shape, (12510, 12510))
        self.assertEqual(weights.nnz, 647558)
        self.assertEqual(int((types == 'LPLC2').sum()), 185)
        self.assertEqual(int((types == 'DNp01').sum()), 2)

    def test_metadata_rejects_reordered_ids(self):
        _, ids = circuit.escape_circuit()
        with self.assertRaisesRegex(ValueError, 'does not match'):
            circuit.cell_types(ids[::-1])

    def test_rebuilt_metadata_roundtrip(self):
        ids = np.array([100, 101, 103])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata.json'
            circuit.save_metadata(ids, ['LPLC2', None, 'DNp01'], path)
            with patch.object(circuit, 'METADATA', path):
                types = circuit.cell_types(ids)
                self.assertEqual(types.iloc[[0, 2]].tolist(), ['LPLC2', 'DNp01'])
                self.assertTrue(types.isna().iloc[1])
                metadata = json.loads(path.read_text())
                metadata['cell_types']['LPLC2'] = [999]
                path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'unknown neuron'):
                    circuit.cell_types(ids)

    def test_matrix_roundtrip(self):
        weights = sp.csr_matrix([[0, -3], [4, 0]])
        ids = np.array([42, 99])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'matrix.npz'
            brain.save_matrix(path, weights, ids)
            restored, restored_ids = brain.load_matrix(path)
        np.testing.assert_array_equal(restored.toarray(), weights.toarray())
        np.testing.assert_array_equal(restored_ids, ids)


if __name__ == '__main__':
    unittest.main()
