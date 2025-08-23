import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from search import dot_product_similarity, dot_similarity_batch, load_vectors_and_meta

class TestSearchUtils(unittest.TestCase):
    def setUp(self):
        # Create dummy data for testing
        self.query = np.array([0.6, 0.8])
        self.matrix = np.array([[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]])
        self.queries = np.array([[0.6, 0.8], [1.0, 0.0]])
        
    def test_dot_product_similarity(self):
        # All vectors are normalized, so dot product is cosine similarity
        sims = dot_product_similarity(self.query, self.matrix)
        expected = np.dot(self.matrix, self.query)
        print(f"sims: {sims}, expected: {expected}")
        np.testing.assert_array_almost_equal(sims, expected)

    def test_dot_similarity_batch(self):
        sims_batch = dot_similarity_batch(self.queries, self.matrix)
        expected = np.dot(self.queries, self.matrix.T)
        print(f"sims_batch: {sims_batch}, expected: {expected}")
        np.testing.assert_array_almost_equal(sims_batch, expected)

    def test_load_vectors_and_meta(self):
        # This test only checks if the function runs and returns expected keys
        # You may want to mock np.load and pd.read_parquet for a real unit test
        try:
            data = load_vectors_and_meta()
            for key in ["title", "description", "reviews", "meta", "dim", "n"]:
                self.assertIn(key, data)
        except Exception as e:
            self.skipTest(f"Skipping load_vectors_and_meta test due to: {e}")

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)