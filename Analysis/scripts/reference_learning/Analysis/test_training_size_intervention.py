"""Analytical and design checks for nested training-size interventions."""
import unittest
import numpy as np
from training_size_intervention import nested_samples, predict_raw, score_arrays

class TrainingSizeTests(unittest.TestCase):

    def test_nested_samples_and_full_endpoint(self):
        pool = np.arange(51)
        samples = nested_samples(pool, np.random.default_rng(17))
        self.assertEqual([len(x) for x in samples], [13, 26, 38, 51])
        for a, b in zip(samples, samples[1:]):
            self.assertTrue(set(a) <= set(b))
        np.testing.assert_array_equal(samples[-1], pool)

    def test_prediction_matches_closed_form_and_missing(self):
        mu = np.array([0.2, 0.4, 0.6])
        cov = np.array([[0.1, 0.02, 0.03], [0.02, 0.2, 0.04], [0.03, 0.04, 0.3]])
        fit = dict(mu=mu, cov=cov)
        observed = np.array([[0.3, 0.7], [np.nan, 0.5], [np.nan, np.nan]])
        pred = predict_raw(fit, observed, np.array([0, 1]))
        expected = mu + cov[:, :2] @ np.linalg.solve(cov[:2, :2], observed[0] - mu[:2])
        np.testing.assert_allclose(pred[0], expected)
        np.testing.assert_allclose(pred[1], mu + cov[:, 1] * 0.1 / 0.2)
        np.testing.assert_array_equal(pred[2], mu)

    def test_anchor_only_input_does_not_need_targets(self):
        fit = dict(mu=np.array([0.1, 0.3, 0.5]), cov=np.eye(3) + 0.2)
        visible = np.array([[0.2], [0.4]])
        before = predict_raw(fit, visible.copy(), np.array([0]))
        hidden = np.array([[0.1, 0.3], [0.5, 0.7]])
        hidden[:] = 1000000000.0
        after = predict_raw(fit, visible.copy(), np.array([0]))
        np.testing.assert_array_equal(before, after)
        with self.assertRaises(AssertionError):
            predict_raw(fit, np.zeros((2, 3)), np.array([0]))

    def test_sse_mean_profile_decomposition_and_clipping(self):
        y = np.array([[0.0, 0.3, 1.0], [0.2, np.nan, 0.6]])
        p = np.array([[-0.1, 0.5, 1.2], [0.1, 0.4, 0.9]])
        s = score_arrays(y, p)
        np.testing.assert_allclose(s['sse'], s['profile_sse'] + s['mean_sse'])
        c = score_arrays(y, np.clip(p, 0, 1))
        self.assertTrue(np.all(c['sse'] <= s['sse']))
        np.testing.assert_array_equal(s['n_targets'], [3, 2])
if __name__ == '__main__':
    unittest.main()
