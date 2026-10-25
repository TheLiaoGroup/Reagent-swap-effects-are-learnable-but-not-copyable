"""Focused invariants for shared latent spectrum decomposition."""
import unittest
import numpy as np
import pandas as pd
import shared_latent_dimension as d

class SharedLatentDimensionTests(unittest.TestCase):

    def test_spectrum_profile_is_normalized_and_rank_metrics_are_bounded(self):
        matrix = np.array([[0.1, 0.2, 0.4, 0.3], [0.2, 0.4, 0.8, 0.6], [0.8, 0.3, 0.1, 0.2], [0.4, 0.6, 0.2, 0.1]])
        profile, metrics = d.spectrum_profile(matrix, max_rank=8)
        self.assertAlmostEqual(float(profile.sum()), 1.0, places=12)
        self.assertGreaterEqual(metrics['k_var90'], 1)
        self.assertLessEqual(metrics['k_var90'], 4)
        self.assertGreater(metrics['participation_rank'], 0.0)

    def test_support_metrics_keep_missing_and_zero_distinct(self):
        matrix = np.array([[0.0, 0.2, np.nan], [0.0, 0.0, np.nan]])
        metrics = d.support_metrics(matrix)
        self.assertEqual(metrics['finite_cells'], 4)
        self.assertAlmostEqual(metrics['missing_fraction'], 2 / 6)
        self.assertAlmostEqual(metrics['zero_fraction_observed'], 3 / 4)
        self.assertAlmostEqual(metrics['all_zero_pair_fraction'], 0.5)

    def test_family_effect_is_identifiable_only_for_repeated_family(self):
        screen_df = pd.DataFrame({'screen': ['A|1', 'A|2', 'B|1'], 'family': ['A', 'A', 'B']})
        spectrum_rows = []
        profiles = {'A|1': [0.6, 0.3, 0.1], 'A|2': [0.5, 0.3, 0.2], 'B|1': [0.2, 0.3, 0.5]}
        for screen, values in profiles.items():
            for rank, value in enumerate(values, 1):
                spectrum_rows.append({'screen': screen, 'family': screen.split('|')[0], 'profile': 'observed_all', 'rank': rank, 'variance_fraction': value, 'cumulative_fraction': sum(values[:rank])})
        spectrum_df = pd.DataFrame(spectrum_rows)
        support_df = pd.DataFrame({'screen': list(profiles), 'family': ['A', 'A', 'B']})
        _, loo, extras = d.build_pooled_tables(screen_df, spectrum_df, support_df)
        self.assertEqual(int(loo.family_reconstruction_identifiable.sum()), 2)
        self.assertTrue(pd.isna(loo.loc[loo.screen.eq('B|1'), 'family_profile_rmse']).all())
        self.assertFalse(extras['variance'].empty)
if __name__ == '__main__':
    unittest.main()
