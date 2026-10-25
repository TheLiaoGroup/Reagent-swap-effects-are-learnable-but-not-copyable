"""Supplementary Table 9 arithmetic, as in REFERENCE_LEARNING build_supplementary_results.py matched_copy().

Reads source_data/Matched_copy_comparison.csv and writes the four headline rows with the
relative gain over the training marginal (percent of the marginal MSE)."""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source_data'

def main():
    data = pd.read_csv(SOURCE / 'Matched_copy_comparison.csv')
    rows = data.loc[data.level.eq('headline')].copy()
    assert len(rows) == 4
    marginal = rows.loc[rows.model.eq('marginal'), 'mse_pp2']
    assert len(marginal) == 1
    rows['relative_gain_percent'] = rows.gain_vs_marginal_pp2 / marginal.iloc[0] * 100
    out = rows[['model', 'model_label', 'mse_pp2', 'gain_vs_marginal_pp2', 'relative_gain_percent', 'target_score_count']]
    out.to_csv(SOURCE / 'Supplementary_Table_9_matched_copy.csv', index=False)
    print(out.to_string(index=False))
if __name__ == '__main__':
    main()
