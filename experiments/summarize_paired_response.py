import argparse
import csv
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
args = parser.parse_args()
records = [json.loads(p.read_text()) for p in sorted(args.results.glob('seed-*/*alpha*.json'))]
metadata = json.loads(next(args.results.glob('seed-*/metadata.json')).read_text())
config = metadata['configuration']
expected = {(p['id'], a, s) for p in config['prompts'] for a in config['alpha'] for s in config['seeds']}
assert {(r['prompt'], r['alpha'], r['seed']) for r in records} == expected
summary = []
for alpha in config['alpha']:
    subset = [r for r in records if r['alpha'] == alpha]
    row = {'alpha': alpha, 'records': len(subset)}
    for condition in ['matched_zero', 'unmatched_zero', 'matched_pulse', 'unmatched_pulse']:
        for metric in ['norm', 'complexity']:
            values = [r['metrics'][condition][metric] for r in subset]
            row[f'{condition}_{metric}_mean'] = float(np.mean(values))
    row['nll_mean'] = float(np.mean([r['language']['nll'] for r in subset]))
    row['accuracy_mean'] = float(np.mean([r['language']['accuracy'] for r in subset]))
    summary.append(row)
with (args.results/'summary.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(summary)
print(json.dumps({
    'records': len(records),
    'matched_zero_nonzero': sum(r['metrics']['matched_zero']['norm'] != 0 for r in records),
    'unmatched_zero_nonzero': sum(r['metrics']['unmatched_zero']['norm'] != 0 for r in records),
    'unmatched_zero_nonzero_complexity': sum(r['metrics']['unmatched_zero']['complexity'] != 0 for r in records),
    'intact_nonzero_pulse_norm': sum(r['alpha'] == 1 and r['metrics']['matched_pulse']['norm'] > 0 for r in records),
    'intact_nonzero_complexity': sum(r['alpha'] == 1 and r['metrics']['matched_pulse']['complexity'] > 0 for r in records),
}, indent=2))
