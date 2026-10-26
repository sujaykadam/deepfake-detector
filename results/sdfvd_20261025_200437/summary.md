# sdfvd experiment

- videos with a face: 101/106
- windows: 197 (real 99, fake 98), window 2.0s step 1.0s
- GroupKFold(5) by person_id
- window accuracy 0.497 (chance 0.503), ROC AUC 0.495
- video accuracy (majority vote) 0.495, ROC AUC 0.477
- per-feature ROC AUC (0.5 = no signal): {'mean_coherence': 0.5202020202020202, 'min_coherence': 0.5394763966192538, 'hr_spread_bpm': 0.5041743970315399, 'quality': 0.46897546897546893}
- paired real-vs-fake: {'pairs': 50, 'hr_bpm': {'mean_real': 63.22041567585409, 'mean_fake': 64.21084444667783, 'mean_abs_pair_diff': 11.668000247587232, 'wilcoxon_p': 0.49932054210353793}, 'hr_quality': {'mean_real': 0.2676657582841305, 'mean_fake': 0.2579922290849831, 'mean_abs_pair_diff': 0.04523675494489225, 'wilcoxon_p': 0.2260100302518655}}
