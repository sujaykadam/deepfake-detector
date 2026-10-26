# Matched real/fake pairs per generator

- 41 pairs (82 clips); each real clip cut to its fake's fps and duration; whole clip = one window
- pooled classifier (GroupKFold by pair): accuracy 0.585, AUC 0.635 (95% bootstrap CI 0.51-0.75), permutation p = 0.035
- accuracy by model: {"Kling": 0.7, "ModelScope": 0.65, "Pika": 0.45, "SVD": 0.6, "Sora": 0.0}
- AUC by model: {"Kling": 0.72, "ModelScope": 0.75, "Pika": 0.59, "SVD": 0.61}

## Pooled per-feature paired comparison

- mean_coherence: AUC 0.437, real higher in 23/41 pairs, sign-test p 0.533, mean real 0.608 / fake 0.567
- min_coherence: AUC 0.504, real higher in 19/41 pairs, sign-test p 0.755, mean real 0.464 / fake 0.450
- hr_spread_bpm: AUC 0.580, real higher in 8/41 pairs, sign-test p 0.024, mean real 27.213 / fake 36.857
- quality: AUC 0.616, real higher in 18/41 pairs, sign-test p 0.533, mean real 0.145 / fake 0.172
- band_fraction: AUC 0.389, real higher in 24/41 pairs, sign-test p 0.349, mean real 0.094 / fake 0.062
- frame_to_frame_std: AUC 0.440, real higher in 25/41 pairs, sign-test p 0.211, mean real 5.724 / fake 5.382
- brightness: AUC 0.503, real higher in 21/41 pairs, sign-test p 1.000, mean real 130.004 / fake 129.109
