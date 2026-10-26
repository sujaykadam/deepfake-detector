# genvid_kling_vs_vript experiment

- videos with a face: 203/204
- windows: 609 (real 456, fake 153), window 3.0s step 1.0s
- GroupKFold(5) by person_id
- window accuracy 0.605 (chance 0.749), ROC AUC 0.655 (permutation p = 0.005)
- video accuracy (majority vote) 0.635, balanced 0.652, ROC AUC 0.701
- window balanced accuracy 0.616
- per-feature ROC AUC (0.5 = no signal): {'mean_coherence': 0.3383786263043229, 'min_coherence': 0.3764906547414288, 'hr_spread_bpm': 0.5761595573902075, 'quality': 0.373552344914574}
- paired real-vs-fake: {'pairs': 0}
