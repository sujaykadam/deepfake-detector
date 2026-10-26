# kling_full experiment

- videos with a face: 104/104
- windows: 312 (real 207, fake 105), window 3.0s step 1.0s
- GroupKFold(5) by person_id
- window accuracy 0.586 (chance 0.663), ROC AUC 0.595 (permutation p = 0.040)
- video accuracy (majority vote) 0.644, balanced 0.661, ROC AUC 0.639
- video ROC AUC 95% CI 0.537-0.733
- window balanced accuracy 0.597
- per-feature ROC AUC (0.5 = no signal): {'mean_coherence': 0.4034966643662296, 'min_coherence': 0.422866344605475, 'hr_spread_bpm': 0.5463077984817115, 'quality': 0.3979756153669197}
- paired real-vs-fake: {'pairs': 35, 'hr_bpm': {'mean_real': 60.00040405797101, 'mean_fake': 61.71467657142857, 'mean_abs_pair_diff': 15.30440985507247, 'wilcoxon_p': 0.4118947303681507}, 'hr_quality': {'mean_real': 0.17892263882143195, 'mean_fake': 0.1839442684507089, 'mean_abs_pair_diff': 0.03845147150119525, 'wilcoxon_p': 0.3843350024224179}}
