Supplementary Table: ML classification benchmark (scaffold split)

File:
  Supplementary_Table_ML_Classification_AUC_F1.csv

Columns included (as requested):
  - roc_auc_oof
  - pr_auc_oof
  - f1_oof
  - roc_auc_test
  - pr_auc_test
  - f1_test

Notes:
  - Metrics are from strict Murcko scaffold holdout + 5-fold scaffold CV.
  - This table reports benchmark-stage results only (multi-model comparison), not final single-model optimization results.
  - Only successful models are listed (n=16).
  - Failed models in this run:
      qda (covariance matrix not full rank at current feature dimension)
      lgbm (runtime callback/iteration error)
