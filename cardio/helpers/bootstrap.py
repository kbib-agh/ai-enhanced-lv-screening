import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_roc_test(y_test, y_proba, y_proba2):
    y_test = np.array(y_test)
    y_proba = np.array(y_proba)
    y_proba2 = np.array(y_proba2)

    rng = np.random.default_rng(0)
    n_boot = 2000
    idx = np.arange(len(y_test))
    diffs = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        diffs.append(roc_auc_score(y_test[b], y_proba[b]) - roc_auc_score(y_test[b], y_proba2[b]))
    diffs = np.array(diffs)
    # two-sided p-value
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    ci = np.quantile(diffs, [0.025, 0.975])  # 95% CI for AUC difference
    return ci
