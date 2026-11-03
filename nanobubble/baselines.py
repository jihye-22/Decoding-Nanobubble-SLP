"""Conventional linear-regression decompositions used as baselines.

All of them fit the *linear* DOS  exp(y) ~ X w  with w >= 0 (the additive relation itself),
as in the original comparison notebook. Only the proposed method compares spectra on a
logarithmic scale.
"""
import warnings

import numpy as np
from scipy.optimize import nnls


def nnls_linear(X, y_log):
    """Non-negative least squares on the linear DOS."""
    w, _ = nnls(X, np.exp(y_log), maxiter=50 * X.shape[1])
    return w



def lasso_cv_positive(X, y_log, cv=5, seed=0):
    """LassoCV(positive=True, no intercept) on the max-normalised linear DOS (settings of the
    original notebook: 200 alphas, max_iter 50 000, tol 1e-4)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import KFold

    Y = np.exp(y_log)
    scale = Y.max()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LassoCV(cv=KFold(cv, shuffle=True, random_state=seed), positive=True, fit_intercept=False,
                        n_alphas=200, max_iter=50000, tol=1e-4, n_jobs=1)
        model.fit(X, Y / scale)
    return model.coef_ * scale


METHODS = {
    "nnls": nnls_linear,
    "lasso_cv": lasso_cv_positive,
}
