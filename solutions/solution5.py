# Use Cross-validation from scikit-learn and compute scores
log_cv_score = cross_val_score(logistic_cv, fmri_masked, labels)
print(log_cv_score)