# Scikit-learn has tools to perform cross-validation in one line.
# Explore "cross_val_score" function and evaluate a score by cross-validation 
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(svc, fmri_masked, labels)
print(cv_score)