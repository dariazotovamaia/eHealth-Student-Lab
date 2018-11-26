# Perform classification by using Logistic Regression classifier.
from sklearn.linear_model import LogisticRegression

# Try to use it with default parameters 
logistic_cv = LogisticRegression(C=1., penalty="l1", solver='liblinear')

# Train the model
logistic_cv.fit(X_train, Y_train)

# Test the model
prediction = logistic_cv.predict(X_test)
# Compute accuracy of Logistic Regression classifier
accuracy = (prediction == Y_test).sum() / float(len(Y_test))
print(accuracy)