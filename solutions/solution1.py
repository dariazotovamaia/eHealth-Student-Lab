# Manually split data into training and test sets
# Choose how many examples you want to put for the training set and put the rest for the test set

# Let's put last 30 examples into test set 
X_train = fmri_masked[:-30]
Y_train = labels[:-30]
X_test = fmri_masked[-30:]
Y_test = labels[-30:]

# Train the model
svc.fit(X_train,Y_train)

# Test the model
prediction = svc.predict(X_test)

# Compute accuracy of SVC classifier
accuracy = (prediction == Y_train).sum() / float(len(Y_test))
print(accuracy)