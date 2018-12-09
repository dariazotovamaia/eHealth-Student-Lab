svc = SVC(kernel='poly')

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
accuracy = (prediction == Y_test).sum() / float(len(Y_test))
print(accuracy)
