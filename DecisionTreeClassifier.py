from sklearn import tree
x = [[180], [183], [179], [165], [170], [160], [182], [155], [177], [178], [176]]
y = ['male', 'male', 'female', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'female']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
prediction = clf.predict([[168]])
print (prediction)