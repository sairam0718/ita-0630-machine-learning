import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("D:\ml\ENJOYSPORT.csv")

X = df.drop('EnjoySport', axis=1)
y = df['EnjoySport']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X_train, y_train)

tree_rules = export_text(clf, feature_names=X.columns.tolist())
print("Decision Tree Rules:")
print(tree_rules)

accuracy = clf.score(X_test, y_test)
print(f"\nAccuracy on test set: {accuracy:.2f}")

new_sample = pd.DataFrame([[0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]], columns=X.columns)

predicted_class = clf.predict(new_sample)
print(f"\nPredicted class for new sample: {predicted_class[0]}")
