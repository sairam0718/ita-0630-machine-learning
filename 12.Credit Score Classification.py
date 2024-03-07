import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    data = pd.read_csv('CREDITSCORE.csv')
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit()

X = data[['Monthly_Inhand_Salary', 'Age', 'Num_Credit_Card', 'Annual_Income']]
y = data['Credit_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

data['Predicted_Credit_Score'] = clf.predict(X)

fig1 = px.box(data, 
             x="Predicted_Credit_Score", 
             y="Monthly_Inhand_Salary", 
             color="Predicted_Credit_Score",
             title="Monthly Inhand Salary for Predicted Credit Score Classification", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig1.update_traces(quartilemethod="exclusive")
fig1.show()

fig2 = px.box(data, 
             x="Predicted_Credit_Score", 
             y="Age", 
             color="Predicted_Credit_Score",
             title="Age for Predicted Credit Score Classification", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig2.update_traces(quartilemethod="exclusive")
fig2.show()
