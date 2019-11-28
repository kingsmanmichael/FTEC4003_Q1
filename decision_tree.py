import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

#####################################
# Read the training data from & convert all string values into integer values.

train = pd.read_csv("bank-train.csv", header=None, sep=';')
train_headers = train.iloc[0]
train_df = pd.DataFrame(train.values[1:], columns=train_headers)

train_df['job'] = train_df['job'].map(
    {'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5,
     'retired': 6, 'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10,
     'unemployed': 11, 'unknown': 12})
train_df['marital'] = train_df['marital'].map({'divorced': 1, 'married': 2, 'single': 3, 'unknown': 4})
train_df['education'] = train_df['education'].map({'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4,
                                                   'illiterate': 5, 'professional.course': 6, 'university.degree': 7,
                                                   'unknown': 8})
train_df['default'] = train_df['default'].map({'no': 1, 'unknown': 2, 'yes': 3})
train_df['housing'] = train_df['housing'].map({'no': 1, 'unknown': 2, 'yes': 3})
train_df['loan'] = train_df['loan'].map({'no': 1, 'unknown': 2, 'yes': 3})
train_df['contact'] = train_df['contact'].map({'cellular': 1, 'telephone': 2})
train_df['month'] = train_df['month'].map(
    {'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
     'nov': 11, 'dec': 12})
train_df['day_of_week'] = train_df['day_of_week'].map({'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5})
train_df['poutcome'] = train_df['poutcome'].map({'failure': 1, 'nonexistent': 2, 'success': 3})

feature_cols = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
                'cons.conf.idx', 'euribor3m', 'nr.employed']
#####################################
# Use the training data to build a decision tree model by gini.

X = train_df.iloc[:, :19]
y = train_df.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=1)

# Create Decision Tree classifer object
dtc = DecisionTreeClassifier(criterion="gini")
# Train Decision Tree Classifer
dtc = dtc.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = dtc.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# train_result = pd.DataFrame(y_pred.tolist(), columns=['Label'])
# train_result.index.name = 'ID'
#####################################
# Use the model to predict the results of the test data.

test = pd.read_csv("bank-test.csv", header=None, sep=';')
test_headers = test.iloc[0]
test_df = pd.DataFrame(test.values[1:], columns=test_headers)

test_df['job'] = test_df['job'].map({'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5,
                                   'retired': 6, 'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10,
                                   'unemployed': 11, 'unknown': 12})
test_df['marital'] = test_df['marital'].map({'divorced': 1, 'married': 2, 'single': 3, 'unknown': 4})
test_df['education'] = test_df['education'].map({'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4,
                                               'illiterate': 5, 'professional.course': 6, 'university.degree': 7,
                                               'unknown': 8})
test_df['default'] = test_df['default'].map({'no': 1, 'unknown': 2, 'yes': 3})
test_df['housing'] = test_df['housing'].map({'no': 1, 'unknown': 2, 'yes': 3})
test_df['loan'] = test_df['loan'].map({'no': 1, 'unknown': 2, 'yes': 3})
test_df['contact'] = test_df['contact'].map({'cellular': 1, 'telephone': 2})
test_df['month'] = test_df['month'].map({'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10,
                                       'nov': 11, 'dec': 12})
test_df['day_of_week'] = test_df['day_of_week'].map({'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5})
test_df['poutcome'] = test_df['poutcome'].map({'failure': 1, 'nonexistent': 2, 'success': 3})

test_pred = dtc.predict(test_df)
test_result = pd.DataFrame(test_pred.tolist(), columns=['Label'])
test_result.index.name = 'ID'

# Output as a csv file.
test_result.to_csv(r'decision_tree_result.csv')
