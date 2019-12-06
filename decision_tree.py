import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

# 1. Read the training data and format it from csv to python dataframe.
train = pd.read_csv("bank-train.csv", header=None, sep=';')
train_headers = train.iloc[0]
train_df = pd.DataFrame(train.values[1:], columns=train_headers)

# 2. Assign the numerical values to the string values.
le = preprocessing.LabelEncoder()
le.fit(['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
        'student', 'technician', 'unemployed', 'unknown'])
train_df['job'] = le.transform(train_df['job'])

le = preprocessing.LabelEncoder()
le.fit(['divorced', 'married', 'single', 'housemaid', 'unknown', 'retired'])
train_df['marital'] = le.transform(train_df['marital'])

le = preprocessing.LabelEncoder()
le.fit(['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree',
        'unknown'])
train_df['education'] = le.transform(train_df['education'])

le = preprocessing.LabelEncoder()
le.fit(['no', 'unknown', 'yes'])
train_df['default'] = le.transform(train_df['default'])

le = preprocessing.LabelEncoder()
le.fit(['no', 'unknown', 'yes'])
train_df['housing'] = le.transform(train_df['housing'])

le = preprocessing.LabelEncoder()
le.fit(['no', 'unknown', 'yes'])
train_df['loan'] = le.transform(train_df['loan'])

le = preprocessing.LabelEncoder()
le.fit(['cellular', 'telephone'])
train_df['contact'] = le.transform(train_df['contact'])

le = preprocessing.LabelEncoder()
le.fit(['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
train_df['month'] = le.transform(train_df['month'])

le = preprocessing.LabelEncoder()
le.fit(['mon', 'tue', 'wed', 'thu', 'fri'])
train_df['day_of_week'] = le.transform(train_df['day_of_week'])

le = preprocessing.LabelEncoder()
le.fit(['failure', 'nonexistent', 'success'])
train_df['poutcome'] = le.transform(train_df['poutcome'])

for col in train_df.columns:
    if col == 'y':
        continue
    train_df[col] = pd.to_numeric(train_df[col])

# 3. Select useful attributes.
# feature_cols = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
#                 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
#                 'cons.conf.idx', 'euribor3m', 'nr.employed']
feature_cols = ['age', 'job', 'default', 'campaign', 'pdays',
                'emp.var.rate', 'cons.conf.idx', 'nr.employed']
X = train_df[feature_cols]
y = train_df.y

# 4. Train the data using decision tree with gini.
dtc = DecisionTreeClassifier(criterion = "gini")
dtc = dtc.fit(X,y)

# 5. Read the testing data and format it from csv to python dataframe.
test = pd.read_csv("bank-test.csv", header=None, sep=';')
test_headers = test.iloc[0]
test_df = pd.DataFrame(test.values[1:], columns=test_headers)

# 6. Assign the numerical values to the string values.
le = preprocessing.LabelEncoder()
le.fit(['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
        'student', 'technician', 'unemployed', 'unknown'])
test_df['job'] = le.transform(test_df['job'])

le = preprocessing.LabelEncoder()
le.fit(['divorced', 'married', 'single', 'housemaid', 'unknown', 'retired'])
test_df['marital'] = le.transform(test_df['marital'])

le = preprocessing.LabelEncoder()
le.fit(['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree',
        'unknown'])
test_df['education'] = le.transform(test_df['education'])

le = preprocessing.LabelEncoder()
le.fit(['no', 'unknown', 'yes'])
test_df['default'] = le.transform(test_df['default'])

le = preprocessing.LabelEncoder()
le.fit(['no', 'unknown', 'yes'])
test_df['housing'] = le.transform(test_df['housing'])

le = preprocessing.LabelEncoder()
le.fit(['no', 'unknown', 'yes'])
test_df['loan'] = le.transform(test_df['loan'])

le = preprocessing.LabelEncoder()
le.fit(['cellular', 'telephone'])
test_df['contact'] = le.transform(test_df['contact'])

le = preprocessing.LabelEncoder()
le.fit(['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
test_df['month'] = le.transform(test_df['month'])

le = preprocessing.LabelEncoder()
le.fit(['mon', 'tue', 'wed', 'thu', 'fri'])
test_df['day_of_week'] = le.transform(test_df['day_of_week'])

le = preprocessing.LabelEncoder()
le.fit(['failure', 'nonexistent', 'success'])
test_df['poutcome'] = le.transform(test_df['poutcome'])

# 7. Fit the testing data into the decision tree model.
test_pred = dtc.predict(test_df[feature_cols])
test_result = pd.DataFrame(test_pred.tolist(), columns=['Label'])
test_result.index.name = 'ID'

# 8. Output the csv file for evaluation of the f-measure.
test_result.to_csv(r'submission1_decision_tree.csv')
# f-meaure = 0.31901
