import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold


def start():
    dev_data, test_data = load_data()
    numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    numerical_transformer = SimpleImputer(strategy='median')
    categorical_cols = ['Sex', 'Embarked']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))]
    )
    name_col = ['Name']
    name_transformer = Pipeline(steps=[
        ('clean_name', FunctionTransformer(clean_name, validate=False)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))]
    )
    ticket_col = ['Ticket']
    ticket_transformer = Pipeline(steps=[
        ('clean_ticket', FunctionTransformer(clean_ticket, validate=False)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))]
    )
    selected_cols = numerical_cols + categorical_cols + name_col + ticket_col
    X_train = dev_data[selected_cols].copy()
    y_train = dev_data.Survived
    X_test = test_data[selected_cols].copy()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('name', name_transformer, name_col),
            ('ticket', ticket_transformer, ticket_col)
        ])

    model = KerasClassifier(build_fn=nn_model, epochs=300, batch_size=50)
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)
                     ])
    clf.fit(X_train, y_train)
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # results = cross_val_score(clf, X_train, y_train, cv=kfold)
    # print(results.mean())
    preds = clf.predict(X_test)
    csv_output = test_data['PassengerId'].to_frame()
    int_prediction = []
    for pred in preds:
        if pred > 0.5:
            int_prediction.append(1)
        else:
            int_prediction.append(0)
    csv_output['Survived'] = int_prediction
    csv_output.to_csv('result.csv', index=False)
    submission = pd.DataFrame()
    submission['PassengerId'] = test_data.index
    submission['Survived'] = preds


def load_data():
    dev_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    return dev_data, test_data


def clean_name(names):
    title_in_names = []
    titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.']
    for name in names.values:
        cleaned_title = 'Other'
        for title in titles:
            if title in name[0]:
                cleaned_title = title
                break
        title_in_names.append(cleaned_title)
    title_in_names = np.reshape(title_in_names, (-1, 1))
    return pd.DataFrame(title_in_names, columns=['Name'])


def clean_ticket(tickets):
    ticket_abbr_col = []
    ticket_abbrs = ['1', '2', '3', 'A', 'C', 'P', 'S']
    for ticket in tickets.values:
        cleaned_ticket = 'Other'
        for ticket_abbr in ticket_abbrs:
            if ticket_abbr in ticket[0]:
                cleaned_ticket = ticket_abbr
                break
        ticket_abbr_col.append(cleaned_ticket)
    ticket_abbr_col = np.reshape(ticket_abbr_col, (-1, 1))
    return pd.DataFrame(ticket_abbr_col, columns=['Ticket'])


def nn_model():
    model = Sequential()
    model.add(Dense(20, activation='relu', input_dim=22))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=["accuracy"])
    return model

start()
