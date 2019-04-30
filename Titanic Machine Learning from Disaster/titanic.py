import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.preprocessing import OneHotEncoder
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# TBD:
# 1. Add more features
# 2. Add more layers/neurons
# 3. Add regularization


def start():
    dev_data, x_test = load_data()
    #x_train, x_validation, y_train, y_validation = split_train_validation(dev_data)
    y_train = dev_data['Survived']
    x_train = dev_data.drop(['Survived'], axis=1)
    scaling_info = extract_scaling_info(dev_data)
    encoders = generate_one_hot_encoders()
    x_train = clean_data(x_train, scaling_info, encoders)
    y_train = y_train.values.reshape(1, -1)
    # x_validation = clean_data(x_validation, scaling_info, encoders)
    # y_validation = y_validation.values.reshape(1, -1)
    #parameters = model(x_train, y_train, x_validation, y_validation)

    # learning_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    # regularizations = [0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    # for learning_rate in learning_rates:
    #     for regularization in regularizations:
    #         tf.reset_default_graph()
    #         model(x_train, y_train, x_validation, y_validation, learning_rate=learning_rate, regularization=regularization)

    tf.reset_default_graph()
    parameters = model(x_train, y_train, learning_rate=0.003, regularization=0.0003)
    csv_output = x_test['PassengerId'].to_frame()
    x_test = clean_data(x_test, scaling_info, encoders)
    predictions = predict(x_test, parameters).transpose()
    int_prediction = []
    for i in range(predictions.shape[0]):
        if predictions[i] > 0.5:
            int_prediction.append(1)
        else:
            int_prediction.append(0)
    csv_output['Survived'] = int_prediction
    csv_output.to_csv('result.csv', index=False)


def load_data():
    dev_data = pd.read_csv('train.csv')
    x_test = pd.read_csv('test.csv')
    return dev_data, x_test


def clean_data(dev_data, scaling_info, encoders):
    cleaned_pclass = encoders['enc_pclass'].transform(dev_data['Pclass'].fillna(4).values.reshape(-1, 1)).toarray()
    cleaned_name = encoders['enc_title'].transform(clean_name(dev_data['Name']).reshape(-1, 1)).toarray()
    cleaned_sex = encoders['enc_gender'].transform(dev_data['Sex'].fillna('Other').values.reshape(-1, 1)).toarray()
    cleaned_age = scaling_to_one(dev_data['Age'], scaling_info['age']).values.reshape(-1, 1)
    cleaned_sib_sp = scaling_to_one(dev_data['SibSp'], scaling_info['sib_sp']).values.reshape(-1, 1)
    cleaned_parch = scaling_to_one(dev_data['Parch'], scaling_info['parch']).values.reshape(-1, 1)
    cleaned_fare = scaling_to_one(dev_data['Fare'], scaling_info['fare']).values.reshape(-1, 1)
    cleaned_cabin = encoders['enc_cabin'].transform(clean_cabin(dev_data['Cabin'].fillna('Other')).
                                                    reshape(-1, 1)).toarray()
    cleaned_embarked = encoders['enc_embarked'].transform(dev_data['Embarked'].fillna('Other').values.
                                                          reshape(-1, 1)).toarray()
    return np.concatenate((cleaned_pclass, cleaned_name, cleaned_sex, cleaned_age, cleaned_sib_sp, cleaned_parch,
                           cleaned_fare, cleaned_cabin, cleaned_embarked), axis=1).transpose()


def scaling_to_one(input, scaling_info):
    return (input.fillna(scaling_info['mean']) - scaling_info['min'])/(scaling_info['max'] - scaling_info['min'])


def split_train_validation(dev_data):
    y_dev = dev_data['Survived']
    x_dev = dev_data.drop(['Survived'], axis=1)
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x_dev, y_dev, train_size=0.9,
                                                                                    test_size=0.1, random_state=1)
    return x_train, x_validation, y_train, y_validation


def extract_scaling_info(dev_data):
    age = {'min': dev_data['Age'].min(), 'mean': dev_data['Age'].mean(), 'max': dev_data['Age'].max()}
    fare = {'min': dev_data['Fare'].min(), 'mean': dev_data['Fare'].mean(), 'max': dev_data['Fare'].max()}
    sib_sp = {'min': dev_data['SibSp'].min(), 'mean': dev_data['SibSp'].mean(), 'max': dev_data['SibSp'].max()}
    parch = {'min': dev_data['Parch'].min(), 'mean': dev_data['Parch'].mean(), 'max': dev_data['Parch'].max()}
    scaling_info = {'age': age, 'fare': fare, 'sib_sp': sib_sp, 'parch': parch}
    return scaling_info


def generate_one_hot_encoders():
    # pclass 4 and other in other encoders are for missing entries
    enc_pclass = OneHotEncoder(handle_unknown='ignore').fit([[1], [2], [3], [4]])
    enc_gender = OneHotEncoder(handle_unknown='ignore').fit([['female'], ['male'], ['Other']])
    enc_title = OneHotEncoder(handle_unknown='ignore').fit([['Mr.'], ['Mrs.'], ['Miss.'], ['Master.'], ['Dr.'],
                                                            ['Rev.'], ['Other']])
    enc_cabin = OneHotEncoder(handle_unknown='ignore').fit([['A'], ['B'], ['C'], ['D'], ['E'], ['F'], ['G'], ['Other']])
    enc_embarked = OneHotEncoder(handle_unknown='ignore').fit([['C'], ['S'], ['Q'], ['Other']])
    return {'enc_pclass': enc_pclass, 'enc_gender': enc_gender, 'enc_title': enc_title, 'enc_cabin': enc_cabin,
            'enc_embarked': enc_embarked}


def clean_name(names):
    title_in_names = []
    titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.', 'Rev.']
    for name in names:
        contains_title = False
        for title in titles:
            if title in name:
                title_in_names.append(title)
                contains_title = True
                break
        if not contains_title:
            title_in_names.append('Other')
    return np.reshape(title_in_names, (-1, 1))


def clean_cabin(cabins):
    cabins_pre = [];
    cabin_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for cabin_full in cabins:
        contains_cabin_type = False
        for cabin_type in cabin_types:
            if cabin_type in cabin_full:
                cabins_pre.append(cabin_type)
                contains_cabin_type = True
                break
        if not contains_cabin_type:
            cabins_pre.append('Other')
    return np.reshape(cabins_pre, (-1, 1))


def initialize_parameters(n_X):
    tf.set_random_seed(1)
    W1 = tf.get_variable('W1', [25, n_X], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [5, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [5, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [1, 5], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [1, 1], initializer=tf.zeros_initializer())
    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2,
                  'W3': W3,
                  'b3': b3}
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3
    Z3 = tf.sigmoid(Z3)
    return Z3


def compute_cost(parameters, Z3, Y, regularizer_coefficient):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    regularizer = tf.nn.l2_loss(parameters['W1'])+tf.nn.l2_loss(parameters['W2'])+tf.nn.l2_loss(parameters['W3'])
    cost = tf.reduce_mean(cost +  regularizer_coefficient * regularizer)
    return cost


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
    return X, Y


def model(X_train, Y_train, learning_rate=0.001, regularization=0,
          num_epochs=20000):
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(n_x)
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(parameters, Z3, Y, regularizer_coefficient=regularization)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            seed = seed + 1
            _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.round(Z3), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print('Learning Rate: ', learning_rate)
        print('Regularization: ', regularization)
        print('Train Accuracy:', accuracy.eval({X: X_train, Y: Y_train}))
        # print('Test Accuracy:', accuracy.eval({X: X_test, Y: Y_test}))
        print()
        return parameters


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters['W1'])
    b1 = tf.convert_to_tensor(parameters['b1'])
    W2 = tf.convert_to_tensor(parameters['W2'])
    b2 = tf.convert_to_tensor(parameters['b2'])
    W3 = tf.convert_to_tensor(parameters['W3'])
    b3 = tf.convert_to_tensor(parameters['b3'])

    params = {'W1': W1,
              'b1': b1,
              'W2': W2,
              'b2': b2,
              'W3': W3,
              'b3': b3}

    x = tf.placeholder('float', [X.shape[0], X.shape[1]])
    z3 = forward_propagation(x, params)
    p = tf.round(z3)
    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})
    return prediction


start()
