import numpy as np
import tensorflow as tf
import pandas as pd
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked


def start():
    train_data, test_data = load_data()
    train_output = train_data["Survived"].values.reshape([891, 1])
    m = train_output.shape[0]
    minAge = train_data["Age"].min()
    maxAge = train_data["Age"].max()
    meanAge = train_data["Age"].mean()
    input_data = preprocess_input(train_data, minAge, maxAge, meanAge, m)
    parameters = model(input_data.transpose(), train_output.transpose())
    X_test = preprocess_input(test_data, minAge, maxAge, meanAge, test_data.shape[0])
    predictions = predict(X_test.transpose(), parameters).transpose()
    int_prediction = []
    for i in range(predictions.shape[0]):
        if predictions[i] > 0.5:
            int_prediction.append(1)
        else:
            int_prediction.append(0)
    csv_output = test_data["PassengerId"].to_frame()
    csv_output["Survived"] = int_prediction
    csv_output.to_csv('result.csv', index=False)


def load_data():
    data_train = pd.read_csv("train.csv")
    data_test = pd.read_csv("test.csv")
    print(data_test.shape)
    return data_train, data_test


def preprocess_input(input_data, minAge, maxAge, meanAge, m):
    cleaned_pclass = preprocess_pclass(input_data["Pclass"]).values
    cleaned_sex = pd.get_dummies(input_data["Sex"]).values
    cleaned_age = preprocess_age(input_data["Age"], minAge, maxAge, meanAge).values
    cleaned_age  = cleaned_age.reshape([m, 1])
    return np.concatenate((cleaned_pclass, cleaned_sex, cleaned_age),axis=1)


def preprocess_pclass(pclass):
    pc = pd.get_dummies(pclass)
    return pc


def preprocess_age(age, minAge, maxAge, meanAge):
    age = age.fillna(meanAge)
    cleaned_age = (age-minAge)/(maxAge - minAge)
    return cleaned_age

def initialize_parameters(n_X):
    tf.set_random_seed(1)  # so that your "random" numbers match ours
    W1 = tf.get_variable("W1", [10, n_X], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [10, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [11, 10], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [11, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 11], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [1, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
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


def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
    return X, Y


def model(X_train, Y_train, learning_rate=0.01,
          num_epochs=15000, print_cost=True):
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(n_x)
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            seed = seed + 1
            _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.round(Z3), Y)

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        # print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [X.shape[0], X.shape[1]])
    z3 = forward_propagation(x, params)
    p = tf.round(z3)
    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})
    return prediction


start()
