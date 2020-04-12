import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z) :
    
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim) :

    w = np.zeros((dim, 1))
    b = 0

    return w, b

def propagate(w, b, X, Y) :

    # No. of training examples
    m = X.shape[1]

    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)     # Activation function
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))   # Cost function

    # Backward propagation
    dw = (1 / m) * np.dot(X, (A - Y).T)    # dJ/dw
    db = (1 / m) * np.sum(A - Y)    # dJ/db

    cost = np.squeeze(cost)

    grads = {'dw' : dw, 'db' : db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False) :

    costs = []

    for i in range(num_iterations) :

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        # Gradient descent
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0 :
            costs.append(cost)

        if print_cost and i % 100 == 0 :
            print(f"Cost after iteration {i} : {cost}")

    params = {'w' : w, 'b' : b}
    grads = {'dw' : dw, 'db' : db}

    return params, grads, costs

def predict(w, b, X) :

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape((X.shape[0], 1))

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]) :

        # Binary classification
        Y_prediction[0, i] = 0 if A[0, i] <= 0.5 else 1

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False) :

    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print(f'\nTraining accuracy : {100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100}')
    print(f'Testing accuracy : {100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100}')

    d = {'costs' : costs, 
         'Y_prediction_test' : Y_prediction_test,
         'T_prediction_train' : Y_prediction_train,
         'w' : w,
         'b' : b,
         'grads' : grads,
         'learning_rate' : learning_rate,
         'num_iterations' : num_iterations}

    return d
    
def binary_sex(sex) :
    return 1 if sex == 'male' else 0

def ternary_embarked(s) :
    if s == 'S' : 
        return 0
    if s == 'C' : 
        return 1
    if s == 'Q' : 
        return 2

    return -1

def preprocess_data(file_obj) :

    df = file_obj.copy()

    df.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace = True)

    df['Age'].fillna(value = df['Age'].median(), inplace = True)
    df['Embarked'].fillna(value = 'N', inplace = True)

    df['Sex'] = df['Sex'].apply(binary_sex)
    df['Embarked'] = df['Embarked'].apply(ternary_embarked)

    dataset = df.to_numpy()

    return dataset

def load_dataset() :

    file_obj = pd.read_csv('titanic_x_y_train.csv')

    training_data = preprocess_data(file_obj)

    X = training_data[ : , : -1]
    Y = training_data[ : , -1 ]

    # Normalizing data
    X = StandardScaler().fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    return X_train.T, Y_train.reshape((1, Y_train.shape[0])), X_test.T, Y_test.reshape((1, Y_test.shape[0]))

# Main
X_train, Y_train, X_test, Y_test = load_dataset()

learning_rates = [0.0001, 0.001, 0.01, 0.1]
models = {}

for lr in learning_rates :

    print(f'learning rate is : {lr}')

    models[str(lr)] = model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = lr, print_cost = False)

    print('\n' + "-------------------------------------------------------" + '\n')

# Plot learning curve (with costs)
for lr in learning_rates :
    
    d = models[str(lr)]

    costs = np.squeeze(d['costs'])
    plt.plot(costs, label = str(d['learning_rate']))

plt.ylabel('cost')
plt.xlabel('iterations (per hundred)')

legend = plt.legend(loc='upper right', shadow = True)

plt.show()