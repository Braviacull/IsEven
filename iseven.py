import numpy as np

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def relu(z) :
    return np.maximum(0,z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def create_even_data_set(m) :
    a0 = np.arange(1, m + 1).reshape(1, m).astype(float)
    #a0 /= np.max(a0)

    y = (a0 % 2 == 0).astype(int)

    return a0,y

def forward_propagation(a0, w1, b1, w2, b2) :
    z1 = np.add(np.dot(w1, a0), b1)  # (4,m)
    a1 = relu(z1)  # (4,m)

    # calcolo z2
    z2 = np.add(np.dot(w2, a1), b2)  # (1,m)
    a2 = sigmoid(z2)  # (1,m)

    return z1, a1, z2, a2

def normalization(x) :
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    x_normalized = (x - mean) / std
    return x_normalized

def backward_propagation(a0, a1, a2, y, w2, z1) :
    dz2 = a2 - y
    dw2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2.T, dz2) * relu_derivative(z1)
    dw1 = np.dot(dz1, a0.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2


def main() :
    #np.random.seed(0)

    learning_rate = 0.01
    # numero caratteristiche e numero esempi
    nx = 1
    m = 1000

    # vettore input e vettore soluzioni
    a0, y = create_even_data_set(m) # (1,m)

    a0_normalized = normalization(a0)

    # numero unità hidden layer, weights e bias
    neurons1 = 4
    w1 = np.random.randn(neurons1, nx) # (4,1)
    b1 = np.random.randn(neurons1, 1) # (4,1)

    # numero unità output layer, weights e bias
    neurons2 = 1
    w2 = np.random.randn(neurons2, neurons1) # (1,4)
    b2 = np.random.randn(neurons2, 1) # (1,1)

    # ALLENAMENTO
    for epoch in range(1000): # numero epoche

        # FORWARD PROPAGATION
        z1, a1, z2, a2 = forward_propagation(a0_normalized, w1, b1, w2, b2)

        # BACKWARD PROPAGATION
        dw1, db1, dw2, db2 = backward_propagation(a0_normalized, a1, a2, y, w2, z1)

        # aggiornamento pesi
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

    # test
    test = np.arange(1, 100 + 1).reshape(1, 100)
    test_normalized = normalization(test)
    z1, a1, z2, a2 = forward_propagation(test_normalized, w1, b1, w2, b2)

    print (a2)

    # approssimo i valori di a2 a 0 o 1
    a2 = np.where(a2 < 0.5, 0, 1)
    print(a2)


main()