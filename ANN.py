import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
class MLP:
    loss_history = []
    act_fun = []
    def __init__(self, layers, act_fun, learning_rate=0.01):             #initialization the MLP
        self.act_fun = act_fun                                           #output laye also need activation function
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.init_weights()
    
    def init_weights(self):
        np.random.seed(42)
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2 / self.layers[i])        # He initialization#
            bias = np.zeros((1, self.layers[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)                                                                           # to avoid overflow
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)       
    
 
    

    
    def forward(self, X):                                                                               #forward pass                               
        activations = [X]
        for i in range(len(self.weights)):
            net_input = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            act_fun = self.act_fun[i]
            if act_fun == 'sigmoid':
                activation = self.sigmoid(net_input)
            elif act_fun == 'relu':
                activation = self.relu(net_input)    
            activations.append(activation)
        return activations


    
    def backward(self, activations, y):         #backward pass
        errors = [y - activations[-1]]                                                              
        act_fun = self.act_fun[-1]
        if act_fun == 'sigmoid':
            deltas = [errors[-1] * self.sigmoid_derivative(activations[-1])]
        elif act_fun == 'relu':
            deltas = [errors[-1] * self.relu_derivative(activations[-1])]                           #output layer gradient
        
        
            
            
        for i in range(len(self.weights) - 1, 0, -1):                                               
            act_fun = self.act_fun[i-1]
            error = np.dot(deltas[-1], self.weights[i].T)
            if act_fun == 'sigmoid':
                delta = error * self.sigmoid_derivative(activations[i])
            elif act_fun == 'relu':
                delta = error * self.relu_derivative(activations[i])
            errors.append(error)
            deltas.append(delta)
        
        deltas.reverse()
        
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] += self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)
    
    def train(self, X, y,  epochs=800, batch_size=16,):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n_samples)                                      #shuffle the data
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                activations = self.forward(X_batch)
                self.backward(activations, y_batch)
            pred = self.forward(X)[-1]
            loss = np.mean(np.square(y - pred))
            self.loss_history.append(loss)
            if epoch % 20 == 0:
                
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict(self, X):
        return self.forward(X)[-1]



#### main code ####



# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images and normalize
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0


y_train_encoded = np.zeros((y_train.size, 10))
y_train_encoded[np.arange(y_train.size), y_train] = 1

y_test_encoded = np.zeros((y_test.size, 10))
y_test_encoded[np.arange(y_test.size), y_test] = 1

# Use a small subset for demonstration    to speed up when testing the code
x_train_small = x_train[:60000]
y_train_small = y_train_encoded[:60000]
x_test_small = x_test[:10000]
y_test_small = y_test_encoded[:10000]

# Create and train MLP
mlp = MLP(layers=[784, 100, 10], act_fun=['relu','relu','relu'],learning_rate=0.001)
mlp.train(x_train_small, y_train_small, epochs=400, batch_size=32)

# Evaluate on test set
predictions = mlp.predict(x_test_small)
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test_small, axis=1))
print(f'Test Accuracy: {accuracy * 100:.2f}%')
plt.plot(mlp.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Convergence Curve')
plt.grid(True)
plt.show()

num_samples = 10
images = x_test_small[:num_samples].reshape(-1, 28, 28)
labels_true = np.argmax(y_test_small[:num_samples], axis=1)
labels_pred = np.argmax(predictions[:num_samples], axis=1)

plt.figure(figsize=(12, 4))
for i in range(num_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f'Pred: {labels_pred[i]}\nTrue: {labels_true[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()