import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward(inputs, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    # Calculate the output of the hidden layer
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Calculate the output of the output layer
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    return output_layer_output

def backward(inputs, targets, output, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, learning_rate):
    # Calculate the error
    error = targets - output

    # Calculate the gradient at the output layer
    delta_output = error * sigmoid_derivative(output)

    # Calculate the error in the hidden layer
    error_hidden = delta_output.dot(weights_hidden_output.T)

    # Calculate the gradient at the hidden layer
    delta_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases using gradients and learning rate
    weights_hidden_output += hidden_layer_output.T.dot(delta_output) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += inputs.reshape(-1, 1).dot(delta_hidden) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

def train(inputs, targets, hidden_size, learning_rate, epochs):
    input_size = inputs.shape[1]
    output_size = targets.shape[1]

    # Initialize weights and biases with random values
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    bias_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_output = np.zeros((1, output_size))

    for epoch in range(epochs):
        for i in range(len(inputs)):
            # Forward pass
            output = forward(inputs[i], weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)

            # Backpropagation
            backward(inputs[i], targets[i], output, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, learning_rate)

        if (epoch + 1) % 1000 == 0:
            error = np.mean(np.square(targets - forward(inputs, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)))
            print(f"Epoch {epoch + 1}/{epochs}, Error: {error}")

if _name_ == "_main_":
    # Define input, target data
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Set hyperparameters
    hidden_size = 2
    learning_rate = 0.1
    epochs = 10000

    # Train the MLP
    train(inputs, targets, hidden_size, learning_rate, epochs)

    # Test the trained MLP
    predictions = forward(inputs, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
    print("Predictions:")
    print(predictions)


