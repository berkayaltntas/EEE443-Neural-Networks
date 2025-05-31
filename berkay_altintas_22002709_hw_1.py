import sys

question = sys.argv[1]

import numpy as np
import matplotlib.pyplot as plt
import h5py

def xavier(Lpre, Lpost):
    # I have first define the Xavier initialization function.
    return np.sqrt(6 / (Lpre + Lpost))

def initialize_weights(Lin, Lhid, Lout):
    # Using Xavier initialization and random uniform, I have defined my learnable parameters.
    # To get similar results for each trial I have used seed.
    np.random.seed(42) 
    W1 = np.random.uniform(-xavier(Lin, Lhid), xavier(Lin, Lhid), (Lin, Lhid))
    W2 = np.random.uniform(-xavier(Lhid, Lout), xavier(Lhid, Lout), (Lhid, Lout))
    b1 = np.random.uniform(-xavier(Lhid, Lout), xavier(Lhid, Lout), (Lhid,))
    b2 = np.random.uniform(-xavier(Lhid, Lout), xavier(Lhid, Lout), (Lout, ))
    return W1, W2, b1, b2

def sigmoid_q1(x):
    # To get better convergence, I have used clip trick.
    x = np.clip(x, -500, 500)  
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative_q1(x):
    # Since I used it regularly, I have defined a function for derivative of sigmoid
    return x * (1 - x)

def forward_pass(W1, W2, b1, b2, data):
    # Forward pass along the autoencoder. First encoder then decoder.
    hidden = sigmoid_q1(np.dot(data, W1) + b1)
    output = sigmoid_q1(np.dot(hidden, W2) + b2)
    return hidden, output

def compute_cost_and_gradients(W1, W2, b1, b2, data, lambdaa, beta, rho):
    # I have do the calcuations for cost and gradients. I have also define stabilizer to get more stable results.
    N = data.shape[0]
    stabilizer = 1e-10
    hidden, output = forward_pass(W1, W2, b1, b2, data)

    # Lets first do the KL divergence term.
    rho_hat = np.mean(hidden, axis=0)  
    kl_divergence = beta * np.sum(rho * np.log(rho / (rho_hat + stabilizer)) + (1 - rho) * np.log((1 - rho) / (1 - rho_hat + stabilizer)))

    # Then define the reconstruction loss.
    reconstruction_loss = (1 / (2 * N)) * np.sum((output - data) ** 2)

    # Then define the regularization term
    regularization = (lambdaa / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    # Now we can get the full expression.
    cost = reconstruction_loss + regularization + kl_divergence

    # Now, lets do the backprop
    delta_out = -(data - output) * sigmoid_derivative_q1(output)
    sparsity_delta = beta * (-rho / (rho_hat + stabilizer) + (1 - rho) / (1 - rho_hat + stabilizer))
    delta_hidden = (np.dot(delta_out, W2.T) + sparsity_delta) * sigmoid_derivative_q1(hidden)

    # Our gradients like as follows,
    gradW2 = (1 / N) * np.dot(hidden.T, delta_out)  + lambdaa * W2
    gradb2 =  (1 / N) *  np.sum(delta_out, axis=0) 
    gradW1 =  (1 / N) * np.dot(data.T, delta_hidden)  + lambdaa * W1
    gradb1 =  (1 / N) * np.sum(delta_hidden, axis=0) 

    return cost, gradW1, gradW2, gradb1, gradb2

def update_weights(W1, W2, b1, b2, data, lambdaa, beta, rho, learning_rate):
    # Here, I have updated our learnable parameters.
    cost, gradW1, gradW2, gradb1, gradb2 = compute_cost_and_gradients(W1, W2, b1, b2, data, lambdaa, beta, rho)
    W1 = W1 - learning_rate * gradW1
    W2 = W2 - learning_rate * gradW2
    b1 = b1 - learning_rate * gradb1
    b2 = b2 - learning_rate * gradb2
    return W1, W2, b1, b2, cost

def train_autoencoder(data, Lin, Lhid, lambdaa, beta, rho, learning_rate, epochs, batch_size):
    # I have trained the autoencoder using mini batch gradient descent. I have first tried the full batch gradient descent. It didnt work like expected
    # Then I used mini batches.
    W1, W2, b1, b2 = initialize_weights(Lin, Lhid, Lin)
    num_samples = data.shape[0]
    cost_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        data_shuffled = data[indices]
        total_cost = 0
        
        for i in range(0, num_samples, batch_size):
            batch_data = data_shuffled[i:i + batch_size]
            W1, W2, b1, b2, batch_cost = update_weights(W1, W2, b1, b2, batch_data, lambdaa, beta, rho, learning_rate)
            total_cost += batch_cost * batch_data.shape[0] / num_samples
        
        cost_history.append(total_cost)
        print(f"Epoch {epoch + 1}, Cost: {total_cost:.6f}")
    
    plt.plot(cost_history)
    plt.title("Training Cost Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.show()

    return W1, W2, b1, b2

def display_learned_features(W_hidden, title):
    # To visualize what my autoencoder learned.
    num_of_learned_features = W_hidden.shape[1]
    num_of_cols = int(np.sqrt(num_of_learned_features)) if int(np.sqrt(num_of_learned_features)) ** 2 == num_of_learned_features else int(np.sqrt(num_of_learned_features)) + 1
    num_of_rows = num_of_learned_features // num_of_cols if num_of_learned_features % num_of_cols == 0 else (num_of_learned_features // num_of_cols) + 1

    plt.figure(figsize=(12, 12))
    plt.suptitle(title, fontsize=16)
    for index, weight in enumerate(W_hidden.T):
        plt.subplot(num_of_rows, num_of_cols, index + 1)
        plt.imshow(weight.reshape(16, 16), cmap='gray')
        plt.axis('off')
    plt.show()


def converting_grayscale(rgb_data):
    # Using the given luminosity model, I have converted RGB data to grayscale data.
    # I have did some manipulations for axes also.
    return 0.2126 * rgb_data[:, 0, :, :] + 0.7152 * rgb_data[:, 1, :, :] + 0.0722 * rgb_data[:, 2, :, :]

def normalize_data(data):
    # I have normalized my data here regarding what is written in the task.
    mean = np.mean(data, axis=(1, 2), keepdims=True)
    centered = data - mean
    std = np.std(centered)
    clipped = np.clip(centered, -3 * std, 3 * std)
    return 0.1 + 0.8 * (clipped - clipped.min()) / (clipped.max() - clipped.min())

def plotting_rgb(data, sample_size):
    shuffled_data = np.random.permutation(data)
    rgb_data = shuffled_data
    grayscale_data = converting_grayscale(rgb_data)
    normalized_data = normalize_data(grayscale_data)

    # I have first plot the 200 RGB data
    fig, axes = plt.subplots(20, 20, figsize=(20, 20))
    for sample in range(sample_size):
        ax = axes[sample //20, sample % 20]
        ax.imshow(np.transpose(shuffled_data[sample], (1, 2, 0)))
        ax.axis('off')

    # After plotting 200 RGB data, I have plotted their normalized versions.
    for sample in range(sample_size, 2 * sample_size):
        ax = axes[sample // 20, sample % 20]
        ax.imshow((normalized_data[sample - 200]), cmap = 'gray')
        ax.axis('off')
        
    plt.show()  

    # To save to my computer
    plt.savefig('image_plot.png', dpi=300)
    
def q1():
        # From here, you can see the Main code which gives the results for this task.
    with h5py.File('data1.h5', 'r') as f:
        rgb_data = np.array(f['data'])
    
    # PART A
    grayscale_data = converting_grayscale(rgb_data)
    norm_data = normalize_data(grayscale_data)
    flattened_data = norm_data.reshape(norm_data.shape[0], -1)
    plotting_rgb(data = rgb_data, sample_size = 200)
    
    # PART B and PART C
    
    Lin = 256  
    lambdaa = 0.0005  
    beta = 0.5  
    rho = 0.05  
    learning_rate = 0.1
    epochs = 100
    batch_size = 32
    Lhid = 64
    lambdaa = 0.0005
    W1, W2, b1, b2 = train_autoencoder(flattened_data, Lin = 256, Lhid = 64, lambdaa = 0.0005, beta = 0.5, rho = 0.05, learning_rate = 0.1 , epochs = 100, batch_size = 32)
    display_learned_features(W1, title=f"Lhid={Lhid}, lambda={lambdaa}")
    
    # Part D:
    Lhid_values = [10, 50, 100]
    lambdaa_values = [0.00001, 0.0005, 0.001]  # Low, medium, high regularization
    Lhid_values = [10, 50, 100]
    for Lhid in Lhid_values:
        for lambdaa in lambdaa_values:
            print(f"Training with Lhid={Lhid}, lambda={lambdaa}")
            W1, W2, b1, b2 = train_autoencoder(flattened_data, Lin, Lhid, lambdaa, beta, rho, learning_rate, epochs, batch_size)
            display_learned_features(W1, title = f"Lhid={Lhid}, lambda ={lambdaa}")
    print("All results are ready")


 
import numpy as np
import matplotlib.pyplot as plt
import h5py


def q2_initialize_w_and_b(D, P, vocab_size = 250):
    # I have initialized the weights using Gaussian distribution with 0 mean 0.01 variance.
    W1 = np.random.normal(0, 0.01, (P, 3 * D))
    b1 = np.random.normal(0, 0.01, (P, 1))
    W2 = np.random.normal(0, 0.01, (vocab_size, P))
    b2 = np.random.normal(0, 0.01, (vocab_size, 1))

    parameters = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    return parameters
    

def q2_word_to_vec(data, D, embedding_matrix):
    # In here, I take the 3 word indices and convert into their word embeddings. 
    # Then I concatenate them to make input vector which is ready for neural network.
    inputt = np.zeros((data[:,].shape[0], 3*D))
    for i in range(data[:, ].shape[0]):
        index = data[i] 
        first_word = embedding_matrix[index[0]-1, :] 
        second_word = embedding_matrix[index[1]-1, :] 
        third_word = embedding_matrix[index[2]-1, :]         
        all_words = np.concatenate([first_word, second_word, third_word])  
        inputt[i, :] = all_words        
    return inputt
    

def q2_output_word_to_vec(data, D, embedding_matrix):
    # The similar process is done which I have done in q2_word_to_vec function.
    x = np.zeros((data.shape[0], D))
    for i in range(data.shape[0]):
        indices = data[i]
        word1 = embedding_matrix[indices - 1, :]  
        x[i, :] = word1
    return x

def q2_sigmoid(z):
    # I have defined my q2_sigmoid function.
    sigm = 1/ (1 + np.exp(-z))
    return sigm


def q2_derivative_sigmoid(z):
    # To use the derivative of q2_sigmoid easily, I have defined 
    sigm_der = q2_sigmoid(z) * (1 - q2_sigmoid(z))
    return sigm_der

def q2_softmax(z):
    # To find the probabilities at the end of the neural network, I have defined q2_softmax function.
    # However, since the regular q2_softmax suffers from converging problems, I have defined different 
    # version of the q2_softmax which is stable q2_softmax. I observed that it gives more stabilized results.
    probability = np.exp(z - np.max(z,axis=0)) / np.sum(np.exp(z - np.max(z ,axis=0)), axis=0, keepdims=True)
    return probability


def q2_one_hot_encoding(data, num_of_samples, vocab_size = 250):
    # Here, I have converted my outputs into one hot encoding representation.
    one_hot = np.zeros((num_of_samples , vocab_size))
    for samples in range(num_of_samples):
        label = data[samples]
        one_hot[samples, label - 1] = 1 
    return one_hot

def q2_forward_pass(data, parameters, output):
    N = data.shape[1]

    # First I have extracted the required weights and biases.
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    
    # Here is the standart forward pass. 
    # Z represents the pre activation form and A represents the after activation form.
    Z1 = np.dot(W1, data) + b1
    A1 = 1/ (1 + np.exp(-Z1))
    Z2 = np.dot(W2, A1) + b2
    A2 =  np.exp(Z2 - np.max(Z2,axis=0)) / np.sum(np.exp(Z2 - np.max(Z2 ,axis=0)), axis=0, keepdims=True)

    # Our cost is
    cost = ( -1 / N) *np.sum(output.T * np.log(A2))

    # I have stored the required variables to use later.
    cache = {"X": data, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "cost": cost}
   
    return cache


def q2_grads(cache, actual_output, parameters):
    
    # Lets extract the required variables.
    X = cache['X']
    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']
    cost = cache['cost']

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    

    N = X.shape[1]

    # I have done the backprop here.
    dZ2 = (A2  - (actual_output).T)
    dW2 = (1 / N ) * np.dot(dZ2, A1.T)
    db2 = np.mean(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * q2_derivative_sigmoid(Z1)
    dW1 = (1 / N) * np.dot(dZ1, X.T)
    db1 = np.mean(dZ1, axis=1, keepdims=True)
    dX = (1 / 3 * N) * np.dot(W1.T, dZ1)

    # I have stored the gradients to use to update later.
    grad = {"dX": dX, "dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}
    return grad


def q2_defining_velocity(parameters, D, batch_size):

    
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # Lets define our velocities to use them in momemtum later.
    v = {}
    v['W1_vel'] = np.zeros_like(W1)
    v['W2_vel'] = np.zeros_like(W2)
    v['b1_vel'] = np.zeros_like(b1)
    v['b2_vel'] = np.zeros_like(b2)
    v['X_vel'] = np.zeros((3 * D, batch_size))
    # I have returned dictionary to extract easily later.
    return v


def q2_updating_parameters_with_momentum(v, momentum, grad, parameters, learning_rate):
    

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    

    W1_vel = v['W1_vel']
    W2_vel = v['W2_vel']
    b1_vel = v['b1_vel']
    b2_vel = v['b2_vel']
   

    dW1 = grad['dW1']
    dW2 = grad['dW2']
    db1 = grad['db1']
    db2 = grad['db2']
    
    # Lets apply the momentum 
    W1_vel = momentum * W1_vel + (1 - momentum) * dW1
    W2_vel = momentum * W2_vel + (1 - momentum) * dW2
    b1_vel = momentum * b1_vel + (1 - momentum) * db1
    b2_vel = momentum * b2_vel + (1 - momentum) * db2
    v={"W1_vel": W1_vel, "W2_vel": W2_vel, "b2_vel": b2_vel, "b1_vel": b1_vel}

    # Now, lets do the parameter updates.
    W1 = W1 - learning_rate * W1_vel
    W2 = W2 - learning_rate * W2_vel
    b1 = b1 - learning_rate * b1_vel
    b2 = b2 - learning_rate * b2_vel
    
    # I have stored the updated parameters
    updated_parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}
    return updated_parameters


def q2_train_model(inputt, parameters, v, output, momentum = 0.85, learning_rate = 0.15):
    # Lets train our model end to end now.
    # First forward pass, then gradient calculations and finally the updating.
    forwardpass = q2_forward_pass(inputt.T, parameters, output)         
    grad = q2_grads(forwardpass, output, parameters)
    updated_parameter = q2_updating_parameters_with_momentum(v, momentum, grad, parameters, learning_rate)
    
    return forwardpass, grad, updated_parameter,forwardpass["cost"]


def q2_train_embeddings(trainx, traind, valx, vald, D=32, P=256, vocab_size=250, batch_size=200, momentum=0.85, learning_rate=0.15, epochs=30, patience=5):
    # Initialize parameters
    Wemb = np.random.normal(loc=0, scale=0.01, size=(vocab_size, D))
    n_samples = trainx.shape[0]

    parameter = q2_initialize_w_and_b(D, P, vocab_size)
    v = q2_defining_velocity(parameter, D, batch_size)
    one_hot_output = q2_one_hot_encoding(data=traind, num_of_samples=traind.shape[0], vocab_size=vocab_size)
    val_one_hot_output = q2_one_hot_encoding(data=vald, num_of_samples=vald.shape[0], vocab_size=vocab_size)

    best_val_cost = 15
    early_stopper = 0
    training_costs = []
    validation_costs = []
    validation_accuracies = []

    for epoch in range(epochs):
        # I have shuffled the samples for each epoch
        indices = np.random.permutation(n_samples)
        cost = 0  

        for i in range(0, n_samples, batch_size):
            # Prepare batch data
            batch_indices = np.sort(indices[i:i + batch_size])
            batch_data = trainx[batch_indices, :]
            out1hot1 = one_hot_output[batch_indices, :]

            # Get input vector and train
            input_vector = q2_word_to_vec(batch_data, D, Wemb)
            forwardpass, grad, parameter, batch_cost = q2_train_model(input_vector, parameter, v, out1hot1, 
                                                                   momentum=momentum, learning_rate=learning_rate)

            # Update momentum and embeddings
            current_batch_size = batch_indices.shape[0]
            # I have updated with momentum.
            v['X_vel'][:, :current_batch_size] = momentum * v['X_vel'][:, :current_batch_size] \
                                                + (1 - momentum) * grad['dX']
            
            # I have splitted the gradients for three words
            matrix1 = v['X_vel'][:D, :].T
            matrix2 = v['X_vel'][D:2*D, :].T
            matrix3 = v['X_vel'][2*D:3*D, :].T

            # I have done the embedding matrix update here finding the corresponding row.
            for j in range(current_batch_size):
                Wemb[batch_data[j, 0] - 1] -= learning_rate * matrix1[j, :D]
                Wemb[batch_data[j, 1] - 1] -= learning_rate * matrix2[j, :D]
                Wemb[batch_data[j, 2] - 1] -= learning_rate * matrix3[j, :D]

            # To accumulate
            cost = cost + batch_cost

        # I have done the similar operations for my validation data.
        val_input_vector = q2_word_to_vec(valx, D, Wemb)
        val_forwardpass = q2_forward_pass(val_input_vector.T, parameter, val_one_hot_output)
        val_predicted_output = np.argmax(val_forwardpass['A2'], axis=0)
        val_ground_truth = np.argmax(val_one_hot_output, axis=1)
        val_correct_predictions = np.sum(val_predicted_output == val_ground_truth)
        validation_percentage = val_correct_predictions / val_ground_truth.shape[0]

        val_cost = val_forwardpass['cost']


        # I have defined my early stopping function
        if best_val_cost - val_cost > 0.05:
            best_val_cost = val_cost
            early_stopper = 0
        else:
            early_stopper += 1
            if early_stopper >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation cost: {best_val_cost:.4f}")
                break

        # 
        training_costs.append(cost / (n_samples // batch_size))
        validation_costs.append(val_cost)
        validation_accuracies.append(validation_percentage * 100)

        print(f"Epoch {epoch + 1}/{epochs} completed. Training Cost: {cost / (n_samples // batch_size)}")
        print(f"Validation Cost: {val_cost:.4f}, Validation Accuracy: {validation_percentage * 100:.2f}%")

    # I have plotted the training loss, validation loss, and validation accuracy
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.plot(training_costs, label = 'Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Percentage')
    

    plt.subplot(1, 3, 2)
    plt.plot(validation_costs, label='Validation Loss')
    plt.title('Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Percentage')
    

    plt.subplot(1, 3, 3)
    plt.plot(validation_accuracies, label = 'Validation Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Percentage')
    
    plt.tight_layout()
    plt.show()

    return Wemb, parameter


# Load the dataset
with h5py.File('data2.h5', 'r') as file:
    trainx = np.array(file['trainx'])
    traind = np.array(file['traind'])
    valx = np.array(file['valx'])
    vald = np.array(file['vald'])
    testx = np.array(file['testx'])
    testd = np.array(file['testd'])
    words = np.array(file['words'])

# Store results for different configurations
results = {}

# Trying out different values of D and P
D_P_values = [(32, 256), (16, 128), (8, 64)]

for D, P in D_P_values:
    print(f"\nTraining the model with D={D}, P={P}")
    Wemb, parameters = q2_train_embeddings(
        trainx, traind, valx, vald, 
        D=D, P=P, vocab_size=250, 
        batch_size=200, momentum=0.85, 
        learning_rate=0.15, epochs=50, 
        patience=2
    )
    
    
    results[(D, P)] = {"Wemb": Wemb, "parameters": parameters}


# Best results obtained with D=32 and P=256
selected_config = (32, 256)
selected_Wemb = results[selected_config]["Wemb"]
selected_parameters = results[selected_config]["parameters"]


words = [word.decode('utf-8') if isinstance(word, bytes) else word for word in words]


out1hot_test = q2_one_hot_encoding(testd, testd.shape[0], vocab_size=250)


random_indices = np.random.choice(testx.shape[0], size=5, replace=False)
random_samples = testx[random_indices]
random_outs = out1hot_test[random_indices]

# Loop through the selected trigrams
for i in range(len(random_samples)):
    # Get the trigram indices and map them to their actual words
    trigram_indices = random_samples[i]
    trigram_words = [words[idx - 1] for idx in trigram_indices]  # Adjust for 1-based indexing
    print(f"Trigram {i + 1}: {' '.join(trigram_words)}")

    
    input_vector = q2_word_to_vec(np.array([trigram_indices]), D=32, embedding_matrix=selected_Wemb)

    
    output = q2_forward_pass(input_vector.T, selected_parameters, np.array([random_outs[i]]))
    predictions = output["A2"]  # Probabilities for each word

    
    predicted_idx = np.argmax(predictions[:, 0])  
    predicted_word = words[predicted_idx]
    print(f"  Predicted Word: {predicted_word}")

    
    top_10_indices = np.argsort(predictions[:, 0])[-10:]  # Get indices of the top 10 probabilities
    top_10_probs = predictions[top_10_indices, 0]

    print("Top 10 Predictions:")
    for rank in range(10):
        word_idx = top_10_indices[9 - rank]  # Reverse order to show highest probabilities first
        word = words[word_idx]
        prob = top_10_probs[9 - rank]
        print(f"    Rank {rank + 1}: {word} (Probability: {prob:.4f})")

    
    plt.figure()
    plt.barh([words[idx] for idx in reversed(top_10_indices)], list(reversed(top_10_probs)))
    plt.xlabel("Probability")
    plt.ylabel("Words")
    plt.title(f"Top 10 Predictions for Trigram {i + 1}")
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import h5py


def q2_initialize_w_and_b(D, P, vocab_size = 250):
    # I have initialized the weights using Gaussian distribution with 0 mean 0.01 variance.
    W1 = np.random.normal(0, 0.01, (P, 3 * D))
    b1 = np.random.normal(0, 0.01, (P, 1))
    W2 = np.random.normal(0, 0.01, (vocab_size, P))
    b2 = np.random.normal(0, 0.01, (vocab_size, 1))

    parameters = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    return parameters
    

def q2_word_to_vec(data, D, embedding_matrix):
    # In here, I take the 3 word indices and convert into their word embeddings. 
    # Then I concatenate them to make input vector which is ready for neural network.
    inputt = np.zeros((data[:,].shape[0], 3*D))
    for i in range(data[:, ].shape[0]):
        index = data[i] 
        first_word = embedding_matrix[index[0]-1, :] 
        second_word = embedding_matrix[index[1]-1, :] 
        third_word = embedding_matrix[index[2]-1, :]         
        all_words = np.concatenate([first_word, second_word, third_word])  
        inputt[i, :] = all_words        
    return inputt
    

def q2_output_word_to_vec(data, D, embedding_matrix):
    # The similar process is done which I have done in q2_word_to_vec function.
    x = np.zeros((data.shape[0], D))
    for i in range(data.shape[0]):
        indices = data[i]
        word1 = embedding_matrix[indices - 1, :]  
        x[i, :] = word1
    return x

def q2_sigmoid(z):
    # I have defined my q2_sigmoid function.
    sigm = 1/ (1 + np.exp(-z))
    return sigm


def q2_derivative_sigmoid(z):
    # To use the derivative of q2_sigmoid easily, I have defined 
    sigm_der = q2_sigmoid(z) * (1 - q2_sigmoid(z))
    return sigm_der

def q2_softmax(z):
    # To find the probabilities at the end of the neural network, I have defined q2_softmax function.
    # However, since the regular q2_softmax suffers from converging problems, I have defined different 
    # version of the q2_softmax which is stable q2_softmax. I observed that it gives more stabilized results.
    probability = np.exp(z - np.max(z,axis=0)) / np.sum(np.exp(z - np.max(z ,axis=0)), axis=0, keepdims=True)
    return probability


def q2_one_hot_encoding(data, num_of_samples, vocab_size = 250):
    # Here, I have converted my outputs into one hot encoding representation.
    one_hot = np.zeros((num_of_samples , vocab_size))
    for samples in range(num_of_samples):
        label = data[samples]
        one_hot[samples, label - 1] = 1 
    return one_hot

def q2_forward_pass(data, parameters, output):
    N = data.shape[1]

    # First I have extracted the required weights and biases.
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    
    # Here is the standart forward pass. 
    # Z represents the pre activation form and A represents the after activation form.
    Z1 = np.dot(W1, data) + b1
    A1 = 1/ (1 + np.exp(-Z1))
    Z2 = np.dot(W2, A1) + b2
    A2 =  np.exp(Z2 - np.max(Z2,axis=0)) / np.sum(np.exp(Z2 - np.max(Z2 ,axis=0)), axis=0, keepdims=True)

    # Our cost is
    cost = ( -1 / N) *np.sum(output.T * np.log(A2))

    # I have stored the required variables to use later.
    cache = {"X": data, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "cost": cost}
   
    return cache


def q2_grads(cache, actual_output, parameters):
    
    # Lets extract the required variables.
    X = cache['X']
    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']
    cost = cache['cost']

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    

    N = X.shape[1]

    # I have done the backprop here.
    dZ2 = (A2  - (actual_output).T)
    dW2 = (1 / N ) * np.dot(dZ2, A1.T)
    db2 = np.mean(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * q2_derivative_sigmoid(Z1)
    dW1 = (1 / N) * np.dot(dZ1, X.T)
    db1 = np.mean(dZ1, axis=1, keepdims=True)
    dX = (1 / 3 * N) * np.dot(W1.T, dZ1)

    # I have stored the gradients to use to update later.
    grad = {"dX": dX, "dW1": dW1, "dW2": dW2, "db1": db1, "db2": db2}
    return grad


def q2_defining_velocity(parameters, D, batch_size):

    
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # Lets define our velocities to use them in momemtum later.
    v = {}
    v['W1_vel'] = np.zeros_like(W1)
    v['W2_vel'] = np.zeros_like(W2)
    v['b1_vel'] = np.zeros_like(b1)
    v['b2_vel'] = np.zeros_like(b2)
    v['X_vel'] = np.zeros((3 * D, batch_size))
    # I have returned dictionary to extract easily later.
    return v


def q2_updating_parameters_with_momentum(v, momentum, grad, parameters, learning_rate):
    

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    

    W1_vel = v['W1_vel']
    W2_vel = v['W2_vel']
    b1_vel = v['b1_vel']
    b2_vel = v['b2_vel']
   

    dW1 = grad['dW1']
    dW2 = grad['dW2']
    db1 = grad['db1']
    db2 = grad['db2']
    
    # Lets apply the momentum 
    W1_vel = momentum * W1_vel + (1 - momentum) * dW1
    W2_vel = momentum * W2_vel + (1 - momentum) * dW2
    b1_vel = momentum * b1_vel + (1 - momentum) * db1
    b2_vel = momentum * b2_vel + (1 - momentum) * db2
    v={"W1_vel": W1_vel, "W2_vel": W2_vel, "b2_vel": b2_vel, "b1_vel": b1_vel}

    # Now, lets do the parameter updates.
    W1 = W1 - learning_rate * W1_vel
    W2 = W2 - learning_rate * W2_vel
    b1 = b1 - learning_rate * b1_vel
    b2 = b2 - learning_rate * b2_vel
    
    # I have stored the updated parameters
    updated_parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}
    return updated_parameters


def q2_train_model(inputt, parameters, v, output, momentum = 0.85, learning_rate = 0.15):
    # Lets train our model end to end now.
    # First forward pass, then gradient calculations and finally the updating.
    forwardpass = q2_forward_pass(inputt.T, parameters, output)         
    grad = q2_grads(forwardpass, output, parameters)
    updated_parameter = q2_updating_parameters_with_momentum(v, momentum, grad, parameters, learning_rate)
    
    return forwardpass, grad, updated_parameter,forwardpass["cost"]


def q2_train_embeddings(trainx, traind, valx, vald, D=32, P=256, vocab_size=250, batch_size=200, momentum=0.85, learning_rate=0.15, epochs=30, patience=5):
    # Initialize parameters
    Wemb = np.random.normal(loc=0, scale=0.01, size=(vocab_size, D))
    n_samples = trainx.shape[0]

    parameter = q2_initialize_w_and_b(D, P, vocab_size)
    v = q2_defining_velocity(parameter, D, batch_size)
    one_hot_output = q2_one_hot_encoding(data=traind, num_of_samples=traind.shape[0], vocab_size=vocab_size)
    val_one_hot_output = q2_one_hot_encoding(data=vald, num_of_samples=vald.shape[0], vocab_size=vocab_size)

    best_val_cost = 15
    early_stopper = 0
    training_costs = []
    validation_costs = []
    validation_accuracies = []

    for epoch in range(epochs):
        # I have shuffled the samples for each epoch
        indices = np.random.permutation(n_samples)
        cost = 0  

        for i in range(0, n_samples, batch_size):
            # Prepare batch data
            batch_indices = np.sort(indices[i:i + batch_size])
            batch_data = trainx[batch_indices, :]
            out1hot1 = one_hot_output[batch_indices, :]

            # Get input vector and train
            input_vector = q2_word_to_vec(batch_data, D, Wemb)
            forwardpass, grad, parameter, batch_cost = q2_train_model(input_vector, parameter, v, out1hot1, 
                                                                   momentum=momentum, learning_rate=learning_rate)

            # Update momentum and embeddings
            current_batch_size = batch_indices.shape[0]
            # I have updated with momentum.
            v['X_vel'][:, :current_batch_size] = momentum * v['X_vel'][:, :current_batch_size] \
                                                + (1 - momentum) * grad['dX']
            
            # I have splitted the gradients for three words
            matrix1 = v['X_vel'][:D, :].T
            matrix2 = v['X_vel'][D:2*D, :].T
            matrix3 = v['X_vel'][2*D:3*D, :].T

            # I have done the embedding matrix update here finding the corresponding row.
            for j in range(current_batch_size):
                Wemb[batch_data[j, 0] - 1] -= learning_rate * matrix1[j, :D]
                Wemb[batch_data[j, 1] - 1] -= learning_rate * matrix2[j, :D]
                Wemb[batch_data[j, 2] - 1] -= learning_rate * matrix3[j, :D]

            # To accumulate
            cost = cost + batch_cost

        # I have done the similar operations for my validation data.
        val_input_vector = q2_word_to_vec(valx, D, Wemb)
        val_forwardpass = q2_forward_pass(val_input_vector.T, parameter, val_one_hot_output)
        val_predicted_output = np.argmax(val_forwardpass['A2'], axis=0)
        val_ground_truth = np.argmax(val_one_hot_output, axis=1)
        val_correct_predictions = np.sum(val_predicted_output == val_ground_truth)
        validation_percentage = val_correct_predictions / val_ground_truth.shape[0]

        val_cost = val_forwardpass['cost']


        # I have defined my early stopping function
        if best_val_cost - val_cost > 0.05:
            best_val_cost = val_cost
            early_stopper = 0
        else:
            early_stopper += 1
            if early_stopper >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation cost: {best_val_cost:.4f}")
                break

        # 
        training_costs.append(cost / (n_samples // batch_size))
        validation_costs.append(val_cost)
        validation_accuracies.append(validation_percentage * 100)

        print(f"Epoch {epoch + 1}/{epochs} completed. Training Cost: {cost / (n_samples // batch_size)}")
        print(f"Validation Cost: {val_cost:.4f}, Validation Accuracy: {validation_percentage * 100:.2f}%")

    # I have plotted the training loss, validation loss, and validation accuracy
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.plot(training_costs, label = 'Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Percentage')
    

    plt.subplot(1, 3, 2)
    plt.plot(validation_costs, label='Validation Loss')
    plt.title('Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Percentage')
    

    plt.subplot(1, 3, 3)
    plt.plot(validation_accuracies, label = 'Validation Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Percentage')
    
    plt.tight_layout()
    plt.show()

    return Wemb, parameter

def q2():
    # Load the dataset
    with h5py.File('data2.h5', 'r') as file:
        trainx = np.array(file['trainx'])
        traind = np.array(file['traind'])
        valx = np.array(file['valx'])
        vald = np.array(file['vald'])
        testx = np.array(file['testx'])
        testd = np.array(file['testd'])
        words = np.array(file['words'])

    # Store results for different configurations
    results = {}

    # Trying out different values of D and P
    D_P_values = [(32, 256), (16, 128), (8, 64)]

    for D, P in D_P_values:
        print(f"\nTraining the model with D={D}, P={P}")
        Wemb, parameters = q2_train_embeddings(
            trainx, traind, valx, vald, 
            D=D, P=P, vocab_size=250, 
            batch_size=200, momentum=0.85, 
            learning_rate=0.15, epochs=50, 
            patience=2
        )
        
        
        results[(D, P)] = {"Wemb": Wemb, "parameters": parameters}


    # Best results obtained with D=32 and P=256
    selected_config = (32, 256)
    selected_Wemb = results[selected_config]["Wemb"]
    selected_parameters = results[selected_config]["parameters"]


    words = [word.decode('utf-8') if isinstance(word, bytes) else word for word in words]


    out1hot_test = q2_one_hot_encoding(testd, testd.shape[0], vocab_size=250)


    random_indices = np.random.choice(testx.shape[0], size=5, replace=False)
    random_samples = testx[random_indices]
    random_outs = out1hot_test[random_indices]

    # Loop through the selected trigrams
    for i in range(len(random_samples)):
        # Get the trigram indices and map them to their actual words
        trigram_indices = random_samples[i]
        trigram_words = [words[idx - 1] for idx in trigram_indices]  # Adjust for 1-based indexing
        print(f"Trigram {i + 1}: {' '.join(trigram_words)}")

        
        input_vector = q2_word_to_vec(np.array([trigram_indices]), D=32, embedding_matrix=selected_Wemb)

        
        output = q2_forward_pass(input_vector.T, selected_parameters, np.array([random_outs[i]]))
        predictions = output["A2"]  # Probabilities for each word

        
        predicted_idx = np.argmax(predictions[:, 0])  
        predicted_word = words[predicted_idx]
        print(f"  Predicted Word: {predicted_word}")

        
        top_10_indices = np.argsort(predictions[:, 0])[-10:]  # Get indices of the top 10 probabilities
        top_10_probs = predictions[top_10_indices, 0]

        print("Top 10 Predictions:")
        for rank in range(10):
            word_idx = top_10_indices[9 - rank]  # Reverse order to show highest probabilities first
            word = words[word_idx]
            prob = top_10_probs[9 - rank]
            print(f"    Rank {rank + 1}: {word} (Probability: {prob:.4f})")

        
        plt.figure()
        plt.barh([words[idx] for idx in reversed(top_10_indices)], list(reversed(top_10_probs)))
        plt.xlabel("Probability")
        plt.ylabel("Words")
        plt.title(f"Top 10 Predictions for Trigram {i + 1}")
        plt.tight_layout()
        plt.show()







def rnn_init_xavier(shape):
    fan_in, fan_out = shape
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)


def rnn_sigmoid(z):
    clipped_z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-clipped_z))

def rnn_sigmoid_derivative(z):
    return z * (1 - z)

def tanh(z):
    clipped_z = np.clip(z, -50, 50)
    return np.tanh(clipped_z)

def rnn_tanh_derivative(z):
    return 1 - (z ** 2)

def rnn_relu(z):
    return np.maximum(0, z)

def rnn_relu_derivative(z):
    return (z > 0).astype(float)

def rnn_softmax(z):
    stabilizer = 1e-9
    shift_z = z - np.max(z, axis=1, keepdims=True)
    shift_z = np.clip(shift_z, -50, 50)
    exp_z = np.exp(shift_z)
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + stabilizer)

def rnn_cross_entropy(y_true, y_pred):
  
    stabilizer = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + stabilizer), axis=1))


def rnn_clip_gradients(grad, threshold = 7):
    return np.clip(grad, -threshold, threshold)


def forward_rnn(X, W_in, W_h, b_h):
    
    batch_size, time_steps, _ = X.shape
    hidden_size = W_in.shape[1]

    h_t = np.zeros((batch_size, hidden_size))
    H_states = []

    for t in range(time_steps):
        
        z = np.dot(X[:, t, :], W_in) + np.dot(h_t, W_h + b_h)
        h_t = tanh(z)
        H_states.append(h_t)
    
    return h_t, H_states

def rnn_forward_mlp(h_t, W_hidden, b_hidden, W_out, b_out):

    z_hidden = np.dot(h_t, W_hidden) + b_hidden
    h_hidden = rnn_relu(z_hidden)
    logits = np.dot(h_hidden, W_out) + b_out
    Y_pred = rnn_softmax(logits)
    return Y_pred, h_hidden, z_hidden



def rnn_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def train_rnn(X_train, Y_train, X_val, Y_val, X_test, Y_test, hidden_size=128, mlp_hidden=64, lr=0.001, batch_size=32, epochs=30,patience=5, momentum=0):
    # To get similar results each time
    np.random.seed(999)

    input_size = X_train.shape[2]
    output_size = Y_train.shape[1]

    # I have initialzied the weights
    W_in = rnn_init_xavier((input_size, hidden_size))
    W_h  = rnn_init_xavier((hidden_size, hidden_size))
    b_h  = np.zeros((1, hidden_size))

    W_hidden = rnn_init_xavier((hidden_size, mlp_hidden))
    b_hidden = np.zeros((1, mlp_hidden))

    W_out = rnn_init_xavier((mlp_hidden, output_size))
    b_out = np.zeros((1, output_size))

    # I have initialized the velocities to use them in momentum later
    v_W_in = np.zeros_like(W_in)
    v_W_h  = np.zeros_like(W_h)
    v_b_h  = np.zeros_like(b_h)
    v_W_hidden = np.zeros_like(W_hidden)
    v_b_hidden = np.zeros_like(b_hidden)
    v_W_out = np.zeros_like(W_out)
    v_b_out = np.zeros_like(b_out)
    
    # I have defined for early stopping
    best_val_loss = 50
    patience_counter = 0
    
    # I have defined required lists to store the values and plot them.
    train_losses_per_epoch = []
    train_accs_per_epoch   = []
    val_losses_per_epoch   = []
    val_accs_per_epoch     = []

    
    for epoch in range(epochs):
        # I am shuffling the dataset for each epoch
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        
        n_batches = X_train.shape[0] // batch_size

        epoch_loss = 0
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i : i + batch_size]
            Y_batch = Y_train[i : i + batch_size]
            bs = X_batch.shape[0]  

            # Forward pass
            h_final, H_states = forward_rnn(X_batch, W_in, W_h, b_h)
            Y_pred, h_hidden, z_hidden = rnn_forward_mlp(h_final, W_hidden, b_hidden, W_out, b_out)

            loss = rnn_cross_entropy(Y_batch, Y_pred)
            epoch_loss += loss

            d_logits = (Y_pred - Y_batch) / bs  
            d_W_out = np.dot(h_hidden.T, d_logits)    
            d_b_out = np.sum(d_logits, axis=0, keepdims=True)

            # The part for MLP
            d_h_hidden = np.dot(d_logits, W_out.T )            
            d_z_hidden = d_h_hidden * rnn_relu_derivative(z_hidden)
            d_W_hidden = np.dot(h_final.T , d_z_hidden)          
            d_b_hidden = np.sum(d_z_hidden, axis=0, keepdims=True)

            
            # Gradients
            d_h_final = np.dot(d_z_hidden, W_hidden.T)  # shape (bs, hidden_size)

            d_W_in = np.zeros_like(W_in)
            d_W_h  = np.zeros_like(W_h)
            d_b_h  = np.zeros_like(b_h)

            dh_next = d_h_final  
            time_steps = X_batch.shape[1]
            for t in reversed(range(time_steps)):
                h_current = H_states[t]
                dh_raw = dh_next * rnn_tanh_derivative(h_current)
                
                d_W_in = d_W_in + np.dot(X_batch[:, t, :].T , dh_raw)
                h_prev = H_states[t - 1] if t > 0 else np.zeros_like(h_current)
                d_W_h  += np.dot(h_prev.T, dh_raw)
                d_b_h  += np.sum(dh_raw, axis=0, keepdims=True)

                dh_next = np.dot(dh_raw, W_h.T)
                
            # I have clipped my gradients to get better convergence
            d_W_in     = rnn_clip_gradients(d_W_in)
            d_W_h      = rnn_clip_gradients(d_W_h)
            d_b_h      = rnn_clip_gradients(d_b_h)
            d_W_hidden = rnn_clip_gradients(d_W_hidden)
            d_b_hidden = rnn_clip_gradients(d_b_hidden)
            d_W_out    = rnn_clip_gradients(d_W_out)
            d_b_out    = rnn_clip_gradients(d_b_out)

            
            
            v_W_in = momentum * v_W_in + (1 - momentum) * d_W_in
            v_W_h  = momentum * v_W_h  + (1 - momentum) * d_W_h
            v_b_h  = momentum * v_b_h  + (1 - momentum) * d_b_h
            v_W_hidden = momentum * v_W_hidden + (1 - momentum) * d_W_hidden
            v_b_hidden = momentum * v_b_hidden + (1 - momentum) * d_b_hidden
            v_W_out    = momentum * v_W_out    + (1 - momentum) * d_W_out
            v_b_out    = momentum * v_b_out    + (1 - momentum) * d_b_out

            
            W_in = W_in - lr * v_W_in
            W_h = W_h - lr * v_W_h
            b_h =  b_h - lr * v_b_h
            W_hidden =  W_hidden - lr * v_W_hidden
            b_hidden =  b_hidden - lr * v_b_hidden
            W_out = W_out - lr * v_W_out
            b_out = b_out - lr * v_b_out

        avg_train_loss = epoch_loss / n_batches

       
        h_train, _ = forward_rnn(X_train, W_in, W_h, b_h)
        Y_pred_train, _, _ = rnn_forward_mlp(h_train, W_hidden, b_hidden, W_out, b_out)
        train_accuracy = np.mean(np.argmax(Y_pred_train, axis=1) == np.argmax(Y_train, axis=1))

        # I have done similar process for validation data
        h_val, _ = forward_rnn(X_val, W_in, W_h, b_h)
        Y_pred_val, _, _ = rnn_forward_mlp(h_val, W_hidden, b_hidden, W_out, b_out)
        val_loss = rnn_cross_entropy(Y_val, Y_pred_val)
        val_accuracy = np.mean(np.argmax(Y_pred_val, axis=1) == np.argmax(Y_val, axis=1))

        train_losses_per_epoch.append(avg_train_loss)
        train_accs_per_epoch.append(train_accuracy)
        val_losses_per_epoch.append(val_loss)
        val_accs_per_epoch.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy*100:.2f}%")

        # My early stopping algorithm
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

  
    h_test, _ = forward_rnn(X_test, W_in, W_h, b_h)
    Y_pred_test, _, _ = rnn_forward_mlp(h_test, W_hidden, b_hidden, W_out, b_out)
    test_accuracy = np.mean(np.argmax(Y_pred_test, axis=1) == np.argmax(Y_test, axis=1))
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

    # I am using confusion matrix for test data
    print("\nTest Confusion Matrix:")
    test_true_labels = np.argmax(Y_test, axis=1)
    test_preds = np.argmax(Y_pred_test, axis=1)
    test_cm = rnn_confusion_matrix(test_true_labels, test_preds, output_size)
    print(test_cm)

    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    train_true_labels = np.argmax(Y_train, axis=1)
    train_preds = np.argmax(Y_pred_train, axis=1)
    
    train_cm = rnn_confusion_matrix(train_true_labels, train_preds, output_size)

    plt.figure(figsize=(10, 8))
    # I have 
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Train Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    

    
    return (train_losses_per_epoch, train_accs_per_epoch,
            val_losses_per_epoch,   val_accs_per_epoch,
            test_accuracy)

import numpy as np
import matplotlib.pyplot as plt

import h5py



def lstm_init_xavier(desired_shape):
    # To define my learnable parameters like weights and biases, I have first defined xavier initialization.
    Lin, Lout = desired_shape
    boundary = np.sqrt(6 / (Lin + Lout))
    return np.random.uniform(-boundary, boundary, size = desired_shape)


def lstm_initialize_mlp_weights(hidden_size, mlp_hidden_size, mlp_hidden_size2, output_size):
    # I have initialized the weights using xavier init
    W_hidden = lstm_init_xavier((hidden_size, mlp_hidden_size))
    b_hidden = lstm_init_xavier((1, mlp_hidden_size))
    W_hidden2 = lstm_init_xavier((mlp_hidden_size, mlp_hidden_size2))
    b_hidden2 = lstm_init_xavier((1, mlp_hidden_size2))
    W_out = lstm_init_xavier((mlp_hidden_size2, output_size))
    b_out = lstm_init_xavier((1, output_size))

    return W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out

def lstm_init_velocities(W_in, W_h, b, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out):
    # I have defined my velocities to use it later
    vel_W_in     = np.zeros_like(W_in)
    vel_W_h      = np.zeros_like(W_h)
    vel_b        = np.zeros_like(b)
    vel_W_hidden = np.zeros_like(W_hidden)
    vel_b_hidden = np.zeros_like(b_hidden)
    vel_W_hidden2= np.zeros_like(W_hidden2)
    vel_b_hidden2= np.zeros_like(b_hidden2)
    vel_W_out    = np.zeros_like(W_out)
    vel_b_out    = np.zeros_like(b_out)

    return vel_W_in, vel_W_h, vel_b, vel_W_hidden, vel_b_hidden, vel_W_hidden2, vel_b_hidden2, vel_W_out, vel_b_out


def lstm_sigmoid(z):
    # I have clipped my sigmoid for better convergence
    clipped_z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-clipped_z))

def lstm_sigmoid_derivative(z):
    # Derivative of sigmoid
    return z * (1 - z)

def tanh(z):
    # Clipped tanh for better convergence
    clipped_z = np.clip(z, -50, 50)
    return np.tanh(clipped_z)

def lstm_tanh_derivative(z):
    # Derivative of tanh
    return 1 - (z ** 2)

def lstm_relu(z):
    # I have defined relu activation
    return np.maximum(0, z)

def lstm_relu_derivative(z):
    # I have defined derivative of relu. It gives either 0 or 1.
    return (z > 0).astype(float)

def lstm_softmax(z):
    # Subtract row-wise max for numerical stability
    stabilizer = 1e-9
    shift_z = z - np.max(z, axis=1, keepdims=True)
    shift_z = np.clip(shift_z, -50, 50)
    exp_z = np.exp(shift_z)
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + stabilizer)


def lstm_cross_entropy(y_true, y_pred):
    # I have defined stabilizer for better convergence
    stabilizer = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + stabilizer), axis=1))

def lstm_clip_gradients(grad, threshold = 10):
    # I have used gradient clipping technique to eliminate exploding gradient problem in backprop.
    # Threshold was determined after some trial.
    return np.clip(grad, -threshold, threshold)

def lstm_cross_entropy(y_true, y_pred):
    # Cross entropy is like negative log likelihood. Thus, I have tried to implement like that
    # I have defined stabilizer to get better convergence
    stabilizer = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + stabilizer), axis=1))

def lstm_clip_each_gradients(grad_W_in_lstm, grad_W_hid_lstm, grad_b_lstm, grad_W_hid_mlp, grad_b_hid_mlp, grad_W_hid2_mlp, grad_b_hid2_mlp, grad_W_out, grad_b_out):
    grad_W_in_lstm = lstm_clip_gradients(grad_W_in_lstm)
    grad_W_hid_lstm = lstm_clip_gradients(grad_W_hid_lstm)
    grad_b_lstm = lstm_clip_gradients(grad_b_lstm)
    grad_W_hid_mlp = lstm_clip_gradients(grad_W_hid_mlp)
    grad_b_hid_mlp = lstm_clip_gradients(grad_b_hid_mlp)
    grad_W_hid2_mlp = lstm_clip_gradients(grad_W_hid2_mlp)
    grad_b_hid2_mlp = lstm_clip_gradients(grad_b_hid2_mlp)
    grad_W_out = lstm_clip_gradients(grad_W_out)
    grad_b_out = lstm_clip_gradients(grad_b_out)
    return grad_W_in_lstm, grad_W_hid_lstm, grad_b_lstm, grad_W_hid_mlp, grad_b_hid_mlp, grad_W_hid2_mlp, grad_b_hid2_mlp, grad_W_out, grad_b_out
    

def forward_pass_lstm(X, W_in, W_h, b, hidden_size):
    # Here I am doing the forward pass of the LSTM
    batch_size, time_steps, input_size = X.shape

    # I have initialized hidden & cell states
    hid_states = np.zeros((batch_size, hidden_size))  
    cell_states = np.zeros((batch_size, hidden_size))  

    # I have defined lists for use it later
    forget_actv = []
    input_actv = []
    output_actv = []
    cand_actv = []
    C_list = []
    h_list = []

    for t in range(time_steps):
        x_t = X[:, t, :]  
        z = np.dot(x_t, W_in) + np.dot(hid_states, W_h) + b

        forget_gate = lstm_sigmoid(z[:, :hidden_size])                  
        in_gate = lstm_sigmoid(z[:, hidden_size:2*hidden_size])     
        out_gate = lstm_sigmoid(z[:, 2*hidden_size:3*hidden_size])   
        cand = tanh(z[:, 3*hidden_size:4*hidden_size])      

        cell_states = forget_gate * cell_states + in_gate * cand
        hid_states = out_gate * tanh(cell_states)

        forget_actv.append(forget_gate)
        input_actv.append(in_gate)
        output_actv.append(out_gate)
        cand_actv.append(cand)
        C_list.append(cell_states.copy())
        h_list.append(hid_states.copy())

    cache = {
        'X': X,
        'forget_actv': forget_actv,
        'input_actv': input_actv,
        'output_actv': output_actv,
        'cand_actv': cand_actv,
        'C_list': C_list,
        'h_list': h_list,
        'W_in': W_in,
        'W_h': W_h,
        'b': b
    }
    return hid_states, cache



def backward_pass_lstm(dh_final, cache, grad_last_cell, hidden_size):
    # To do backward of LSTM, I have first extract the varaibles

    X = cache['X']
    forget_actv = cache['forget_actv']
    input_actv = cache['input_actv']
    output_actv = cache['output_actv']
    cand_actv = cache['cand_actv']
    C_list = cache['C_list']
    h_list = cache['h_list']
    W_in = cache['W_in']
    W_h = cache['W_h']

    # My gradients
    batch_size, time_steps, input_size = X.shape
    dh_next = dh_final
    dC_next = grad_last_cell
    dW_in = np.zeros_like(W_in)
    dW_h  = np.zeros_like(W_h)
    db = np.zeros((1, 4*hidden_size))


    for t in reversed(range(time_steps)):
        forget_gate = forget_actv[t]
        in_gate = input_actv[t]
        out_gate = output_actv[t]
        cand = cand_actv[t]

        C_t = C_list[t]
        if t > 0:
            prev_cell = C_list[t-1]
        else:
            prev_cell = np.zeros_like(C_t)
            
        h_t = h_list[t]
        if t > 0:
            h_prev = h_list[t-1]
        else:
            h_prev = np.zeros_like(h_t)

        # My output gate
        do_t = dh_next * tanh(C_t)
        do_raw = do_t * lstm_sigmoid_derivative(out_gate)

        # My cell state
        dC_t = dh_next * out_gate * (1 - tanh(C_t)**2) + dC_next

        # My forget gate
        df_t = dC_t * prev_cell
        df_raw = df_t * lstm_sigmoid_derivative(forget_gate)

        # My input gate
        di_t = dC_t * cand
        di_raw = di_t * lstm_sigmoid_derivative(in_gate)

        # My candidate 
        dg_t = dC_t * in_gate
        dg_raw = dg_t * lstm_tanh_derivative(cand)

        dz = np.hstack([df_raw, di_raw, do_raw, dg_raw])  

        x_t = X[:, t, :]
        dW_in = dW_in + np.dot(x_t.T, dz)
        dW_h = dW_h + np.dot(h_prev.T, dz)
        db = db + np.sum(dz, axis=0, keepdims=True)

        dh_prev = np.dot(dz, W_h.T)
        dC_prev = dC_t * forget_gate

        dh_next = dh_prev
        dC_next = dC_prev

    return dW_in, dW_h, db

def lstm_forward_pass_mlp(h_t, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out):
    # I have done the forward pass as before

    z_hidden = np.dot(h_t , W_hidden) + b_hidden  
    h_hidden = lstm_relu(z_hidden)

    z_hidden2 = np.dot(h_hidden, W_hidden2) + b_hidden2
    h_hidden2 = lstm_relu(z_hidden2)

    logits = np.dot(h_hidden2, W_out) + b_out  
    y_pred = lstm_softmax(logits)           

    return y_pred, h_hidden, z_hidden, h_hidden2, z_hidden2, logits



def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm



def train_lstm(X_train_full, Y_train_full, X_test, Y_test, params):
    
    # To get similar results
    np.random.seed(42)  

    # I have extracted hyperparameters
    lr = params['lr']
    momentum = params['momentum']
    batch_size = params['batch_size']
    epochs = params['epochs']
    hidden_size = params['hidden_size']
    mlp_hidden_size = params['mlp_hidden_size']
    mlp_hidden_size2 = params['mlp_hidden_size2']
    patience = params['patience']
    
    # Overriding from code
    lr = 0.1
    momentum = 0.85
    batch_size = 32
    epochs = 50
    hidden_size = 128
    mlp_hidden_size = 64
    mlp_hidden_size2 = 32
    patience = 5
    

    total_samples = X_train_full.shape[0]
    val_size = int(0.1 * total_samples)
    train_size = total_samples - val_size

    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train = X_train_full[train_indices]
    Y_train = Y_train_full[train_indices]
    X_val   = X_train_full[val_indices]
    Y_val   = Y_train_full[val_indices]

    time_steps = X_train.shape[1]
    input_size = X_train.shape[2]
    output_size = Y_train.shape[1]
    

    W_in = lstm_init_xavier((input_size, 4*hidden_size))
    W_h  = lstm_init_xavier((hidden_size, 4*hidden_size))
    b = lstm_init_xavier((1, 4*hidden_size))

    
    W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out = lstm_initialize_mlp_weights(hidden_size, mlp_hidden_size, mlp_hidden_size2, output_size)

    
    v_W_in, v_W_h, v_b, v_W_hidden, v_b_hidden, v_W_hidden2, v_b_hidden2, v_W_out, v_b_out = lstm_init_velocities(W_in, W_h, b, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out)


    best_cost = 50
    stopping_criteria = 0  # how many epochs with no improvement
    best_params = None  # to store the best weights

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(train_size)
        X_train = X_train[perm]
        Y_train = Y_train[perm]

        epoch_loss = 0
        correct_predictions = 0

        for start_idx in range(0, train_size, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx]
            bs = X_batch.shape[0]

            # Forward pass
            h_t, cache_lstm = forward_pass_lstm(X_batch, W_in, W_h, b, hidden_size)
            Y_pred, h_hidden, z_hidden, h_hidden2, z_hidden2, logits = lstm_forward_pass_mlp(
                h_t, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out
            )

            # Compute training loss
            loss = lstm_cross_entropy(Y_batch, Y_pred)
            epoch_loss += loss

            # Count correct predictions
            correct_predictions += np.sum(np.argmax(Y_pred, axis=1) == np.argmax(Y_batch, axis=1))

            # Backpropagation
            grad_out = (Y_pred - Y_batch) / bs
            grad_W_out = np.dot(h_hidden2.T, grad_out)
            grad_b_out = np.sum(grad_out, axis=0, keepdims=True)

            grad_hid2_mlp = np.dot(grad_out, W_out.T)
            d_z_hidden2 = grad_hid2_mlp * lstm_relu_derivative(z_hidden2)
            grad_W_hid2_mlp = np.dot(h_hidden.T, d_z_hidden2)
            grad_b_hid2_mlp = np.sum(d_z_hidden2, axis=0, keepdims=True)

            grad_hid_mlp = np.dot(d_z_hidden2, W_hidden2.T)
            d_z_hidden = grad_hid_mlp * lstm_relu_derivative(z_hidden)
            grad_W_hid_mlp = np.dot(h_t.T, d_z_hidden)
            grad_b_hid_mlp = np.sum(d_z_hidden, axis=0, keepdims=True)

            grad_hid_state = np.dot(d_z_hidden, W_hidden.T)
            grad_last_cell = np.zeros((bs, hidden_size))
            grad_W_in_lstm, grad_W_hid_lstm, grad_b_lstm = backward_pass_lstm(
                grad_hid_state, cache_lstm, grad_last_cell, hidden_size
            )

            grad_W_in_lstm, grad_W_hid_lstm, grad_b_lstm, grad_W_hid_mlp, grad_b_hid_mlp, grad_W_hid2_mlp, grad_b_hid2_mlp, grad_W_out, grad_b_out = lstm_clip_each_gradients(
                grad_W_in_lstm, grad_W_hid_lstm, grad_b_lstm, grad_W_hid_mlp, grad_b_hid_mlp, grad_W_hid2_mlp, grad_b_hid2_mlp, grad_W_out, grad_b_out
            )

            # Momentum update
            v_W_in = (momentum * v_W_in) - (lr * grad_W_in_lstm)
            v_W_h = (momentum * v_W_h)  - (lr * grad_W_hid_lstm)
            v_b  = (momentum * v_b) - (lr * grad_b_lstm)
            v_W_hidden = (momentum * v_W_hidden) - (lr * grad_W_hid_mlp)
            v_b_hidden = (momentum * v_b_hidden) - (lr * grad_b_hid_mlp)
            v_W_hidden2 = (momentum * v_W_hidden2) - (lr * grad_W_hid2_mlp)
            v_b_hidden2 = momentum * v_b_hidden2 - lr * grad_b_hid2_mlp
            v_W_out = (momentum * v_W_out) - (lr * grad_W_out)
            v_b_out = (momentum * v_b_out) - (lr * grad_b_out)

            # updates
            W_in = W_in + v_W_in
            W_h = W_h + v_W_h
            b = b + v_b
            W_hidden = W_hidden + v_W_hidden
            b_hidden = b_hidden + v_b_hidden
            W_hidden2 = W_hidden2 + v_W_hidden2
            b_hidden2 = b_hidden2 + v_b_hidden2
            W_out = W_out + v_W_out
            b_out = b_out + v_b_out
            

        
        train_loss = epoch_loss / (train_size // batch_size)
        train_accuracy = correct_predictions / train_size

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        

        
        h_val, _ = forward_pass_lstm(X_val, W_in, W_h, b, hidden_size)
        Y_val_pred, _, _, _, _, _ = lstm_forward_pass_mlp(h_val, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out)
        
        
        val_loss = lstm_cross_entropy(Y_val, Y_val_pred)
        
        val_accuracy = np.mean(np.argmax(Y_val_pred, axis=1) == np.argmax(Y_val, axis=1))

        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        
        if val_loss < best_cost:
            
            best_cost = val_loss
            stopping_criteria = 0
            
            best_params = (W_in, W_h, b,
                W_hidden, b_hidden,
                W_hidden2, b_hidden2,
                W_out, b_out
            )
        else:
            
            stopping_criteria += 1
            if stopping_criteria >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}, best validation loss: {best_cost:.4f}")
                break

   
    h_test, _ = forward_pass_lstm(X_test, W_in, W_h, b, hidden_size)
    Y_pred_test, _, _, _, _, _ = lstm_forward_pass_mlp(h_test, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out)

    
    preds = np.argmax(Y_pred_test, axis=1)
    labels = np.argmax(Y_test, axis=1)
    test_acc = np.mean(preds == labels) * 100
    print(f"Test Accuracy: {test_acc:.2f}%")

    
    num_classes = Y_test.shape[1]
    cm = compute_confusion_matrix(labels, preds, num_classes)
    print("Confusion Matrix:\n", cm)

    return train_losses, train_accuracies, val_losses, val_accuracies, test_acc, preds, labels, W_in, W_h, b, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out


import numpy as np
import matplotlib.pyplot as plt
import h5py


def gru_compute_confusion_matrix(y_true, y_pred, num_classes):

    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix

def gru_init_xavier(desired_shape):
    # To define my learnable parameters like weights and biases, I have first defined xavier initialization.
    Lin, Lout = desired_shape
    boundary = np.sqrt(6 / (Lin + Lout))
    return np.random.uniform(-boundary, boundary, size = desired_shape)

def gru_sigmoid(z):
    # Sigmoid for this task. I have clipped to get better convergence
    clipped_z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-clipped_z))

def gru_sigmoid_derivative(z):
    # Derivative of sigmoid
    return z * (1 - z)

def tanh(z):
    # tanh activation. I have clipped to get better convergence
    clipped_z = np.clip(z, -50, 50)
    return np.tanh(clipped_z)

def gru_tanh_derivative(z):
    # Derivative of tanh
    return 1 - (z ** 2)

def gru_relu(z):
    # Relu which gives positive numbers itself or zero
    return np.maximum(0, z)
    
def gru_relu_derivative(z):
    # I have designed such a way that it is either gives 0 or 1 as expected.
    return (z > 0).astype(float)

def gru_softmax(z):
    # Since the standart softmax does not work accurately, I have defined a different type of softmax
    # To get better convergence, I have defined stabilizer and subtracted the maximum value in the row from this row
    # Again, to get better convergence, I have used clipping.
    stabilizer = 1e-9
    shift_z = z - np.max(z, axis=1, keepdims=True)
    shift_z = np.clip(shift_z, -50, 50)
    exp_z = np.exp(shift_z)
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + stabilizer)


def gru_cross_entropy(y_true, y_pred):
    # Cross entropy is like negative log likelihood. Thus, I have tried to implement like that
    # I have defined stabilizer to get better convergence
    stabilizer = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + stabilizer), axis=1))


def gru_clip_gradients(grad, threshold = 10):
    # I have used gradient clipping technique to eliminate exploding gradient problem in backprop.
    # Threshold was determined after some trial.
    return np.clip(grad, -threshold, threshold)

def gru_clip_each_gradients(grad_W_in_gru, grad_W_hid_gru, grad_b_gru, grad_W_hid_mlp, grad_b_hid_mlp, grad_W_hid2_mlp, grad_b_hid2_mlp, grad_W_out, grad_b_out):
    # By using the my gru_clip_gradients function, I have clipped all gradients.
    grad_W_in_gru = gru_clip_gradients(grad_W_in_gru)
    grad_W_hid_gru = gru_clip_gradients(grad_W_hid_gru)
    grad_b_gru = gru_clip_gradients(grad_b_gru)
    grad_W_hid_mlp = gru_clip_gradients(grad_W_hid_mlp)
    grad_b_hid_mlp = gru_clip_gradients(grad_b_hid_mlp)
    grad_W_hid2_mlp = gru_clip_gradients(grad_W_hid2_mlp)
    grad_b_hid2_mlp = gru_clip_gradients(grad_b_hid2_mlp)
    grad_W_out = gru_clip_gradients(grad_W_out)
    grad_b_out = gru_clip_gradients(grad_b_out)
    
    return grad_W_in_gru, grad_W_hid_gru, grad_b_gru, grad_W_hid_mlp, grad_b_hid_mlp, grad_W_hid2_mlp, grad_b_hid2_mlp, grad_W_out, grad_b_out

def gru_initialize_mlp_weights(hidden_size, mlp_hidden_size, mlp_hidden_size2, output_size):
    # I have defined my learnable parameters for MLP using the xavier initialization.
    W_hidden = gru_init_xavier((hidden_size, mlp_hidden_size))
    b_hidden = gru_init_xavier((1, mlp_hidden_size))
    W_hidden2 = gru_init_xavier((mlp_hidden_size, mlp_hidden_size2))
    b_hidden2 = gru_init_xavier((1, mlp_hidden_size2))
    W_out = gru_init_xavier((mlp_hidden_size2, output_size))
    b_out = gru_init_xavier((1, output_size))

    return W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out
    
    
def gru_init_velocities(W_in, W_h, b, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out):
    # Here, I have created velocities to use it for momentum later.
    vel_W_in     = np.zeros_like(W_in)
    vel_W_h      = np.zeros_like(W_h)
    vel_b        = np.zeros_like(b)
    vel_W_hidden = np.zeros_like(W_hidden)
    vel_b_hidden = np.zeros_like(b_hidden)
    vel_W_hidden2= np.zeros_like(W_hidden2)
    vel_b_hidden2= np.zeros_like(b_hidden2)
    vel_W_out    = np.zeros_like(W_out)
    vel_b_out    = np.zeros_like(b_out)

    return vel_W_in, vel_W_h, vel_b, vel_W_hidden, vel_b_hidden, vel_W_hidden2, vel_b_hidden2, vel_W_out, vel_b_out



def forward_pass_gru(X, W_in, W_h, b, hidden_size):

    batch_size, time_steps, input_size = X.shape

    # I have first initialize the hidden state
    h = np.zeros((batch_size, hidden_size))

    
    # I have defined different lists to store different gate activations and hidden states.
    z_list = []
    r_list = []
    h_tilde_list = []
    h_list = []

    for t in range(time_steps):
        # First, I am taking the input for specific time step t.
        x_t = X[:, t, :]  

        # Here is the combination of update, reset and candidate gates
        concat = np.dot(x_t, W_in) + np.dot(h, W_h) + b

        # Now, split into seperate update, reset, candidate
        z_t = gru_sigmoid(concat[:, :hidden_size])                   
        r_t = gru_sigmoid(concat[:, hidden_size:2*hidden_size])      
        h_tilde_t = tanh(concat[:, 2*hidden_size:3*hidden_size]) 

        # New hidden state is as follows
        h_new = (1 - z_t) * h + z_t * h_tilde_t

        # Now lets store what we have done so far for this time step.
        z_list.append(z_t)
        r_list.append(r_t)
        h_tilde_list.append(h_tilde_t)
        h_list.append(h_new.copy())

        # Finally, update the hidden state
        h = h_new

    cache = {'X': X, 'z_list': z_list, 'r_list': r_list, 'h_tilde_list': h_tilde_list, 'h_list': h_list, 'W_in': W_in, 'W_h': W_h, 'b': b}
    return h, cache


def backward_pass_gru(dh_final, cache, hidden_size):
    # Now to do backprop, lets first extract what we have done in forward pass.
    X = cache['X']
    z_list = cache['z_list']
    r_list = cache['r_list']
    h_tilde_list = cache['h_tilde_list']
    h_list = cache['h_list']
    W_in = cache['W_in']
    W_h = cache['W_h']

    batch_size, time_steps, input_size = X.shape

    # Lets initialize gradients
    dW_in = np.zeros_like(W_in)
    dW_h  = np.zeros_like(W_h)
    db    = np.zeros((1, 3*hidden_size))
    
    dh_next = dh_final  
    

    for t in reversed(range(time_steps)):

        # Here is the gates, cells 
        z_t = z_list[t]
        r_t = r_list[t]
        h_tilde_t = h_tilde_list[t]
        h_t = h_list[t]
        h_prev = h_list[t-1] if t > 0 else np.zeros_like(h_t)


        #Now the gradients
        dh_t = dh_next.copy()

        # Our candidate state
        dh_tilde_t = dh_t * z_t  
        
        dz_t = dh_t * (h_tilde_t - h_prev)
        dh_prev_part = dh_t * (1 - z_t)  
        dtanh_t = dh_tilde_t * gru_tanh_derivative(h_tilde_t)

        # Here is the our gradients
        dz_raw = dz_t * gru_sigmoid_derivative(z_t)

        dr_raw = np.zeros_like(r_t) 
     
        d_tilde_raw = dtanh_t  
        
        # Here, I have combined the gradients
        d_concat = np.hstack([dz_raw, dr_raw, d_tilde_raw])  

        # To accumulate the gradients
        x_t = X[:, t, :]  
        dW_in = dW_in + np.dot(x_t.T, d_concat) 
        dW_h  = dW_h + np.dot(h_prev.T, d_concat)
        db = db + np.sum(d_concat, axis=0, keepdims=True)

        # Grad for previous hidden
        dh_prev = np.dot(d_concat, W_h.T)  
        dh_prev = dh_prev + dh_prev_part

        # Lets pass the gradient to the following time steps.
        dh_next = dh_prev

    return dW_in, dW_h, db

def gru_forward_pass_mlp(h_t, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out):
    # I have defined the forward pass of the multi layer perceptron.
    # It is a standart straightforward implementation.
    # I have used relu for hidden layers and softmax at the output layer.
    z_hidden = np.dot(h_t , W_hidden) + b_hidden  
    h_hidden = gru_relu(z_hidden)

    z_hidden2 = np.dot(h_hidden, W_hidden2) + b_hidden2
    h_hidden2 = gru_relu(z_hidden2)

    logits = np.dot(h_hidden2, W_out) + b_out  
    y_pred = gru_softmax(logits)           

    return y_pred, h_hidden, z_hidden, h_hidden2, z_hidden2, logits


def train_gru(X_train_full, Y_train_full, X_test, Y_test, params):
    # To get the similar results at each trial.
    np.random.seed(42) 

    # Extracting the required parameters.
    lr = params['lr']
    momentum = params['momentum']
    batch_size = params['batch_size']
    epochs = params['epochs']
    hidden_size = params['hidden_size']
    mlp_hidden_size = params['mlp_hidden_size']
    mlp_hidden_size2 = params['mlp_hidden_size2']
    patience = params['patience']

    # I have specified the parameters.
    lr = 0.01
    momentum = 0.85
    batch_size = 32
    epochs = 50
    hidden_size = 128
    mlp_hidden_size = 64
    mlp_hidden_size2 = 32
    patience = 5

    # To split the data set into %90 training and %10 validation.
    total_samples = X_train_full.shape[0]
    val_size = int(0.1 * total_samples)
    train_size = total_samples - val_size

    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train = X_train_full[train_indices]
    Y_train = Y_train_full[train_indices]
    X_val   = X_train_full[val_indices]
    Y_val   = Y_train_full[val_indices]

    input_size = X_train.shape[2]
    output_size = Y_train.shape[1]
    time_steps = X_train.shape[1]

    # Lets initialize the GRU weights and biase using xavier initialization.
    W_in = gru_init_xavier((input_size, 3*hidden_size))
    W_h  = gru_init_xavier((hidden_size, 3*hidden_size))
    b  = gru_init_xavier((1, 3*hidden_size))

    # Lets extract MLP weights anb biases
    W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out = gru_initialize_mlp_weights(hidden_size, mlp_hidden_size, mlp_hidden_size2, output_size)
    

    # Lets extract velocities to use in momentum
    v_W_in, v_W_h, v_b, v_W_hidden, v_b_hidden, v_W_hidden2, v_b_hidden2, v_W_out, v_b_out = gru_init_velocities(W_in, W_h, b, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out)   


    best_cost = 50
    stopping_criteria = 0  # how many epochs with no improvement
    best_params = None  # to store the best weights

    # I have defined different lists to store and plot the related results
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
  
 
    for epoch in range(epochs):
        # I have shuffled the dataset before each epoch
        perm = np.random.permutation(train_size)
        X_train = X_train[perm]
        Y_train = Y_train[perm]

        epoch_loss = 0
        correct_predictions = 0

        # Mini batch implementation
        for starting_index in range(0, train_size, batch_size):
            ending_index = starting_index + batch_size
            X_batch = X_train[starting_index : ending_index]
            Y_batch = Y_train[starting_index : ending_index]
            batch_s = X_batch.shape[0]

            # Now, we can start the forward pass.
            # First GRU and then MLP
            h_t, cache_gru = forward_pass_gru(X_batch, W_in, W_h, b, hidden_size)
            
            Y_pred, h_hidden, z_hidden, h_hidden2, z_hidden2, logits = gru_forward_pass_mlp(
                h_t, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out
            )

            # After forward pass, lets calculate the loss
            loss = gru_cross_entropy(Y_batch, Y_pred)
            epoch_loss += loss
            correct_predictions += np.sum(np.argmax(Y_pred, axis=1) == np.argmax(Y_batch, axis=1))
            
            # After calculating loss, we can do the backprop
            # First MLP
            grad_out = (Y_pred - Y_batch) / batch_s
            grad_W_out = np.dot(h_hidden2.T, grad_out)
            grad_b_out = np.sum(grad_out, axis=0, keepdims=True)

            grad_hid2_mlp = np.dot(grad_out, W_out.T)
            d_z_hidden2 = grad_hid2_mlp * gru_relu_derivative(z_hidden2)
            grad_W_hid2_mlp = np.dot(h_hidden.T, d_z_hidden2)
            grad_b_hid2_mlp = np.sum(d_z_hidden2, axis=0, keepdims=True)

            grad_hid_mlp = np.dot(d_z_hidden2, W_hidden2.T)
            d_z_hidden = grad_hid_mlp * gru_relu_derivative(z_hidden)
            grad_W_hid_mlp = np.dot(h_t.T, d_z_hidden)
            grad_b_hid_mlp = np.sum(d_z_hidden, axis=0, keepdims=True)

            # Then GRU
            d_h_t = np.dot(d_z_hidden, W_hidden.T)
            grad_W_in_gru, grad_W_hid_gru, grad_b_gru = backward_pass_gru(d_h_t, cache_gru, hidden_size)

            # Lets clipped the gradients before update to get better convergence.
            grad_W_in_gru, grad_W_hid_gru, grad_b_gru, grad_W_hid_mlp, grad_b_hid_mlp, grad_W_hid2_mlp, grad_b_hid2_mlp, grad_W_out, grad_b_out = gru_clip_each_gradients(grad_W_in_gru, grad_W_hid_gru, grad_b_gru, grad_W_hid_mlp, grad_b_hid_mlp, grad_W_hid2_mlp, grad_b_hid2_mlp, grad_W_out, grad_b_out)

            # Applying momentum. I have tried the momentum formula having (1-momentum) in it first.
            # However, it was hard to find the best parameters for that and I have returned to the following formula
            v_W_in = (momentum * v_W_in) - (lr * grad_W_in_gru)
            v_W_h = (momentum * v_W_h)  - (lr * grad_W_hid_gru)
            v_b  = (momentum * v_b) - (lr * grad_b_gru)
            v_W_hidden = (momentum * v_W_hidden) - (lr * grad_W_hid_mlp)
            v_b_hidden = (momentum * v_b_hidden) - (lr * grad_b_hid_mlp)
            v_W_hidden2 = (momentum * v_W_hidden2) - (lr * grad_W_hid2_mlp)
            v_b_hidden2 = momentum * v_b_hidden2 - lr * grad_b_hid2_mlp
            v_W_out = (momentum * v_W_out) - (lr * grad_W_out)
            v_b_out = (momentum * v_b_out) - (lr * grad_b_out)

            # I have updated from now on
            W_in = W_in + v_W_in
            W_h = W_h + v_W_h
            b = b + v_b
            W_hidden = W_hidden + v_W_hidden
            b_hidden = b_hidden + v_b_hidden
            W_hidden2 = W_hidden2 + v_W_hidden2
            b_hidden2 = b_hidden2 + v_b_hidden2
            W_out = W_out + v_W_out
            b_out = b_out + v_b_out

        # Calculating average training loss
        train_loss = epoch_loss / (train_size // batch_size)
        train_accuracy = correct_predictions / train_size

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        h_val, _ = forward_pass_gru(X_val, W_in, W_h, b, hidden_size)
        Y_val_pred, *_ = gru_forward_pass_mlp(h_val, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out)

        # Calculating validation loss
        val_loss = gru_cross_entropy(Y_val, Y_val_pred)
        
        # Calculating validation accuracy
        val_accuracy = np.mean(np.argmax(Y_val_pred, axis=1) == np.argmax(Y_val, axis=1))
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f} %")

        # Lets define our early stopping criteria
        if val_loss < best_cost:   
            best_cost = val_loss
            stopping_criteria = 0
            best_params = ( W_in, W_h, b, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out)
        else:
            stopping_criteria += 1
            if stopping_criteria >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}, best validation loss: {best_cost:.4f}")
                break


    # Testing in test set.
    h_test, _ = forward_pass_gru(X_test, W_in, W_h, b, hidden_size)
    Y_pred_test, _, _, _, _, _ = gru_forward_pass_mlp(h_test, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out)

    preds = np.argmax(Y_pred_test, axis=1)
    labels = np.argmax(Y_test, axis=1)
    test_acc = np.mean(preds == labels) * 100
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Compute confusion matrix
    num_classes = Y_test.shape[1]
    con_mat = gru_compute_confusion_matrix(labels, preds, num_classes)
    print("Confusion Matrix:\n", con_mat)

    return (train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
        test_acc,
        preds,
        labels,
        best_params[0],  # W_in
        best_params[1],  # W_h
        best_params[2],  # b
        best_params[3],  # W_hidden
        best_params[4],  # b_hidden
        best_params[5],  # W_hidden2
        best_params[6],  # b_hidden2
        best_params[7],  # W_out
        best_params[8])  # b_out


def q3():

    with h5py.File('data3.h5', 'r') as file:
        
        X_data = np.array(file['trX'])   
        Y_data = np.array(file['trY'])  
        X_test = np.array(file['tstX'])         
        Y_test = np.array(file['tstY'])        

    # To get similar results.
    np.random.seed(42)  
    indices = np.random.permutation(X_data.shape[0])
    X_data = X_data[indices]
    Y_data = Y_data[indices]

    # Splitting data set into validation and training
    val_size = int(0.1 * X_data.shape[0])  
    X_val = X_data[:val_size]              
    Y_val = Y_data[:val_size]
    X_train = X_data[val_size:]           
    Y_train = Y_data[val_size:]


    params = {
        'hidden_size': 128,  # RNN hidden layer size
        'mlp_hidden': 64,    # MLP hidden layer size
        'lr': 0.001,         # learning rate
        'batch_size': 32,
        'epochs': 50,
        'patience': 7,       # early stopping patience
        'momentum': 0.85,    # you can set 0 if you don't want momentum
    }

    train_losses, train_accs, val_losses, val_accs, test_acc = train_rnn(
        X_train, Y_train, 
        X_val,   Y_val,
        X_test,  Y_test,
        hidden_size=params['hidden_size'],
        mlp_hidden=params['mlp_hidden'],
        lr=params['lr'],
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        patience=params['patience'],
        momentum=params['momentum']
    )


    epochs_ran = len(train_losses)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_ran+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs_ran+1), val_losses,   label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_ran+1), train_accs, label='Train Accuracy')
    plt.plot(range(1, epochs_ran+1), val_accs,   label='Valiation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

    with h5py.File('data3.h5', 'r') as f:
        train_X = np.array(f['trX'])
        train_Y = np.array(f['trY'])
        test_X = np.array(f['tstX'])
        test_Y = np.array(f['tstY'])

    # Hyperparameters
    params = {
        'lr': 0.1,
        'momentum': 0.85,
        'batch_size': 32,
        'epochs': 50,
        'hidden_size': 128,
        'mlp_hidden_size': 64,
        'mlp_hidden_size2': 32,
        'patience': 5
    }

    # I have trained the LSTM
    (train_losses, train_accuracies, val_losses, val_accuracies, test_accuracy, 
    predictions, true_labels, W_in, W_h, b, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out) = train_lstm(
        train_X, train_Y, test_X, test_Y, params
    )


    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # For accuracies
    plt.figure(figsize=(12, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    import seaborn as sns

    # I am doing confusion matrix for the trainset
    num_classes = train_Y.shape[1]
    train_preds = []
    for start_idx in range(0, train_X.shape[0], params['batch_size']):
        end_idx = start_idx + params['batch_size']
        X_batch = train_X[start_idx:end_idx]
        h_train, _ = forward_pass_lstm(X_batch, W_in, W_h, b, params['hidden_size'])
        train_Y_pred, *_ = lstm_forward_pass_mlp(h_train, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out)
        train_preds.append(np.argmax(train_Y_pred, axis=1))

    train_preds = np.concatenate(train_preds)
    train_labels = np.argmax(train_Y, axis=1)
    train_conf_matrix = compute_confusion_matrix(train_labels, train_preds, num_classes)


    plt.figure(figsize=(10, 8))
    sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Training Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # I am doing confusion matrix of the test data
    conf_matrix = compute_confusion_matrix(true_labels, predictions, num_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


    with h5py.File('data3.h5', 'r') as f:
        train_X = np.array(f['trX'])
        train_Y = np.array(f['trY'])
        test_X = np.array(f['tstX'])
        test_Y = np.array(f['tstY'])

    # Hyperparameters
    params = {
        'lr': 0.01,
        'momentum': 0.85,
        'batch_size': 32,
        'epochs': 50,
        'hidden_size': 128,
        'mlp_hidden_size': 64,
        'mlp_hidden_size2': 32,
        'patience': 5
    }

    # Train the GRU model
    (train_losses, 
    train_accuracies,
    val_losses, 
    val_accuracies, 
    test_accuracy, 
    predictions, 
    true_labels,
    W_in, 
    W_h, 
    b,
    W_hidden, 
    b_hidden, 
    W_hidden2, 
    b_hidden2, 
    W_out, 
    b_out) = train_gru(train_X, train_Y, test_X, test_Y, params)


    # Plot Training and Validation Loss
    plt.figure(figsize=(8, 8))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Training and Validation Accuracy
    plt.figure(figsize=(8, 8))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Print final test accuracy
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    import seaborn as sns

    # Confusion matrix part
    num_classes = train_Y.shape[1]  
    train_preds = []
    for start_idx in range(0, train_X.shape[0], params['batch_size']):
        end_idx = start_idx + params['batch_size']
        X_batch = train_X[start_idx:end_idx]
        h_train, _ = forward_pass_gru(X_batch, W_in, W_h, b, params['hidden_size'])
        # Not to get irrelevant ones.
        train_Y_pred, *_ = gru_forward_pass_mlp(h_train, W_hidden, b_hidden, W_hidden2, b_hidden2, W_out, b_out)
        train_preds.append(np.argmax(train_Y_pred, axis=1))

    train_preds = np.concatenate(train_preds)
    train_labels = np.argmax(train_Y, axis=1)
    train_conf_matrix = gru_compute_confusion_matrix(train_labels, train_preds, num_classes)



    plt.figure(figsize=(10, 8))
    sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Training Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # I have computed confusion matrix for test set
    num_classes = train_Y.shape[1]
    conf_matrix = gru_compute_confusion_matrix(true_labels, predictions, num_classes)


    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def berkay_altintas_22002709_hw1(question):
    if question == '1' :
        q1()
	
    elif question == '2' :
        q2()
        
    elif question == '3' :
        q3()
        



berkay_altintas_22002709_hw1(question)



