"""
Simple PyTorch classifier for MNIST, demonstrating 
GOOD comment practice.
"""

import torch
import torch.nn as nn
import numpy as np
import os

# we use a manual seed for reproducibility
SEED = 0
torch.manual_seed(SEED)

def load_mnist():
    data_dir = 'mnist'
    train_prefix = 'mnist-train'
    test_prefix = 'mnist-test'
    image_suffix = '-images.dat'
    label_suffix = '-labels.dat'

    # the starts of the files contain some metadata headers that we can skip
    IMAGE_HEADER_LENGTH = 16
    LABEL_HEADER_LENGTH = 8

    # the data was saved as a uint8 so we need to match the type to load it
    ENCODING_DTYPE = np.uint8

    X_train = load_from_file(data_dir, train_prefix, image_suffix, ENCODING_DTYPE, IMAGE_HEADER_LENGTH)
    y_train = load_from_file(data_dir, train_prefix, label_suffix, ENCODING_DTYPE, LABEL_HEADER_LENGTH)

    X_test = load_from_file(data_dir, test_prefix, image_suffix, ENCODING_DTYPE, IMAGE_HEADER_LENGTH)
    y_test = load_from_file(data_dir, test_prefix, label_suffix, ENCODING_DTYPE, LABEL_HEADER_LENGTH)

    X_train = reshape_into_mnist_images(X_train)
    X_test = reshape_into_mnist_images(X_test)
    # because the data was imported as np.uint8 in numpy arrays
    # and we are using a pytorch model, we have to reformat the data
    # before passing it through the model
    X_train, X_test = format_images(X_train), format_images(X_test)
    y_train, y_test = format_labels(y_train), format_labels(y_test)

    return X_train, y_train, X_test, y_test


def load_from_file(path_to_directory, prefix, suffix, encoding_type, offset):
    file_name = prefix + suffix
    full_path = os.path.join(path_to_directory, file_name)
    data = np.fromfile(full_path, dtype=encoding_type, offset=offset)
    return data


def reshape_into_mnist_images(training_data):
    mnist_image_width = 28
    mnist_image_height = 28
    mnist_image_size = mnist_image_width * mnist_image_height
    return training_data.reshape(-1, mnist_image_size)


def format_images(input_images):
    reformatted_input_images = torch.tensor(input_images, dtype=torch.float)
    return reformatted_input_images

def format_labels(input_labels):
    reformatted_input_labels = torch.tensor(input_labels, dtype=torch.long)
    return reformatted_input_labels


# TODO: add convolutional network option
def build_feedforward_neural_network(input_size, output_size, hidden_size):
    model = nn.Sequential(
                    nn.Linear(input_size, hidden_size), 
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size)
                    )
    return model


def sample_minibatch_of_image_and_labels(images, labels, minibatch_size):
    # pytorch has no 'choice' function so we generate a random permutation
    # of the indices and pick a subset of those to use
    random_permutation = torch.randperm(len(images))
    indices_of_minibatch = random_permutation[:minibatch_size]
    return images[indices_of_minibatch], labels[indices_of_minibatch]


def train_for_one_epoch(model, optimiser, training_images, training_labels, minibatch_size):
    X_train_minibatch, y_train_minibatch = sample_minibatch_of_image_and_labels(training_images, training_labels, minibatch_size)

    # the cross-entropy loss from PyTorch expects logits (unnormalised log probs) rather than probabilities
    logit_predictions = model(X_train_minibatch)
    loss = nn.functional.cross_entropy(logit_predictions, y_train_minibatch)

    # It is very important here to use the optimiser to zero the parameter gradients.
    # If we do not, then they accumulate between epochs and we make the wrong updates.
    optimiser.zero_grad()

    # Compute gradients from the loss
    # in the parameters using backpropagation
    loss.backward()

    optimiser.step()
    return loss


def compute_test_accuracy(model, test_images, test_labels):
    logit_predictions = model(test_images)

    class_probs = nn.functional.softmax(logit_predictions, dim=1)
    class_with_best_prob = torch.argmax(class_probs, dim=1)
    accuracy = compute_accuracy(class_with_best_prob, test_labels)

    return accuracy


def compute_accuracy(predicted_labels, target_labels):
    predictions_and_targets = zip(predicted_labels, target_labels)
    boolean_list_of_prediction_success = [prediction == target for prediction, target in predictions_and_targets]

    # we can sum because True is automatically cast to 1 and False to 0
    number_of_correct_predictions = sum(boolean_list_of_prediction_success)
    total_number_of_predictions = len(predicted_labels)

    accuracy = number_of_correct_predictions/total_number_of_predictions
    return accuracy


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    mnist_image_width = mnist_image_height = 28
    mnist_image_size = mnist_image_width * mnist_image_height
    hidden_size = 256
    number_of_classes = 10
    model = build_feedforward_neural_network(mnist_image_size, number_of_classes, hidden_size)

    learning_rate = 1e-3
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_training_epochs = 1000
    # Don't set larger than ~1e6 or you'll run out of memory
    minibatch_size = 32

    for current_epoch in range(num_training_epochs):
        loss = train_for_one_epoch(model, optimiser, X_train, y_train, minibatch_size)

    print(f"Test accuracy is: {compute_test_accuracy(model, X_test, y_test):0.4f}")

if __name__ == "__main__":
    main()
