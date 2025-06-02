import json
import nltk
import random
import pickle
import tflearn
import numpy as np
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
import matplotlib.pyplot as plt  # For visualizing loss/accuracy

# Initialize Lancaster Stemmer
stemmer = LancasterStemmer()

# Load dataset
with open('dataset/dataset.json') as f:
    data = json.load(f)

try:
    with open('data.pickle', 'rb') as file:
        words, labels, train, output = pickle.load(file)

except:
    words = []
    x_docs = []  # Patterns - Sentences
    y_docs = []  # Tags for patterns
    labels = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            tokenizedWords = nltk.word_tokenize(pattern)
            words.extend(tokenizedWords)
            x_docs.append(tokenizedWords)
            y_docs.append(intent['tag'])
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    # Sorting labels
    labels = sorted(labels)

    # Stemming words and sorting
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))

    train = []
    output = []

    # Creating a Bag of Words - One Hot Encoding
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(x_docs):
        bag = []
        stemmedWords = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in stemmedWords:
                bag.append(1)
            else:
                bag.append(0)

        outputRow = out_empty[:]
        outputRow[labels.index(y_docs[x])] = 1

        train.append(bag)
        output.append(outputRow)

    # Converting data into NumPy array
    train = np.array(train)
    output = np.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, train, output), f)

# Build the model
net = tflearn.input_data(shape=[None, len(train[0])])  # Input layer
net = tflearn.fully_connected(net, 8)  # First hidden layer
net = tflearn.fully_connected(net, 8)  # Second hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')  # Output layer
net = tflearn.regression(net)  # Regression layer

model = tflearn.DNN(net)

print('[INFO] Training Model...')

# Custom callback to log loss and accuracy
from tflearn.callbacks import Callback

step_metrics = {'loss': [], 'accuracy': []}


class LogMetricsCallback(Callback):
    def on_epoch_end(self, training_state):
        # Append loss and accuracy for each epoch
        step_metrics['loss'].append(training_state.global_loss)
        step_metrics['accuracy'].append(training_state.acc_value)
        print(f"[Epoch: {training_state.epoch + 1}] Loss: {training_state.global_loss:.4f}, "
              f"Accuracy: {training_state.acc_value:.4f}")


# Create an instance of the callback
log_metrics_callback = LogMetricsCallback()

# Train the model
model.fit(train, output,
          n_epoch=400,
          batch_size=8,
          show_metric=True,  # Show loss & accuracy while training
          snapshot_step=100,  # Save logs every 100 steps
          run_id='chatbot_training',  # TensorBoard logs ID
          callbacks=[log_metrics_callback])  # Use the custom callback

# Save model weights
model.save('models/chatbot-model.tflearn')
print('[INFO] Model successfully trained and saved!')

# Visualize loss and accuracy
plt.figure(figsize=(10, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(step_metrics['loss'], label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(step_metrics['accuracy'], label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


# Function for Bag of Words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


# Chat function
def chat():
    print('[INFO] Start talking...(type "quit" to exit)')
    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break

        predict = model.predict([bag_of_words(inp, words)])
        predictions = np.argmax(predict)
        tag = labels[predictions]

        for t in data['intents']:
            if t['tag'] == tag:
                responses = t['responses']

        outputText = random.choice(responses)
        print(outputText)


chat()
