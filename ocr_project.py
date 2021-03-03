import numpy as np
import string
import csv
import neurolab as nl



# convert data in letter.data to a stored numpy array
# containing only useful data
def string_vectorizer(strng, alphabet=string.ascii_lowercase):
    vector = [[0 if char != letter else 1 for char in alphabet] for letter in strng]
    return vector[0]

def extract_data(x):
    return {"label": string_vectorizer(x[1]), "pixels": x[6:134]}

pixels = []
labels = []

# Define the input file containing the OCR data:
letter_file = 'C:/Users/DELL/Documents/my-folder/letter.data'

with open(letter_file) as file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        data = extract_data(row)
        pixels.append(data["pixels"])
        labels.append(data["label"])

np.savez("letters", pixels=pixels, labels=labels)

def print_letter(letter):
    for i in range(0, 16):
        for j in range(0, 8):
            print("||" if (letter[(8 * i + j)] == "1") else "  ", end=""),
        print()

def print_data(label, pixel):
    print("_____________________________________________________")
    print("label: ",label)
    print("image: ")
    print_letter(pixel)

data = np.load("letters.npz")

pixels = data["pixels"]
labels = data["labels"]

for i in range(len(pixels)):
    print_data(labels[i], pixels[i])



# Define the number of datapoints to 
# be loaded from the input file
num_datapoints = 50
# String containing all the distinct characters
orig_labels = 'omandig'

# Compute the number of distinct characters
num_orig_labels = len(orig_labels)

# Defining the training and testing parameters and storing it as an integer
num_train = int(0.7 * num_datapoints)
num_test = num_datapoints - num_train

# Defining the dataset extraction parameters
start = 6
end = -1 

# Creating the dataset
data = []
labels = []
with open(letter_file, 'r') as f:
    for line in f.readlines():
        # Splitting the current line tabwise
        list_vals = line.split('\t')
        
        # skipping label if it is not in the list of lables
        if list_vals[1] not in orig_labels:
            continue
        
# Extracting the current label and appendding it to the main list
        label = np.zeros((num_orig_labels, 1))
        label[orig_labels.index(list_vals[1])] = 1
        labels.append(label)

# Extracting the character vector and appending it to the data list
        cur_char = np.array([float(x) for x in list_vals[start:end]])
        data.append(cur_char)
        if len(data) >= num_datapoints:
            break

# Converting the data and labels to numpy arrays
data = np.asfarray(data)
labels = np.array(labels).reshape(num_datapoints, num_orig_labels)

# Extracting the number of dimensions
num_dims = len(data[0])

# Creating a feedforward neural network
nn = nl.net.newff([[0, 1] for _ in range(len(data[0]))],
        [128, 16, num_orig_labels])

# Setting the training algorithm to gradient descent
nn.trainf = nl.train.train_gd

# Training the network
error_progress = nn.train(data[:num_train,:], labels[:num_train,:],
        epochs=9000, show=100, goal=0.01)

# Predicting the output for test input
print('\nTesting on unknown data:')
predicted_test = nn.sim(data[num_train:, :])
for i in range(num_test):
    print('\nOriginal:', orig_labels[np.argmax(labels[i])])
    print('Predicted:', orig_labels[np.argmax(predicted_test[i])])
