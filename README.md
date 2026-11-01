# Handwriting-Classification-with-EMNIST
This is a Python app that recognizes handwritten letters (A-Z). It uses the EMNIST Letters dataset to train a neural network and opens a GUI where you can sketch a letter and get a prediction. It trains for 40 epochs with the Adam optimizer and reports accuracy on the test set.

# Process: 
1. loads EMNIST letters dataset (thousands of handwritten letters from A-Z)
2. trains simple neural network on those letters (3 hidden layers with 128 neurons each that each use ReLU activation)
3. after training, it tests itself on letters it hasn't seen and prints the accuracy
4. opens window where you can draw letter (using Tinkter) 

# Notes: 
1. takes a long time to finish training (retrains every time you run it)
