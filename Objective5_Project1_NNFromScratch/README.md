# Objective 5 – Project 1: Neural Network From Scratch (NumPy)

## Why this meets Objective 5
This project demonstrates a deep understanding of neural networks by implementing forward propagation, backpropagation, gradients, and weight updates without using ML frameworks. XOR shows why hidden layers enable non-linear decision-making in intelligent systems.

This project meets Objective 5 by demonstrating a deep understanding of how neural networks are designed and how architectural choices affect intelligent behavior. By building and comparing a shallow neural network, a deeper multi-layer network, and a regularized deep network with dropout, I was able to clearly analyze how depth increases learning capacity while also increasing the risk of overfitting. The layer-by-layer summaries show how data flows through each network, how parameter counts grow as layers are added, and why activation functions and output layers are chosen for specific tasks. This project focuses on explaining how neural networks learn, generalize, and make predictions, rather than just training a model, which directly demonstrates an understanding of the role neural networks play in creating intelligent systems.

## Run
```bash
pip install -r requirements.txt
cd src
python nn_from_scratch.py

## Understanding the Neural Network Tables

## The tables above show the internal structure of two neural networks: a deep network and the same network with regularization added. Each row represents a step in how information flows through the system, starting from the input features and ending with a final prediction. The “Output Shape” column shows how the data is transformed at each stage, while the “Parameter” count shows how much the network can learn at that point.

## As the data moves through the hidden layers, the network first expands and then refines patterns, allowing it to learn increasingly complex relationships. The final output layer converts those learned patterns into a probability that can be used for decision-making. In the regularized model, dropout layers are added to randomly disable neurons during training, which helps prevent overfitting without increasing the total number of parameters. This comparison demonstrates how neural network intelligence is shaped not just by size, but by how the architecture is designed to learn and generalize.