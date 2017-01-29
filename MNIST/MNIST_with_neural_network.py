# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:03:30 2017

@author: yashwanth
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

total_classes = 10
total_nodes_hl1 = 500
total_nodes_hl2 = 500
total_nodes_hl3 = 500

batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_layer_1 = {'weights' :tf.Variable(tf.random_normal([
        784, total_nodes_hl1])),
                     'biases' :tf.Variable(tf.random_normal([
        total_nodes_hl1]))
    }
    
    hidden_layer_2 = {'weights' :tf.Variable(tf.random_normal([
        total_nodes_hl1, total_nodes_hl2])),
                     'biases' :tf.Variable(tf.random_normal([
        total_nodes_hl2]))
    }
    
    hidden_layer_3 = {'weights' :tf.Variable(tf.random_normal([
        total_nodes_hl2, total_nodes_hl3])),
                     'biases' :tf.Variable(tf.random_normal([
        total_nodes_hl3]))
    }
    
    output_layer = {'weights' :tf.Variable(tf.random_normal([
        total_nodes_hl3, total_classes])),
                     'biases' :tf.Variable(tf.random_normal([
        total_classes]))
    }
    
    layer1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    layer1 = tf.nn.relu(layer1)    
    
    layer2 = tf.add(tf.matmul(layer1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer2 = tf.nn.relu(layer2)
    
    layer3 = tf.add(tf.matmul(layer2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer3 = tf.nn.relu(layer3)
    
    output = tf.matmul(layer3, output_layer['weights'] + output_layer['biases'])
    
    return output


def training_neural_net(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost) #learining rate
    # learning rate 1-e
    #cycles for feedforward + backprop
    total_epochs = 15
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(total_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, l = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += l
            print("Epoch", epoch, "completed out of", total_epochs, "loss:", epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
training_neural_net(x)