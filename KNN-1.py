import random
from cmath import sqrt
from collections import Counter
from statistics import mode

import numpy as np
import pandas as pd
import operator
# from natsort import index_natsorted, order_by_index



# Calculate the Euclidean distance between two vectors
# q, p: are lists
def euclidean_distance(q, p):
    p = np.array(p)
    q = np.array(q)
    distance = 0.0
    for i in range(len(p[0])-1):
        distance += (q[0][i] - p[0][i])**2
    return sqrt(distance).real


# def sort_according_to_class_lbl(data):
#     return data.reindex(index=order_by_index(data.index, index_natsorted(data['class_lbl'])))


# Get the most nearest
def get_neighbors(train_data, test_row, k):
    distances = []
    for i in range(len(train_data)):
        train_row = train_data[i:i+1]
        train_row = np.array(train_row)
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    # Sort according to the second parameter which is the distance
    distances.sort(key=lambda tupl: tupl[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def get_response(neighbors):
    class_lbls = {}
    for i in range(len(neighbors)):
        response = neighbors[i][0][-1]
        if response in class_lbls:
            class_lbls[response] += 1
        else:
            class_lbls[response] = 1
    sorted_class_lbls = sorted(class_lbls.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_lbls


def predict_classification(train_data, test_row, k):
    neighbors = get_neighbors(train_data, test_row, k)
    # output_values: is the distances of each neighbor
    output_values = [row[-1] for row in neighbors]
    class_lbls = [row[-1] for row in output_values]
    # Get the most repeated value.
    prediction = get_most_rep_class_lbl(class_lbls)

    # print(mode(class_lbls))
    return prediction


def get_most_rep_class_lbl(class_lbls):
    # get a dict, key is the elements in class_lbls, value is count of the element
    d_mem_count = Counter(class_lbls)
    maxi = max(d_mem_count, key=d_mem_count.get)
    return maxi


def predict_all_tests(train_data, test_data, k):
    predictions = []
    for i in range(len(test_data)):
        predic = predict_classification(train_data, test_data[i:i+1], k)
        actual = test_data[i:i+1]['class_lbl'].to_string()[-3:]
        print("Predicted class: %s" %predic + ", Actual class: %s" %actual)
        predictions.append(predic)
    return predictions


def cal_accuracy(test_data, predictions):
    right = 0.0
    for i in range(len(test_data)):
        actual = test_data[i:i+1]['class_lbl'].to_string()[-3:]
        # print(actual, predictions[i])
        if actual == predictions[i]:
            right += 1.0
    print("Number of correctly classified instances : %d" %right + ", Total number of instances : %d" %len(test_data))
    accuracy = right/len(test_data)
    return accuracy


names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'class_lbl']

test_data = pd.read_csv('TestData.txt', header=None, names=names)
train_data = pd.read_csv('TrainData.txt', header=None, names=names)
# print(euclidean_distance(train_data[0:1], train_data[1:2]))
k = 3
print("k value = %d" %k)
neighbors = get_neighbors(train_data, test_data[0:1], k)
predictions = predict_all_tests(train_data, test_data, k)
accuracy = cal_accuracy(test_data, predictions)
print("Accuracy : %f" %accuracy)
