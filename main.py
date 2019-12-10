from random import uniform
import numpy as np
from math import sqrt
from os import path
import csv
import pandas as pd

class KohonenModel:
    WeightVectors = set()

    def __init__(self, vectors=2, vectorLength=1):
        if 1 >= vectors:
            raise Exception("There has to be at lest 2 weight vectors")
        if 0 >= vectorLength:
            raise Exception("Minimum length of weight vector is 1")
        self.WeightVectors = np.array(
            [WeightVector(i, vectorLength) for i in range(vectors)]
        )

    def train(self, teaching_examples):
        for example in teaching_examples:
            distances = pd.DataFrame(
                [vector.calculate_distance(example) for vector in self.WeightVectors],
                columns=["difference", "length", "vector"]
            )
            distances.sort_values(by="length", inplace=True)
            distances.reset_index(inplace=True, drop=True)
            winner = distances.loc[0]
            winner["vector"].adjust_weights(winner["difference"])
            for distance in distances.loc[1:4].as_matrix():
                difference, length, neighbor = distance
                neighbor.adjust_weights(
                    difference,
                    winner["vector"].neighbor_factor(neighbor.Weights),
                )

    def recognize(self, example):
        distances = pd.DataFrame(
                [vector.calculate_distance(example) for vector in self.WeightVectors],
                columns=["difference", "length", "vector"]
            )
        distances.sort_values(inplace=True, by="length")
        distances.reset_index(inplace=True, drop=True)
        return distances.loc[0, "vector"].index


class WeightVector:
    Weights = set()
    learning_restraint_fraction_length = 5
    previous_learning_restraint = 0.95
    learning_restraint = 0.95

    def __init__(self, index, length=1):
        if 0 >= length:
            raise Exception("Minimum length of weight vector is 1")
        self.Weights = np.array([uniform(0, 255) for i in range(length)])
        self.index = index

    def calculate_distance(self, point):
        if point.size != self.Weights.size:
            raise Exception(
                "Cannot calculate distance between weight vector and point. \nPoint has different length that vector"
            )
        difference = point - self.Weights
        length = sqrt(np.square((difference)).sum())
        return pd.Series([difference, length, self], index=["difference", "length", "vector"])

    def adjust_weights(self, difference, neighbor_factor=1):
        self.Weights = self.Weights + (
            difference * self.learning_restraint * neighbor_factor
        )
        temp_learning_restraing = self.learning_restraint
        self.learning_restraint = (
            self.learning_restraint * self.previous_learning_restraint
        )
        self.previous_learning_restraint = temp_learning_restraing

    def neighbor_factor(self, neighbor_vector):
        distance = self.calculate_distance(neighbor_vector)["length"]
        return self.learning_restraint - (self.learning_restraint / max(distance, 1.2))

train = np.array(list(csv.reader(open("mnist_train.csv"), delimiter=',')), dtype="uint8")
train = train[:, 1:]
classifier = KohonenModel(10, 784)
classifier.train(train)

test = np.array(list(csv.reader(open("mnist_test.csv"), delimiter=',')), dtype="uint8")
result = pd.DataFrame([], columns=["number", "index"])
for example in test:
    result.append([example[0], classifier.recognize(example[1:])])
print(result)
