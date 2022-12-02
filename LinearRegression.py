import numpy as np
import matplotlib.pyplot as plt
import random as rd

print("Step1: Development Linear Regression Algorithm")

# input: number of rooms, output: price
# So, input_array, target_array will be 1-d array
def linear_regression_1D(input_array, target_array, epoch):
    weight, bias = rd.random() #intialize weight & bias (0 ~ 1)
    
    for i in range(0, epoch):
        ### have to make predict function(look at the page below)
        predict_array = predict(input_array, weight, bias)
        diff = target_array - predict_array
        ### have to make loss function(look at the page below)
        loss = lossfunction(diff)
        ### have to make update function(look at the page below)
        weight, bias = update_weight_bias(loss)
    return weight, bias


def predict(input_array):
    return predict_array


def lossfunction(differnce):
    return loss


def update_weight_bias(loss):
    return weight, bias

