import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd #잠깐 넣어둠

print("Step1: Development Linear Regression Algorithm")

# input_array, target_array will be 1-d array(1입력1출력 선형회귀)
#paramaters = weight, bias will be scalar
#input_array=입력값=방개수, target_array=결과값=부동산가격, predict_array=출력값=입력값 넣어서 계산한
def linear_regression_1D(input_array, target_array, epoch):
    weight, bias = 0, 0 #intialize weight & bias = 0
    loss = 0 #initialize loss =0
    lr = 1e-4 #learning rate
    print(f"Initial weight, bias, loss: {weight}, {bias}, {loss}")

    for i in range(0, epoch):
        #print(input_array)
        predict_array = predict(input_array, weight, bias)
        #print(predict_array)
        diff_array = predict_array - target_array
        #print(diff_array)
        gradient_weight = np.sum(2*input_array*diff_array) / len(diff_array)
        #print(gradient_weight)
        gradient_bias = np.sum(2*diff_array) / len(diff_array)
        #print(gradient_bias)
        weight -= (lr * gradient_weight)
        bias -= (lr * gradient_bias)
        loss = lossfunction(diff_array)

    
    print(f"Final weight, bias, loss: {weight}, {bias}, {loss}")
    return weight, bias, loss

### pulling data function
def get_data(): #use 'pandas' for just now
    room_number_list, price_list = [], []

    with open("BostonData1.txt", "r") as f:
             for lines in f:
                a = lines[0:5]
                b = lines[5:10]
                room_number_list.append(float(a.replace(" ", "")))
                price_list.append(float(b.replace(" ", "")))

    input_array = np.array(room_number_list)
    target_array = np.array(price_list)
    return input_array, target_array
# 
def predict(input_array, weight, bias):
    predict_array = weight*input_array + bias
    return predict_array


def lossfunction(diff_array):
    loss = np.sum((diff_array)**2)/len(diff_array)
    return loss





#여긴 수정 필요
if __name__ == "__main__": 
    number_of_rooms, prices = get_data()
    epoch = 3000

    #print(number_of_rooms)
    print(prices)
    #print(len(number_of_rooms))

    weight, bias, loss = linear_regression_1D(number_of_rooms, prices, epoch)

    #
    plt.plot(number_of_rooms, prices, ".", label="target")
    plt.plot(number_of_rooms, predict(number_of_rooms, weight, bias), ".", label="predict")
    plt.savefig("test.png")

    #print(f"weight, bias, loss: {weight}, {bias}, {loss}")