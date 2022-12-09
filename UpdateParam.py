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
    data_url="http://lib.stat.cmu.edu/datasets/boston"
    raw_df=pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    input_array1=np.hstack([raw_df.values[::2, :],
                   raw_df.values[1::2, :2]])
    target_array = raw_df.values[1::2, 2] #제일 마지막 데이터=부동산 가격
    feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                          'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT'])
    input_array=input_array1[:,feature_names=='RM']
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
    epoch = 1

    #print(number_of_rooms)
    print(prices)
    #print(len(number_of_rooms))

    #weight, bias, loss = linear_regression_1D(number_of_rooms, prices, epoch)

    #
    #plt.plot(number_of_rooms, prices, ".")
    #plt.plot(number_of_rooms, predict(number_of_rooms, weight, bias), ".", label=f"loss: {loss}")
    #plt.savefig("test.png")

    #print(f"weight, bias, loss: {weight}, {bias}, {loss}")