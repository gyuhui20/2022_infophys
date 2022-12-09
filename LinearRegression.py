import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd #잠깐 넣어둠

print("Step1: Development Linear Regression Algorithm")

# input_array, target_array will be 1-d array(1입력1출력 선형회귀)
#paramaters = weight, bias will be scalar
#input_array=입력값=방개수, target_array=결과값=부동산가격, predict_array=출력값=입력값 넣어서 계산한
f linear_regression_1D(input_array, target_array, epoch):
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
    predict_array=weight*(input_array)+bias
    return predict_array


def lossfunction(differnce):
    loss=(1/len(target_array))*np.sum((diff)^2)
    return loss


def update_weight_bias(loss):
    #SGD.step function 대체
    return weight, bias




#여긴 수정 필요
if __name__ == "__main__": 
    number_of_rooms, prices = get_data()
    weight, bias = linear_regression_1D(number_of_rooms, prices)

    plt.plot(number_of_rooms, prices, ".")
    plt.plot(number_of_rooms, predict(number_of_rooms, weight, bias), ".")
    plt.show()




