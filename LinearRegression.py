import numpy as np
import matplotlib.pyplot as plt
import random as rd

print("Step1: Development Linear Regression Algorithm")

# input_array, target_array will be 1-d array(1입력1출력 선형회귀)
#paramaters = weight, bias will be scalar
#input_array=입력값=방개수, target_array=결과값=부동산가격, predict_array=출력값=입력값 넣어서 계산한

def get_data():
    with open("BostonData1.txt", "r") as f: #처리할 txt 파일을 읽기 모드로 읽음
        lines = f.readlines() #각 라인들을 "lines"라는 리스트에 넣음. 각 줄이 리스트 멤버가 된 것 
        for line in lines: # "lines"의 멤버들을 "line"이라는 변수에 넣어서 반복 계산을 하겠다
            input_array=line.strip()[0:5]
            target_array=line.strip()[5:10]

    return input_array, target_array


def predict(input_array, weight, bias):
    predict_array=weight*input_array+bias

    return predict_array


def lossfunction(differnce):
    target_array=target_array
    predict_array=predict_array
    diff=target_array - predict_array
    loss=(1/len(target_array))*np.sum((diff)^2)

    return loss


def update_weight_bias(loss):
    #SGD.step function 대체

    return weight, bias


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



if __name__ == "__main__": 
    number_of_rooms, prices = get_data()
    weight, bias = linear_regression_1D(number_of_rooms, prices)

    plt.plot(number_of_rooms, prices, ".")
    plt.plot(number_of_rooms, predict(number_of_rooms, weight, bias), ".")
    plt.show()




