import numpy as np
import matplotlib.pyplot as plt

print("Step1: Development Single Linear Regression Algorithm")

#input_array, target_array will be 1-d array
#paramaters = weight, bias will be scalar
#input_array=RM, target_array=MEDV, predict_array=예측한 출력값
def linear_regression_1D(input_array, target_array, epoch):
    weight, bias = 0, 0 #intialize weight & bias = 0
    loss = 0 #initialize loss =0
    lr = 1e-4 #learning rate
    print(f"initial weight, bias, loss: {weight}, {bias}, {loss}")

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

    #record history
    
    
    print(f"final weight, bias, loss: {weight:.4f}, {bias:.4f}, {loss:.4f}")
    return weight, bias, loss

### pulling data function
def get_data(): 
    number_of_rooms_list, price_list = [], []

    with open("BostonData1.txt", "r") as f: #read text file as 'read mode'
             for lines in f: 
                a = lines[0:5]
                b = lines[5:10]
                number_of_rooms_list.append(float(a.replace(" ", "")))
                price_list.append(float(b.replace(" ", ""))) #still 'list'

    input_array = np.array(number_of_rooms_list) #행렬 형태로 변형
    target_array = np.array(price_list)
    return input_array, target_array


def predict(input_array, weight, bias):
    predict_array = weight*(input_array) + bias
    return predict_array


def lossfunction(diff_array):
    loss = np.sum((diff_array)**2)/len(diff_array)
    return loss


#
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

#loss graph

#accuracy graph 

