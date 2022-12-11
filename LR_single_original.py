import numpy as np
import matplotlib.pyplot as plt

print("Step1: Development Single Linear Regression Algorithm")
#input_array, target_array will be 1-d array
#paramaters = weight, bias will be scalar
#input_array=RM, target_array=MEDV, predict_array=예측한 출력값

def linear_regression_1D(input_array, target_array, epoch):
    weight, bias = 1,1
    loss_list = []
    lr = 1e-4
    print(f"initial : weight, bias : {weight}, {bias}")
    
    for i in range(0, epoch):
        #predict variable
        predict_array = predict(input_array, weight, bias)
        #difference function(편차 함수)
        diff_array = predict_array - target_array
        #calculate gradients(경사 계산)
        gradient_weight = np.sum(2*input_array*diff_array) / len(diff_array) 
        gradient_bias = np.sum(2*diff_array) / len(diff_array)
        #adjust parameters(weight, bias)(파라미터 수정)
        weight -= (lr * gradient_weight)
        bias -= (lr * gradient_bias)
        #loss variable
        loss = lossfunction(diff_array)
        loss_list.append(loss)
    loss_arr = np.array(loss_list)

    print(f"final : weight, bias, loss: {weight:.4f}, {bias:.4f}, {loss:.4f}")
    #accuracy(R^2 결정계수)
    target_mean=target_array.mean()
    acc=1-((target_array-predict_array)**2).sum()/((target_array-target_mean)**2).sum()
    print("accuracy :", round(acc,4))
    return weight, bias, loss_arr
    
def get_data(): 
    number_of_rooms_list, price_list = [], []
    with open("BostonData1.txt", "r") as f: #read text file as 'read mode'
             for lines in f: 
                a = lines[0:5]
                b = lines[5:10]
                number_of_rooms_list.append(float(a.replace(" ", ""))) #부동소수점 자료형
                price_list.append(float(b.replace(" ", ""))) #still 'list'
    input_array = np.array(number_of_rooms_list) #change to array
    target_array = np.array(price_list)
    return input_array, target_array

#predict function(예측함수)
def predict(input_array, weight, bias):
    predict_array = weight*(input_array) + bias
    return predict_array

#loss function(손실함수)
def lossfunction(diff_array):
    loss = np.sum((diff_array)**2)/len(diff_array)
    return loss

#__name__ : interpreter 글로벌 변수 
if __name__ == "__main__": #인터프리터에서 직접 실행했을 경우에만 if문을 실행해라(import 말고)
    number_of_rooms, prices = get_data()
    epoch = 100000
    weight, bias, loss_arr = linear_regression_1D(number_of_rooms, prices, epoch)
    #draw result graph
    plt.plot(number_of_rooms, prices, ".", label="target")
    plt.plot(number_of_rooms, predict(number_of_rooms, weight, bias), ".", label="predict")
    plt.xlabel('number of rooms')
    plt.ylabel('price')
    plt.title('"x: Room, y: Price"')
    plt.savefig("LR_single_Scatter.png")
    
    #loss graph
    plt.plot(loss_arr)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning graph(loss)')
    plt.savefig("LR_single_lossgraph.png")