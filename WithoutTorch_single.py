import numpy as np
import matplotlib.pyplot as plt

print("Step1: Development Single Linear Regression Algorithm")
#input_array, target_array will be 1-d array
#paramaters = weight, bias will be scalar
#input_array=RM, target_array=MEDV, predict_array=예측한 출력값

def linear_regression_1D(input_array, target_array, epoch):
    weight, bias = 3,3 #intialize weight & bias = 3
    loss = 0 #initialize loss =0
    lr = 1e-3
    print(f"initial : weight, bias, loss: {weight}, {bias}, {loss}")

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
        history=np.zeros((0,2))
        #accuracy(R^2 결정계수)
        target_mean=target_array.mean()
        acc=1-((target_array-predict_array)**2).sum()/((target_array-target_mean)**2).sum()
        #??gradient 초기화시키고 다시 위로 올라가서 시작 #0으로 만들어주고 반복하게끔
        print(f"final : weight, bias, loss: {weight:.4f}, {bias:.4f}, {loss:.4f}")
        print(f"acc : {acc:.4f}")
        return weight, bias, loss
    
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


def predict(input_array, weight, bias):
    predict_array = weight*(input_array) + bias
    return predict_array


def lossfunction(diff_array):
    loss = np.sum((diff_array)**2)/len(diff_array)
    return loss

history=np.zeros((0,2))
#__name__ : interpreter 글로벌 변수 
if __name__ == "__main__": #인터프리터에서 직접 실행했을 경우에만 if문을 실행해라(import 말고)
    number_of_rooms, prices = get_data()
    epoch = 20000
    #print(number_of_rooms)
    #print(prices)
    weight, bias, loss = linear_regression_1D(number_of_rooms, prices, epoch)
    #draw result graph
    plt.plot(number_of_rooms, prices, ".", label="target")
    plt.plot(number_of_rooms, predict(number_of_rooms, weight, bias), ".", label="predict")
    plt.savefig("LR_single_Scatter.png")
    #??100회마다 기록
    if (epoch%100 == 0) :
        history=np.vstack((history, np.array([epoch, loss.item()])))
        print(f'epoch : {epoch} / loss : {loss : .4f}')
    #lost list=[], 추가될 때마다 append를 해라.     
    
    #loss graph
    plt.plot(history[:,0], history[:,1],'b')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning graph(loss)')
    plt.savefig("LR_single_lossgraph.png")