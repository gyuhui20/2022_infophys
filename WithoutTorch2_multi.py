import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("Step2: Development Multi Linear Regression Algorithm")

# input_array will be 2-d array, target_array will be 1-d array(2입력1출력 다중선형회귀)
#paramaters = weight, bias will be matrix
#input_array1=RM, input_array2=CRIM , target_array=MEDV, predict_array=예측한 출력값

def linear_regression_2D(input_array, target_array, epoch):
    #intialize weight & bias
    weight, bias = np.ones(shape=(2,1)), np.ones(shape=(506,1))
    loss = []
    lr = 1e-3
    #print(f"initial : weight, bias, loss: {weight}, {bias}, {loss}")

    for i in range(0, epoch):
        predict_array = predict(input_array, weight, bias)
        #print(predict_array)
        diff_array = predict_array - target_array
        #print(diff_array)
        gradient_weight1 = np.sum(2*input_array[:,0]*diff_array) / len(diff_array) 
        gradient_weight2 = np.sum(2*input_array[:,1]*diff_array) / len(diff_array)
        #print(gradient_weight)
        gradient_bias = np.sum(2*diff_array) / len(diff_array)
        #print(gradient_bias)
        weight[0,:] -= (lr * gradient_weight1)
        weight[1,:] -= (lr * gradient_weight2)
        bias[:,0] -= (lr * gradient_bias)
        loss.append(lossfunction(diff_array))
        #accuracy(R^2 결정계수)
        #target_mean=target_array.mean()
        #acc=1-((target_array-predict_array)**2).sum()/((target_array-target_mean)**2).sum()
        #??gradient 초기화시키고 다시 위로 올라가서 시작 #0으로 만들어주고 반복하게끔
        print(bias[0][0])
        print(weight[1][0])
        #print(f"final : weight1, weight2, bias, loss: {weight[0][0]:.4f}, {weight[1][0]:.4f}, {bias[0][0]:.4f}, {loss:.4f}")
        #print(f"acc : {acc:.4f}")
        return weight, bias, loss
    
def get_data(): 
    number_of_rooms_list, price_list, crime_rate_list = [], [], []
    with open("BostonData1.txt", "r") as f: #read text file as 'read mode'
                for lines in f: 
                    a = lines[0:5]
                    b = lines[5:10]
                    number_of_rooms_list.append(float(a.replace(" ", ""))) #부동소수점 자료형
                    price_list.append(float(b.replace(" ", ""))) #still 'list'
    with open("BostonData2.txt", "r") as f:
                for lines in f:
                    c = lines[3:10]
                    crime_rate_list.append(float(c))
    input_array = np.c_[np.array(number_of_rooms_list), np.array(crime_rate_list)] #change to array, 열 추가
    target_array = np.array(price_list)
    return(input_array, target_array)

def predict(input_array, weight, bias):
    predict_array = (input_array)@weight+bias
    return predict_array


def lossfunction(diff_array):
    loss = np.sum((diff_array)**2)/len(diff_array)
    return loss


#__name__ : interpreter 글로벌 변수 
if __name__ == "__main__": #인터프리터에서 직접 실행했을 경우에만 if문을 실행해라(import 말고)
    RM_CRIM, prices = get_data()
    epoch = 20000
    print("*"*70)
    print(RM_CRIM.shape)
    print("*"*70)
    print(prices.shape)
    #print("*"*70)
    weight, bias, loss = linear_regression_2D(RM_CRIM, prices, epoch)
    #draw result graph
    plt.subplot(2, 1, 1)
    plt.plot(RM_CRIM[:,0], prices, ".", label="target")
    plt.plot(RM_CRIM[:,0], predict(RM_CRIM, weight, bias), ".", label="predict")
    plt.legend()
    plt.title("x: Room, y: Price")
    
    plt.subplot(2, 1, 2)
    plt.plot(RM_CRIM[:,1], prices, ".", label="target")
    plt.plot(RM_CRIM[:,1], predict(RM_CRIM, weight, bias), ".", label="predict")
    plt.legend()
    plt.title("x: Criminal, y: Price")

    plt.tight_layout()
    plt.savefig("LR_multi_scatter.png")
    
    #loss graph
    #plt.plot(history[:,0], history[:,1],'b')
    #plt.xlabel('epoch')
    #plt.ylabel('loss')
    #plt.title('learning graph(loss)')
    #plt.savefig("LR_multi_lossgraph.png")

