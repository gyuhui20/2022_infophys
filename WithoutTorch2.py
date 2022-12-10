import numpy as np
import matplotlib.pyplot as plt

print("Step2: Development Multi Linear Regression Algorithm")

# input_array will be 2-d array, target_array will be 1-d array(2입력1출력 다중선형회귀)
#paramaters = weight, bias will be matrix, vector
#input_array1=RM, input_array2=CRIM , target_array=MEDV, predict_array=예측한 출력값

def linear_regression_2D(input_array, target_array, epoch):
    #intialize weight & bias
    weight_bias = np.array([[1],[1],[1]]) #앞 2개=weight
    loss = 0 #initialize loss =0
    lr = 1e-3
    print(f"initial : weight_bias, loss: {weight_bias}, {loss}")

    for i in range(0, epoch):
        predict_array = predict(input_array, weight_bias)
        #print(predict_array)
        diff_array = predict_array - target_array
        #print(diff_array)
        gradient_weight = np.sum(2*input_array*diff_array) / len(diff_array) 
        #print(gradient_weight)
        gradient_bias = np.sum(2*diff_array) / len(diff_array)
        #print(gradient_bias)
        weight_bias -= (lr * gradient_weight)
        weight_bias -= (lr * gradient_bias)
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
    number_of_rooms_list, price_list, crime_rate_list = [], [], []
    with open("BostonData1.txt", "r") as f: #read text file as 'read mode'
            for lines in f: 
                a = lines[0:5]
                b = lines[5:10]
                RM = number_of_rooms_list.append(float(a.replace(" ", ""))) #부동소수점 자료형
                MEDV = price_list.append(float(b.replace(" ", ""))) #still 'list'
    with open("BostonData2.txt", "r") as f:
            for lines in f:
                c = lines[3:10]
                CRIM = crime_rate_list.append(float(c))
    input_array = np.c_[np.array(RM), np.array(CRIM)] #change to array, 열 추가
    target_array = np.array(MEDV)
    return input_array, target_array
    

def predict(input_array, weight_bias):
    predict_array = weight_bias@(input_array)
    return predict_array


def lossfunction(diff_array):
    loss = np.sum((diff_array)**2)/len(diff_array)
    return loss


#__name__ : interpreter 글로벌 변수 
if __name__ == "__main__": #인터프리터에서 직접 실행했을 경우에만 if문을 실행해라(import 말고)
    RM_CRIM, prices = get_data()
    epoch = 15000
    #print(RM_CRIM)
    #print(prices)
    weight, bias, loss = linear_regression_2D(RM_CRIM, prices, epoch)
    #draw result graph
    plt.plot(RM_CRIM, prices, ".", label="target")
    plt.plot(RM_CRIM, predict(RM_CRIM, weight, bias), ".", label="predict")
    plt.savefig("result.png")
    
#loss graph
plt.plot(history[:,0], history[:,1],'b')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('learning graph(loss)')
plt.savefig("loss graph.png")

