import numpy as np
import matplotlib.pyplot as plt

print("Step2: Development Multi Linear Regression Algorithm")
# input_array will be 2-d array, target_array will be 1-d array(2입력1출력 다중선형회귀)
#paramaters = weight, bias will be matrix
#input_array1=RM, input_array2=CRIM, target_array=MEDV, predict_array=예측한 출력값

def linear_regression_2D(input_array, target_array, epoch):
    weight, bias = np.zeros(shape=(2,1)), np.zeros(shape=(506,1))
    #for record history(그래프 그리려고)
    loss_list = []
    weight_list, bias_list = [[], []], []
    grad_weight_list, grad_bias_list = [[], []], []
    lr = 1e-6
    print(f"initial : weight1, weight2, bias : {weight[0,0]:.4f}, {weight[1,0]:.4f}, {bias[0,0]:.4f}")
    
    for i in range(0, epoch):
        #predict variable
        predict_array = predict(input_array, weight, bias)
        #difference function(편차 함수)
        diff_array = predict_array - target_array
        #calculate gradients(경사 계산)
        gradient_weight1 = np.sum(2*input_array[:,0]*diff_array) / len(diff_array) 
        gradient_weight2 = np.sum(2*input_array[:,1]*diff_array) / len(diff_array)
        gradient_bias = np.sum(2*diff_array) / len(diff_array)
        #adjust parameters(2 weights, bias)(파라미터 수정)
        weight[0,0] -= (lr * gradient_weight1)
        weight[1,0] -= (lr * gradient_weight2)
        bias[:,0] -= (lr * gradient_bias)
        #loss variable
        loss = lossfunction(diff_array)

        #for record history(그래프 그리려고)
        weight_list[0].append(weight[0,0])
        weight_list[1].append(weight[1,0])
        bias_list.append(bias[0,0])

        grad_weight_list[0].append(gradient_weight1)
        grad_weight_list[1].append(gradient_weight2)
        grad_bias_list.append(gradient_bias)

        loss_list.append(loss)
    weight_arr = np.array(weight_list)
    bias_arr = np.array(bias_list)
    
    grad_weight_arr = np.array(grad_weight_list)
    grad_bias_arr = np.array(grad_bias_list)

    loss_arr = np.array(loss_list)
    
    print(f"final : weight1, weight2, bias, loss: {weight_arr[0, -1]:.4f}, {weight_arr[1, -1]:.4f}, {bias_arr[-1]:.4f}, {loss_arr[-1]:.4f}")
    
    #accuracy(R^2 결정계수)
    #target_mean=target_array.mean()
    #acc=1-((target_array-predict_array)**2).sum()/((target_array-target_mean)**2).sum()
    #print("accuracy :", round(acc,4))
    
    return weight_arr, bias_arr, loss_arr, grad_weight_arr, grad_bias_arr
    
def get_data(): 
    number_of_rooms_list, price_list, lower_population_list = [], [], []
    with open("BostonData1.txt", "r") as f: #read text file as 'read mode'
                for lines in f: 
                    a = lines[0:5]
                    b = lines[5:10]
                    number_of_rooms_list.append(float(a.replace(" ", ""))) #부동소수점 자료형
                    price_list.append(float(b.replace(" ", ""))) #still 'list'
    with open("BostonData3.txt", "r") as f:
                for lines in f:
                    d = lines[0:4]
                    lower_population_list.append(float(d))
    #change to array / add row
    input_array = np.c_[np.array(number_of_rooms_list), np.array(lower_population_list)] 
    target_array = np.array(price_list)
    return(input_array, target_array)

#predict function(예측함수)
def predict(input_array, weight, bias):
    predict_array = (input_array)@weight+bias
    return predict_array

#loss function(손실함수)
def lossfunction(diff_array):
    loss = np.sum((diff_array)**2)/len(diff_array)
    return loss

#__name__ : interpreter 글로벌 변수 
if __name__ == "__main__":
    #인터프리터에서 직접 실행했을 경우에만 if문을 실행해라(import 말고)
    input_arr, target_arr = get_data()
    epoch = 100
    weight_arr, bias_arr, loss_arr, grad_weight_arr, grad_bias_arr = linear_regression_2D(input_arr, target_arr, epoch)
    #result graph
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(input_arr[:,0], target_arr, ".", label="target")
    plt.plot(input_arr[:,0], (input_arr[:,0]*weight_arr[0, -1]), ".", label="predict")
    plt.legend()
    plt.title("x: Room, y: Price")
    
    plt.subplot(2, 1, 2)
    plt.plot(input_arr[:,1], target_arr, ".", label="target")
    plt.plot(input_arr[:,1], (input_arr[:,1]*weight_arr[1, -1]), ".", label="predict")
    plt.legend()
    plt.title("x: Lower Population, y: Price")

    plt.tight_layout()
    plt.savefig("Assist_LR_multi_scatter1.png")
    
    #total graph
    plt.figure()
    plt.plot(input_arr[:,1], target_arr, ".", label="target")
    plt.plot(input_arr[:,1], (input_arr[:,1]@weight_arr[:,:]), ".", label="predict")
    plt.legend()
    plt.title("x: Lower Population, y: Price")
    plt.savefig("Assist_LR_multi_scaater_total.png")

    #loss graph
    plt.figure()
    plt.plot(loss_arr)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning graph(loss)')
    plt.savefig("Assist_LR_multi_lossgraph1.png")

    #weight, bias graph
    plt.figure()
    plt.plot(weight_arr[0, :], ".", label="weight1")
    plt.plot(weight_arr[1, :], ".", label="weight2")
    plt.plot(bias_arr[:], ".", label="bias")
    plt.xlabel('epoch')
    plt.ylabel('param')
    plt.title('learning graph(param)')
    plt.legend()
    plt.savefig("Assist_LR_multi_paramgraph1.png")

    #weight, bias's gradient graph
    plt.figure()
    plt.plot(grad_weight_arr[0, :], ".", label="grad_weight1")
    plt.plot(grad_weight_arr[1, :], ".", label="grad_weight2")
    plt.plot(grad_bias_arr[:], ".", label="grad_bias")
    plt.xlabel('epoch')
    plt.ylabel('grad_param')
    plt.title('learning graph(grad_param)')
    plt.legend()
    plt.savefig("Assist_LR_multi_grad_paramgraph1.png")

