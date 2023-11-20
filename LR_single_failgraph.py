import numpy as np
import matplotlib.pyplot as plt

print("Step1: Development Single Linear Regression Algorithm")
#input_array, target_array will be 1-d array
#paramaters = weight, bias will be scalar
#input_array=RM, target_array=MEDV, predict_array=예측한 출력값

def linear_regression_1D(input_array, target_array, epoch):
    weight, bias = np.zeros(shape=(1,1)), np.zeros(shape=(506,1))
    loss_list = []
    weight_list, bias_list = [], []
    grad_weight_list, grad_bias_list = [], []
    lr = 1e-3
    #print(f"initial : weight, bias : {weight}, {bias}")

    for i in range(0, epoch):
        predict_array = predict(input_array, weight, bias)
        diff_array = predict_array - target_array
        gradient_weight = np.sum(2*input_array[:]*diff_array) / len(diff_array) 
        gradient_bias = np.sum(2*diff_array) / len(diff_array)
        weight -= (lr * gradient_weight)
        bias -= (lr * gradient_bias)
        loss = lossfunction(diff_array)
        #accuracy(R^2 결정계수)
        #target_mean=target_array.mean()
        #acc=1-((target_array-predict_array)**2).sum()/((target_array-target_mean)**2).sum()
        weight_list.append(weight[0,0])
        bias_list.append(bias[0,0])

        grad_weight_list.append(gradient_weight)
        grad_bias_list.append(gradient_bias)

        loss_list.append(loss)
        #print(f"final : weight, bias, loss: {np.round(weight_list, 4)}, {bias_list[-1]:.4f}, {loss_list[-1]:.4f}")
        #print(f"acc : {acc:.4f}")
    weight_arr = np.array(weight_list)
    bias_arr = np.array(bias_list)
    grad_weight_arr = np.array(grad_weight_list)
    grad_bias_arr = np.array(grad_bias_list)
    loss_arr = np.array(loss_list)
    return weight_arr, bias_arr, loss_arr, grad_weight_arr, grad_bias_arr
    
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
    return (input_array, target_array)


def predict(input_array, weight, bias):
    predict_array =np.array([input_array])*np.array([weight]) + (bias)
    return predict_array


def lossfunction(diff_array):
    loss = np.sum((diff_array)**2)/len(diff_array)
    return loss

#__name__ : interpreter 글로벌 변수 
if __name__ == "__main__": #인터프리터에서 직접 실행했을 경우에만 if문을 실행해라(import 말고)
    input_arr, target_arr = get_data()
    epoch = 300
    weight_arr, bias_arr, loss_arr, grad_weight_arr, grad_bias_arr = linear_regression_1D(input_arr, target_arr, epoch)
    #print(grad_weight_arr[0,-1])
    plt.figure()
    plt.plot(input_arr, target_arr, ".", label="target")
    plt.plot(input_arr, predict(input_arr, weight_arr, bias_arr), ".", label="predict")
    plt.title("x: Room, y: Price")
    plt.savefig("LR_single_Scatter.png")

    #loss graph
    plt.plot(loss_arr)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning graph(loss)')
    plt.savefig("LR_single_lossgraph.png")

    #weight, bias graph
    #plt.figure()
    #plt.plot(weight_arr[:], ".", label="weight")
    #plt.plot(bias_arr[:], ".", label="bias")
    #plt.xlabel('epoch')
    #plt.ylabel('param')
    #plt.title('learning graph(param)')
    #plt.legend()
    #plt.savefig("LR_single_paramgraph.png")

    #weight, bias's gradient graph
    #plt.plot(grad_weight_arr[:], ".", label="grad_weight")
    #plt.plot(grad_bias_arr[:], ".", label="grad_bias")
    #plt.xlabel('epoch')
    #plt.ylabel('grad_param')
    #plt.title('learning graph(grad_param)')
    #plt.savefig("LR_single_grad_paramgraph.png")