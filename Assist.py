import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def linear_regression_2D(input_array, target_array, epoch):
    weight, bias = np.zeros(shape=(2,1)), np.zeros(shape=(506,1))
    loss_list = []
    weight_list, bias_list = [[], []], []
    grad_weight_list, grad_bias_list = [[], []], []
    lr = 1e-6

    for i in range(0, epoch):
        predict_array = predict(input_array, weight, bias)
        diff_array = predict_array - target_array
        gradient_weight1 = np.sum(2*input_array[:,0]*diff_array) / len(diff_array) 
        gradient_weight2 = np.sum(2*input_array[:,1]*diff_array) / len(diff_array)
        gradient_bias = np.sum(2*diff_array) / len(diff_array)
        weight[0,0] -= (lr * gradient_weight1)
        weight[1,0] -= (lr * gradient_weight2)
        bias[:,0] -= (lr * gradient_bias)
        loss = lossfunction(diff_array)

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

    return weight_arr, bias_arr, loss_arr, grad_weight_arr, grad_bias_arr
    
def get_data(): 
    number_of_rooms_list, price_list, crime_rate_list = [], [], []
    with open("BostonData1.txt", "r") as f:
                for lines in f: 
                    a = lines[0:5]
                    b = lines[5:10]
                    number_of_rooms_list.append(float(a.replace(" ", "")))
                    price_list.append(float(b.replace(" ", "")))
    with open("BostonData2.txt", "r") as f:
                for lines in f:
                    c = lines[3:10]
                    crime_rate_list.append(float(c))
    input_array = np.c_[np.array(number_of_rooms_list), np.array(crime_rate_list)]
    target_array = np.array(price_list)
    return(input_array, target_array)

def predict(input_array, weight, bias):
    predict_array = (input_array)@weight+bias
    return predict_array


def lossfunction(diff_array):
    loss = np.sum((diff_array)**2)/len(diff_array)
    return loss


if __name__ == "__main__":
    input_arr, target_arr = get_data()
    epoch = 500
    weight_arr, bias_arr, loss_arr, grad_weight_arr, grad_bias_arr = linear_regression_2D(input_arr, target_arr, epoch)


    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(input_arr[:,0], target_arr, ".", label="target")
    plt.plot(input_arr[:,0], predict(input_arr, weight_arr[:, -1], bias_arr[-1]), ".", label="predict")
    plt.legend()
    plt.title("x: Room, y: Price")
    
    plt.subplot(2, 1, 2)
    plt.plot(input_arr[:,1], target_arr, ".", label="target")
    plt.plot(input_arr[:,1], predict(input_arr, weight_arr[:, -1], bias_arr[-1]), ".", label="predict")
    plt.legend()
    plt.title("x: Criminal, y: Price")

    plt.tight_layout()
    plt.savefig("Assist_LR_multi_scatter.png")
    
    #loss graph
    plt.figure()
    plt.plot(loss_arr)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('learning graph(loss)')
    plt.savefig("Assist_LR_multi_lossgraph.png")

    #weight, bias graph
    plt.figure()
    plt.plot(weight_arr[0, :], ".", label="weight1")
    plt.plot(weight_arr[1, :], ".", label="weight2")
    plt.plot(bias_arr[:], ".", label="bias")
    plt.xlabel('epoch')
    plt.ylabel('param')
    plt.title('learning graph(param)')
    plt.legend()
    plt.savefig("Assist_LR_multi_paramgraph.png")

    #weight, bias's gradient graph
    plt.figure()
    plt.plot(grad_weight_arr[0, :], ".", label="grad_weight1")
    plt.plot(grad_weight_arr[1, :], ".", label="grad_weight2")
    plt.plot(grad_bias_arr[:], ".", label="grad_bias")
    plt.xlabel('epoch')
    plt.ylabel('grad_param')
    plt.title('learning graph(grad_param)')
    plt.legend()
    plt.savefig("Assist_LR_multi_grad_paramgraph.png")

