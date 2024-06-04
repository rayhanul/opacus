

import pickle 
import matplotlib.pyplot as plt
import os 





def plot_accuracy_over_iterations(data_R2DP, data_Gaussian, title="Accuracy over Epoch"):

    acc_r2dps=data_R2DP['acc']
    acc_gaussian=data_Gaussian['acc']
    
    key_r2dp=list(acc_r2dps.keys())
    key_gaussian=list(acc_gaussian.keys())

    r2dp=list(acc_r2dps.values())
    gaussian=list(acc_gaussian.values())

    # key_r2dp=key_r2dp[0:len(gaussian)]
    # r2dp = r2dp[0:len(gaussian)]
    # print(len(gaussian))


    plt.figure(figsize=(10, 6))
    plt.plot(key_r2dp, r2dp, '-b', label='Accuracy R2DP')
    plt.plot(key_gaussian, gaussian, '-r', label='Accuracy Gaussian')

    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.legend()

    file_name=f"{title}.png"
    plt.title(title, fontsize=14, color='blue', fontweight='bold')
    plt.savefig(file_name)

    plt.show()
    plt.close()




def plot_epsilon_over_iterations(data_R2DP, data_Gaussian, title="epsilon_over_iterations"):

    acc_r2dps=data_R2DP['epsilon']
    acc_gaussian=data_Gaussian['epsilon']
    
    key_r2dp=list(acc_r2dps.keys())
    key_gaussian=list(acc_gaussian.keys())

    r2dp=list(acc_r2dps.values())
    gaussian=list(acc_gaussian.values())


    print(len(gaussian))
    plt.figure(figsize=(10, 6))
    plt.plot(key_r2dp, r2dp, '-b', label='Epsilon R2DP')
    plt.plot(key_gaussian, gaussian, '-r', label='Epsilon Gaussian')

    plt.xlabel("Time")
    plt.ylabel("Epsilon")
    plt.legend()

    file_name=f"{title}.png"
    plt.title(title, fontsize=14, color='blue', fontweight='bold')
    plt.savefig(file_name)

    plt.show()
    plt.close()



if __name__=="__main__":


    file_path=os.path.join(os.path.dirname(__file__), 'data_r2dp_dynamic0.5.pkl')

    with open(file_path, 'rb') as file:
        data_r2dp = pickle.load(file)

    file_path=os.path.join(os.path.dirname(__file__), 'data_gaussian_0.5.pkl')
    with open(file_path, 'rb') as file:
        data_gaussian = pickle.load(file)

    title="Accuracy over Epoch for Epsilon=0.5"

    plot_accuracy_over_iterations(data_R2DP=data_r2dp, data_Gaussian=data_gaussian, title=title)

    title="Epsilon over Epoch with budget=0.5"

    plot_epsilon_over_iterations(data_R2DP=data_r2dp, data_Gaussian=data_gaussian, title=title)
