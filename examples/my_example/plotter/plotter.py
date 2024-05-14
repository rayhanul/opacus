

import pickle 
import matplotlib.pyplot as plt






def plot_accuracy_over_iterations(data_R2DP, data_Gaussian, title="accuracy_over_iterations"):

    acc_r2dps=data_R2DP['acc']
    acc_gaussian=data_Gaussian['acc']
    
    key=list(acc_r2dps.keys())
    r2dp=list(acc_r2dps.values())
    gaussian=list(acc_gaussian.values())

    plt.figure(figsize=(10, 6))
    plt.plot(key, r2dp, '-b', label='Accuracy R2DP')
    plt.plot(key, gaussian, '-r', label='Accuracy Gaussian')

    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.legend()

    file_name=f"{title}.png"
    plt.title(title, fontsize=14, color='blue', fontweight='bold')
    plt.savefig(file_name)

    plt.show()
    plt.close()








if __name__=="__main__":
    with open('data_r2dp.pkl', 'rb') as file:
        data_r2dp = pickle.load(file)

    with open('data_gaussian.pkl', 'rb') as file:
        data_gaussian = pickle.load(file)


    plot_accuracy_over_iterations(data_R2DP=data_r2dp, data_Gaussian=data_gaussian)