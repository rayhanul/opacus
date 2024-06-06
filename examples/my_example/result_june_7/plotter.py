

import pickle 
import matplotlib.pyplot as plt
import os 
import numpy as np 




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

def plot_accuracy(data, is_inset=True, title="Accuracy_vs_Epochs"):

    legends={
        "0.3" : "Gaussian noise with sigma=0.3", 
        "0.5" :"Gaussian noise with sigma=0.5",
        "0.8" : "Gaussian noise with sigma=0.8", 
        "1.0": "Gaussian noise with sigma=1.0", 
        "1.2":"Gaussian noise with sigma=1.2", 
        "1.5":"Gaussian noise with sigma=1.5",
        "0.0": "PLRV noise"
    }

    line_styles={
        "0.3" : '-', 
        "0.5" :'--',
        "0.8" : (0, (3, 10, 1, 10, 1, 10)), 
        "1.0": (0, (3, 5, 1, 5)), 
        "1.2":(0, (1, 5,5, 3)), 
        "1.5":(0, (3, 5, 1, 5)),
        "0.0": (0, (5, 10))
    }

    # if not is_inset: 

    plt.figure(figsize=(10, 6))

    for key, val in data.items():
        val_key=val.keys()
        val_values=val.values()
        plt.plot(val_key, val_values, label=legends[key], linestyle=line_styles[key])

    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.title(title, fontsize=14, color='blue', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    # else:
    #     fig, ax = plt.subplots()
    #     for key, val in data.items():
    #         val_key=val.keys()
    #         val_values=val.values()
    #         ax.plot(val_key, val_values, label=legends[key], linestyle=line_styles[key])

    #     ax.set_title(title)
    #     ax.set_xlabel('Epoch')
    #     ax.set_ylabel('Accuracy')
    #     ax.legend()

    #     # Create an inset within the main plot for the zoomed-in view

    #     ax_inset = ax.inset_axes([0.5, 0.5, 0.45, 0.45])  # Change the position and size as needed
    #     for key, val in data.items():
    #         val_key=list(val.keys())
    #         val_values=list(val.values())
    #         ax_inset.plot(val_key[:70], val_values[:70], label=legends[key], linestyle=line_styles[key])
            
    #     ax_inset.set_xlim(0, 60)
    #     ax_inset.set_ylim(np.min(val_values[:70]),15)

    file_name=f"{title}.png"
    plt.savefig(file_name)
    plt.show()
    plt.close()




def plot_epsilon(data, is_inset=True, title="Epsilon_vs_Epochs"):

    legends={
        "0.3" : "Gaussian noise with sigma=0.3", 
        "0.5" :"Gaussian noise with sigma=0.5",
        "0.8" : "Gaussian noise with sigma=0.8", 
        "1.0": "Gaussian noise with sigma=1.0", 
        "1.2":"Gaussian noise with sigma=1.2", 
        "1.5":"Gaussian noise with sigma=1.5",
        "0.0": "PLRV noise"
    }

    line_styles={
                "0.3" : '-', 
        "0.5" :'--',
        "0.8" : (0, (3, 10, 1, 10, 1, 10)), 
        "1.0": (0, (3, 5, 1, 5)), 
        "1.2":(0, (1, 5,5, 3)), 
        "1.5":(0, (3, 5, 1, 5)),
        "0.0": (0, (5, 10))
    }

    if not is_inset: 

        plt.figure(figsize=(10, 6))

        for key, val in data.items():
            val_key=val.keys()
            val_values=val.values()
            plt.plot(val_key, val_values, label=legends[key], linestyle=line_styles[key])

        plt.xlabel("Time")
        plt.ylabel("Epsilon")
        plt.title(title, fontsize=14, color='blue', fontweight='bold')
        plt.legend()
        plt.tight_layout()
    else:
        fig, ax = plt.subplots()
        for key, val in data.items():
            val_key=val.keys()
            val_values=val.values()
            ax.plot(val_key, val_values, label=legends[key], linestyle=line_styles[key])

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Epsilon')
        ax.legend()

        # Create an inset within the main plot for the zoomed-in view

        ax_inset = ax.inset_axes([0.5, 0.5, 0.45, 0.45])  # Change the position and size as needed
        for key, val in data.items():
            val_key=list(val.keys())
            val_values=list(val.values())
            ax_inset.plot(val_key[:70], val_values[:70], label=legends[key], linestyle=line_styles[key])
            

        # ax_inset.set_title('Zoomed View (0 to 100)')
        # ax_inset.set_xlabel('X-axis (0 to 100)')
        # ax_inset.set_ylabel('Data values')

        # Optionally, set tighter limits for the zoomed-in axes
        ax_inset.set_xlim(0, 60)
        ax_inset.set_ylim(np.min(val_values[:70]),15)
    
    file_name=f"{title}.png"
    plt.savefig(file_name)
    plt.show()
    plt.close()

if __name__=="__main__":

    files= {
        "0.3" : "data_gaussian_sigma_1.0_epochs_1000.pkl", 
        "0.5" :"data_gaussian_sigma_0.5_epochs_1000.pkl",
        "0.8" : "data_gaussian_sigma_1.0_epochs_1000.pkl", 
        "1.0": "data_gaussian_sigma_1.0_epochs_1000.pkl", 
        "1.2":"data_gaussian_sigma_1.2_epochs_1000.pkl", 
        "1.5":"data_gaussian_sigma_1.5_epochs_1000.pkl",
        "0.0": "data_r2dp_dynamic0.pkl"
    }
    data_epsilon={}
    data_accuracy={}

    for sigma, file in files.items():
        file_path=os.path.join(os.path.dirname(__file__), file)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

            data_epsilon.update({sigma:data['epsilon']})
            data_accuracy.update({sigma:data['acc']})


    print("data reading done")

    plot_epsilon(data_epsilon, True)
    
    plot_accuracy(data_accuracy, True)

    # with open(file_path, 'rb') as file:
    #     data_r2dp = pickle.load(file)

    # file_path=os.path.join(os.path.dirname(__file__), 'data_gaussian_sigma_0.5_epochs_1000.pkl')
    # with open(file_path, 'rb') as file:
    #     data_gaussian = pickle.load(file)

    # title="Accuracy over Epoch for Epsilon=1.0"

    # plot_accuracy_over_iterations(data_R2DP=data_r2dp, data_Gaussian=data_gaussian, title=title)

    # title="Epsilon over Epoch with budget=1.0"

    # plot_epsilon_over_iterations(data_R2DP=data_r2dp, data_Gaussian=data_gaussian, title=title)
