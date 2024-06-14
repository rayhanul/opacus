

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

def plot_accuracy(data, legends, line_styles, is_inset=True, title="Accuracy_vs_Epochs"):


    # if not is_inset: 

    plt.figure(figsize=(10, 6))

    for key, val in data.items():
        val_key=val.keys()
        val_values=val.values()
        plt.plot(val_key, val_values, line_styles[key], label=legends[key])

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
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
    plt.title("Accuracy over Epochs")
    plt.savefig(file_name)
    plt.show()
    plt.close()




def plot_epsilon(data, legends, line_styles, is_inset=True, title="Epsilon_vs_Epochs"):

 

    if not is_inset: 

        plt.figure(figsize=(10, 6))

        for key, val in data.items():
            val_key=val.keys()
            val_values=val.values()
            plt.plot(val_key, val_values, line_styles[key], label=legends[key])

        plt.xlabel("Time")
        plt.ylabel('$\\epsilon$')
        plt.title(title, fontsize=14, color='blue', fontweight='bold')
        plt.legend()
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        for key, val in data.items():
            val_key=val.keys()
            val_values=val.values()
            ax.plot(val_key, val_values, line_styles[key], label=legends[key])

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('$\\epsilon$')
        ax.legend()

        # Create an inset within the main plot for the zoomed-in view

        ax_inset = ax.inset_axes([0.5, 0.5, 0.45, 0.45])  # Change the position and size as needed
        for key, val in data.items():
            val_key=list(val.keys())
            val_values=list(val.values())
            ax_inset.plot(val_key[:70], val_values[:70], line_styles[key], label=legends[key])
            

        # ax_inset.set_title('Zoomed View (0 to 100)')
        # ax_inset.set_xlabel('X-axis (0 to 100)')
        # ax_inset.set_ylabel('Data values')

        # Optionally, set tighter limits for the zoomed-in axes
        ax_inset.set_xlim(0, 60)
        ax_inset.set_ylim(np.min(val_values[:70]),15)
    
    if is_inset:
        file_name=f"{title}_with_inset.png"
    else: 
        file_name=f"{title}.png"
    plt.grid(True)
    plt.title("Epsilon over Epochs")
    plt.savefig(file_name)
    plt.show()
    plt.close()

def plot_epsilon_given_budget(data, budgets, legends, line_styles, title="Epsilon_vs_Epochs_with_budget"):

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for key, budget in enumerate(budgets):
        ax = axs[key//3, key%3]

        data_under_budget={}
        for key, val in data.items():
            val_key=val.keys()
            val_values=list(val.values())

            values_under_budget = [eps for eps in val_values if eps < budget]

            data_under_budget.update({key: values_under_budget})

        for key, val in data_under_budget.items():
            val_key=range(0, len(val))

        
            ax.plot(val_key, val, line_styles[key], label=legends[key])
            ax.legend(loc='best')
            subplot_title=f"Privacy budget={budget}"
            ax.title.set_text(subplot_title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('$\\epsilon$')

    file_name=f"{title}.png"
    # plt.title("Epsilon under given budget")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    # plt.close()


def plot_accuracy_given_budget(epsilon, accuracy, budgets, legends, line_styles, title="Epsilon_vs_Epochs_with_budget"):

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    for key, budget in enumerate(budgets):
        ax = axs[key//3, key%3]

        data_under_budget={}
        accuracy_under_budget={}
        for inner_key, val in epsilon.items():
            val_key=val.keys()
            val_values=list(val.values())

            values_under_budget = [eps for eps in val_values if eps < budget]

            acc_under_budget = list(accuracy[inner_key].values())
            acc_under_budget = acc_under_budget[0: len(values_under_budget)]

            data_under_budget.update({inner_key: values_under_budget})
            accuracy_under_budget.update({inner_key: acc_under_budget})

        for key, val in data_under_budget.items():
            val_key=range(0, len(val))

        
            ax.plot(val_key, val, line_styles[key], label=legends[key])
            ax.legend(loc='best')
            subplot_title=f"Privacy budget={budget}"
            ax.title.set_text(subplot_title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('$\\epsilon$')

    file_name=f"{title}.png"
    # plt.title("Epsilon under given budget")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    # plt.close() 

def plot_accuracy_vs_epsilon(epsilon, accuracy, budgets, legends, line_styles, noise, title):
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    index=0
    for budget in budgets:
        data_under_budget={}
        for sigma in noise:
            epsilon_sigma=epsilon[sigma]
            accuracy_sigma=accuracy[sigma]

            combined_data = { epsilon_sigma[key]: accuracy_sigma[key] for key in epsilon_sigma}
            under_budget={eps: acc for eps, acc in combined_data.items() if eps < budget}
            data_under_budget.update({sigma: under_budget })

            # for key, val in data_under_budget:
            # plot_epsilon_given_budget(data_under_budget, budgets, legends, line_styles, title)

            
            
        for sigma, data in data_under_budget.items():

            ax = axs[index//2, index%2]
            
            
            x_axis_data=list(data.keys())
            y_axis_data=list(data.values())

            ax.plot(x_axis_data, y_axis_data, line_styles[sigma], label=legends[sigma])
            ax.legend(loc='best')
            subplot_title=f"Privacy budget={budget}"
            ax.title.set_text(subplot_title)
            ax.set_ylabel('Accuracy (%)')
            ax.set_xlabel('$\\epsilon$')
        
        index = index+1 

    file_name=f"{title}.png"

    plt.xlabel('$\\epsilon$')
    plt.ylabel('Accuracy (%)')
    # plt.title('Accuracy over epsilon')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

if __name__=="__main__":

    legends={
        # "0.3" : "Gaussian noise, $\sigma$=0.3", 
        "0.5" :"Gaussian noise, $\sigma$=0.5",
        "0.8" : "Gaussian noise, $\sigma$=0.8", 
        "1.0": "Gaussian noise, $\sigma$=1.0", 
        "1.2":"Gaussian noise, $\sigma$=1.2", 
        "1.5":"Gaussian noise, $\sigma$=1.5",
        # "1.7":"Gaussian noise, $\sigma$=1.7",
        "0.0": "PLRV noise"
    }


    line_styles4={
        # "0.3" : '-', 
        # "0.5" :'--',
        # "0.8" : ':', 
        "1.0": '-.', 
        "1.2": (0, (3, 5, 1, 5)), 
        "1.5":(0, (3, 5, 1, 5)),
        # "1.7" : '-',
        # "0.0": (0, (3, 10, 1, 10, 1, 10))
    }

    line_styles44={
        # "0.3" : '-', 
        "0.5" :'--',
        "0.8" : ':', 
        "1.0": '-.', 
        "1.2": '-', 
        "1.5":'--',
        "1.7" : '-',
        "0.0": ':'
    }

    line_styles={
        # "0.3" : 'k--', 
        "0.5" :'k-',
        "0.8" : '-', 
        "1.0": ':', 
        "1.2": '--', 
        "1.5":'-.',
        # "1.7" : '-',
        "0.0": '-'
    }


    files= {
        # "0.3" : "data_gaussian_sigma_1.0_epochs_1000.pkl", 
        "0.5" :"data_gaussian_cifar10_sigma_0.5_epochs_1000.pkl",
        "0.8" : "data_gaussian_cifar10_sigma_0.8_epochs_1000.pkl", 
        "1.0": "data_gaussian_cifar10_sigma_1.0_epochs_1000.pkl", 
        "1.2":"data_gaussian_cifar10_sigma_1.2_epochs_1000.pkl", 
        "1.5":"data_gaussian_cifar10_sigma_1.5_epochs_1000.pkl",
        # "1.7":"data_gaussian_sigma_1.7_epochs_1000.pkl",
        "0.0": "data_gaussian_cifar10_dynamic_noise_epochs_1000.pkl"
    }

    # noise= [ '0.5', '0.8','1.0', '1.2','1.5', '1.7', '0.0']
    noise= [ '0.5','0.8', '1.0', '1.2','1.5', '0.0']

    data_epsilon={}
    data_accuracy={}

    for sigma, file in files.items():
        file_path=os.path.join(os.path.dirname(__file__), file)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

            data_eps = { key:val['epsilon'] for key, val in data.items()}
            data_acc = { key: val['accuracy'] for key, val in data.items()}

            data_epsilon.update({sigma:data_eps})
            data_accuracy.update({sigma:data_acc})


    print("data reading done")

    plot_epsilon(data_epsilon, legends, line_styles, True)
    
    plot_accuracy(data_accuracy, legends, line_styles, True)

    plot_epsilon_given_budget(data_epsilon, [0.3, 0.5, 0.8, 1, 1.5, 2.0], legends, line_styles)
    plot_accuracy_given_budget(data_epsilon, data_accuracy, [0.3, 0.5, 0.8, 1, 1.5, 2.0], legends, line_styles)
    plot_accuracy_vs_epsilon(data_epsilon, data_accuracy, [ 0.5, 1, 2.0, 4.0], legends, line_styles, noise, title="Accuracy_vs_Epsilon")
    
    