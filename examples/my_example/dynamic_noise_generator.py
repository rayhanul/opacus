


import numpy as np
import json 
import os 

def generate_dynamic_noise(path, distributions=["Gamma", "Exponential", "Uniform"], noise_size=1):

    file_path = os.path.join(os.path.dirname(__file__), 'parameters', path)

    with open(file_path, 'r') as file:
        params = json.load(file)

    b = generate_dynamic_b(params, distributions=distributions, noise_size=noise_size)
    noise = np.random.laplace(0, 1/b)
    return noise


def generate_dynamic_b(params, distributions, noise_size=1):
    b = 0
    
    if "Gamma" in distributions:
        b = b + params['a1']*np.random.gamma(params["G_k"], params["G_theta"], noise_size)
    else:
        b = b + 0
    
    if "Exponential" in distributions:
        b = b + params['a3']*np.random.exponential(params["E_lambda"], noise_size)
    else:
        b = b + 0
    
    if "Uniform" in distributions:
        b = b + params['a4'] * np.random.uniform(params["U_a"], params["U_b"], noise_size)
    else:
        b = b + 0
    
    return b


def calculate_log_epsilon(epsilon, gamma):
    """
    Calculate the logarithmic expression given gamma and x.

    Parameters:
        gamma (float): The parameter gamma in the equation.
        x (float): The exponent variable in the equation.

    Returns:
        float: The result of the logarithmic expression.
    """
    # Calculate the inner expression
    inner_expression = 1 + gamma * (np.exp(epsilon) - 1)

    # Calculate the logarithm of the inner expression
    result = np.log(inner_expression)

    return result


# if __name__=="__main__":
#     # generate_dynamic_noise('optimized_usefulness/lmo_eps3.json')

#     epsilon = calculate_log_epsilon(0.24, 0.00426666666)
#     print(epsilon)