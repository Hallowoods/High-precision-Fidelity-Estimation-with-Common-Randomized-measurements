"""

This module provides functions for calculating upper bound terms(including cross-characteristic, characteristic, variance, and their amplified version, sample complexity) for snk states with different k. Both for CRM and THR protocol.

Pauli noise is assumed.

The code to caulate the data for CRM/THR based on 4-design measurements in Fig. 2. The columns 'Sample#', 'Sample#_thrifty', 'infid^{-1}' is the result.
Author: YZY
"""



import time
import numpy as np
# import paddle
import matplotlib.pyplot as plt
import paddle as paddle
import paddle_quantum  as pq
from paddle_quantum.qinfo import trace_distance
from paddle_quantum.qinfo import shadow_trace
from paddle_quantum.state import to_state
from paddle_quantum.backend import Backend
from paddle_quantum import Hamiltonian
from paddle_quantum.qinfo import random_hamiltonian_generator
from paddle_quantum.trotter import  get_1d_heisenberg_hamiltonian
from paddle_quantum.intrinsic import _type_fetch, _type_transform
from paddle_quantum.ansatz import Circuit
from datetime import datetime  # 导入 datetime 模块
import multiprocessing as mp
from paddle_quantum.ansatz import Circuit
from paulichannel_utils import generate_channel
import pandas as pd
import pickle
import random
import csv
from scipy import stats
from itertools import product
import json
from multiprocessing import Pool

def generate_pauli_group(nqubit):
    """Generate n-qubit  Pauli Group，in JSON or Pickle files"""
    pauli_labels = ['x', 'y', 'z']
    pauli_group = []
    json_path = f"pauli_group_{nqubit}qubit.json"
    pkl_path = f"pauli_group_{nqubit}qubit.pkl"
    
    for pauli_ops in product(['I'] + pauli_labels, repeat=nqubit):
        formatted_pauli = [1]  
        pauli_str = []
        for idx, op in enumerate(pauli_ops):
            if op != 'I':
                pauli_str.append(f"{op}{idx}")
        if pauli_str:
            formatted_pauli.append(','.join(pauli_str))      
        elif not pauli_str and pauli_ops != tuple(['I'] * nqubit):
            continue
         pauli_group.append(formatted_pauli)
    # Write into a JSON file
    with open(json_path, "w") as f:
        json.dump(pauli_group, f, indent=4)

    # Write into a Picle file
    with open(pkl_path, "wb") as f:
        pickle.dump(pauli_group, f)
    del pauli_group[0]
    return pauli_group 

def load_pauli_group(nqubit):
    """Load the Pauli Group from JSON or Picle files"""
    json_path = f"pauli_group_{nqubit}qubit.json"
    pkl_path = f"pauli_group_{nqubit}qubit.pkl"
    
    with open(json_path, "r") as f:
        pauli_group_json = json.load(f)

    with open(pkl_path, "rb") as f:
        pauli_group_pickle = pickle.load(f)
    
    return pauli_group_json, pauli_group_pickle

def save_pauli_channel(channelinfo, filename):
    """Save the generated Pauli channel from channelinfo, write-in Pauli Channel file"""
    with open(filename, 'wb') as file:
        pickle.dump(channelinfo, file)

def load_pauli_channel(filename):
    '''Read the Pauli Channel file'''
    with open(filename, 'rb') as file:
        channelinfo = pickle.load(file)  
    return channelinfo


def generate_random_pauli_channel(nqubit,magnitude = 0.1,channel_num = 0):
    """Generated random Pauli channel, parameterized by magnitude and channel_num. channel_num indicates the number of choices of Pauli noise as an average (default to be 4**n-1), and magnitude controls the amplitude of the noise"""
    channelinfo = []
    existing_pauli_strings = set() 
    maxallowed = 4**nqubit - 1
    if channel_num == 0:    #default:random
        number_of_string = maxallowed
    else:
        number_of_string = channel_num
    

    while len(channelinfo) < number_of_string:  
        coefficient = random.uniform(0, 1)  #  [0, 1] noise probability 
        num_pauli = random.randint(1, nqubit)  # random chosen Pauli weight
        indices = random.sample(range(nqubit), num_pauli)  # total number of the randomly chosen sites equals the specific weight
        pauli_string = ','.join(f'{random.choice("xyz")}{i}' for i in sorted(indices))  # generate the string
        
        if pauli_string not in existing_pauli_strings:  
            existing_pauli_strings.add(pauli_string)  
            channelinfo.append([coefficient, pauli_string])  
    sum_p = sum(ele[0] for ele in channelinfo)
    for ele in channelinfo:
        ele[0] = ele[0]/sum_p*magnitude  
        save_pauli_channel(channelinfo, 'pauli_channel.pkl') 
    return channelinfo,magnitude





def calcualte_terms_vu(state,sigma,nqubit):
    '''Calculate 4 upper terms and their summation that contributes to the expression of v_u'''
    d = 2**nqubit
    delta = state- sigma
    I = np.identity(d)
    operator = sigma-I/d    #traceless part
    term1 = np.real((d**2+3*d+4)/(d*(d+2)*(d+3))*(np.trace(delta @ operator))**2)
    term2 = np.real(2*(1+d)**2/(d*(d+2)*(d+3))*np.trace(delta@delta@operator@operator))
    term3 = np.real((d+1)/(d*(d+3))*np.trace(delta@delta)*np.trace(operator@operator))
    term4 = np.real(-2*(1+d)/(d*(d+2)*(d+3))*np.trace(delta@operator@delta@operator))
    vu = term1+term2+term3+term4
    return vu



def calcualte_terms_vstar(state,sigma,nqubit):
    '''Calculate 4 upper terms and their summation that contributes to the expression of v*'''
    d = 2**nqubit
    I = np.identity(d)
    operator = sigma-I/d    #traceless part
    term1 = np.real((d**2+3*d+4)/(d*(d+2)*(d+3))*(np.trace(state @ operator))**2)
    term2 = np.real(2*(1+d)**2/(d*(d+2)*(d+3))*np.trace(state@state@operator@operator))
    term3 = np.real((d+1)/(d*(d+3))*np.trace(state@state)*np.trace(operator@operator))
    term4 = np.real(-2*(1+d)/(d*(d+2)*(d+3))*np.trace(state@operator@state@operator))
    vstar = term1+term2+term3+term4
    return vstar



def compute_for_magnitude(magnow, state, nqubit):

    d = 2**nqubit
    channel, mag = generate_random_pauli_channel(nqubit, magnitude=magnow,channel_num=0)
    state_noised, l1_norm = generate_channel(state, nqubit, channel)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    
    vu = calcualte_terms_vu(state_noised, state, nqubit)
    fidelity = np.real(np.trace(state_noised @ state))
    vstar = calcualte_terms_vstar(state_noised,state,nqubit)

    N_m = 20*np.real(1/(1-fidelity)**2)
    variance = vu + 1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-vstar)
    N_m_term = (-fidelity**2+d*(2*fidelity+1)/(d+2)-vstar)
    thrifty_variance = vstar+1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-vstar)
    amp_thrifty_variance = np.real(thrifty_variance/(1-fidelity)**2)
    amp_variance= np.real(variance/(1-fidelity)**2)
    sample = int(68*np.log(200)*variance*16/(1-fidelity)**2)
    sample_thr = int(68*np.log(200)*thrifty_variance*16/(1-fidelity)**2)
    remain = (-fidelity**2+d*(2*fidelity+1)/(d+2)-vstar)
    return [1 / (1 - np.real(fidelity)), sample,np.real(variance),amp_variance,np.real(thrifty_variance),amp_thrifty_variance,sample_thr,np.real(vu),N_m_term, vstar, remain]



def main(nqubit, k):
    current_time = datetime.now().strftime("%m%d_%H%M")
    folder_path = "xxx" # User defined 

    filename = f'4design_variance_changing_snk_state_n={nqubit}_k={k}_{current_time}.csv'
    full_path = f"{folder_path}\\{filename}"

    N = 40     # Instance number
    log_inv_x = np.linspace(-4.1, 0, N)

    magnitudelist = 10 ** log_inv_x    # A parameter to control the range of infidelity
    
     #target state generation |S_{n,k}\>
    if k != 0:
        cir = Circuit(nqubit)
        cir.h(range(k))
        cir.t(range(k))
        state = cir()
    else:
        state = pq.state.zero_state(nqubit)
    state = _type_transform(state, "density_matrix").numpy()

    with Pool(processes=10) as pool:
        results = pool.starmap(compute_for_magnitude, [(magnow, state, nqubit) for magnow in magnitudelist])


    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['infid^{-1}', 'Sample#','Variance','Variance(amp)','Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty','vu','N_mterm','vstar','remain'])
        writer.writerows(results)

    print(f'Data written to {full_path}')
    return full_path


if __name__ == "__main__":
     #target state generation |S_{n,k}\>
    klist = [1,7]
    for k in klist:
        print(f"Processing k={k}")
        filename = main(nqubit=7, k=k)
        variance_amp_values = []

        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader) 
            index = header.index('Variance(amp)')

            for row in reader:
                try:
                    variance_amp_values.append(float(row[index])) 
                except ValueError:
                    pass 
