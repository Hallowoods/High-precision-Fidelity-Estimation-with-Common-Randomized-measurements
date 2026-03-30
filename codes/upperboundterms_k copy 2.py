"""
upperboundterms_k.py

This module provides functions for calculating upper bound terms(including cross-characteristic, characteristic, variance, and their amplified version, sample complexity) for snk states with different k. Both for CRM and THR protocol.

Pauli noise is assumed.

Author: YZY
Date: 2025-04-01
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
from datetime import datetime  
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





def caculate_characters_CRM(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
    d = 2**nqubit
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
    delta = state-sigma
    sigma -= np.identity(d)/d
    delta_vec = []
    sigma_vec = []
    cross_vec = []
    for i in range(len(pauli_group)):
        print(i)
        print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        delta_vec.append(np.trace(delta@ele[1]))  
        sigma_vec.append(np.trace(sigma@ele[1]))
        cross_vec.append(np.trace(delta@ele[1]@sigma@ele[1]))
    delta_vec = np.array(delta_vec)
    sigma_vec = np.array(sigma_vec)
    cross_vec = np.array(cross_vec)

    
    cross_character = np.sum(delta_vec*sigma_vec*cross_vec)

    character = np.dot(delta_vec**2,sigma_vec**2)
    return character,cross_character


def caculate_characters_THR(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
    d = 2**nqubit
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
    state_vec = []
    cross_vec = []
    traceles_sigma_vec = []
    traceless_sigma = sigma-np.identity(d)/d
    for i in range(len(pauli_group)):
        print(i)
        print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        # pauli =pq.qinfo.pauli_str_to_matrix([[1,pauli_group[i][1]]],nqubit)
        # print(pauli) 
        state_vec.append(np.trace(state@ele[1]))
        traceles_sigma_vec.append(np.trace(traceless_sigma@ele[1]))
        cross_vec.append(np.trace(state@ele[1]@traceless_sigma@ele[1]))
    state_vec = np.array(state_vec)
    traceles_sigma_vec = np.array(traceles_sigma_vec)
    cross_vec = np.array(cross_vec)

    
    cross_character = np.sum(traceles_sigma_vec*state_vec*cross_vec)

    character = np.dot(state_vec**2,traceles_sigma_vec**2)
    return character,cross_character



def compute_for_magnitude(magnow, state, nqubit,N_m =100):
    d = 2**nqubit
    channel, mag = generate_random_pauli_channel(nqubit, magnitude=magnow,channel_num=3)
    state_noised, l1_norm = generate_channel(state, nqubit, channel)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    
    character, cross_character = caculate_characters_CRM(state_noised, state, nqubit)
    fidelity = np.trace(state_noised @ state)
    variance_u = ((d+1)/(d*(d+2)))*(np.real(character)+np.real(cross_character))-(1 - np.real(fidelity))**2/(d+2)
    character2, cross_character2 = caculate_characters_THR(state_noised,state,nqubit)
    amp_character = np.real(character/(1-fidelity)**2)*16
    amp_cross = np.real(cross_character/(1-fidelity)**2)*16


    variance_0 = (d+1)/(d*(d+2))*(np.real(character2)+np.real(cross_character2))-(1 - np.real(fidelity))**2/(d+2)

    variance = variance_u + 1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    thrifty_variance = variance_0+1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    amp_thrifty_variance = np.real(16*thrifty_variance/(1-fidelity)**2)
    amp_variance= np.real(variance/(1-fidelity)**2)*16
    sample = int(16*136*np.log(200)*variance/(1-fidelity)**2)
    sample_thr = int(16*136*np.log(200)*thrifty_variance/(1-fidelity)**2)
    return [1 / (1 - np.real(fidelity)), np.real(character), np.real(cross_character), sample,np.real(variance),amp_variance,amp_character,amp_cross,np.real(thrifty_variance),amp_thrifty_variance,sample_thr]



def main_varying_Nm(nqubit, k, magnitude=0.01):
    start_time = time.time()
    current_time = datetime.now().strftime("%m%d_%H%M")
    folder_path = rf"xxxx"    # User defined
    filename = f'shadow_norm_haar_state_nqubit={nqubit}_varyNm_mag={magnitude}_{current_time}.csv'
    full_path = f"{folder_path}\\{filename}"
    
    # log-spaced R of circuit reusing
    N_M_list = np.unique(np.logspace(0, 7, num=70, dtype=int))  
    
    state = pq.state.random_state(nqubit)
    state = _type_transform(state, "density_matrix").numpy()

   
    channel, _ = generate_random_pauli_channel(nqubit, magnitude=magnitude, channel_num=3)
    state_noised, l1_norm = generate_channel(state, nqubit, channel)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()


    character, cross_character = caculate_characters_CRM(state_noised, state, nqubit)
    character2, cross_character2 = caculate_characters_THR(state_noised,state, nqubit)
    fidelity = np.real(np.trace(state_noised @ state))
    d = 2**nqubit
    variance_0 = (d+1)/(d*(d+2))*(np.real(character2)+np.real(cross_character2))-(fidelity-1/d)**2/(d+2)

    results = []

    for N_m in N_M_list:
        variance_u = ((d+1)/(d*(d+2)))*(np.real(character)+np.real(cross_character))-(1 - fidelity)**2/(d+2)
        variance = variance_u + 1/N_m*(-fidelity**2 + d*(2*fidelity+1)/(d+2) - variance_0)
        thrifty_variance = variance_0 + 1/N_m*(-fidelity**2 + d*(2*fidelity+1)/(d+2) - variance_0)
        print(('term',-fidelity**2 + d*(2*fidelity+1)/(d+2) -variance_0))
        amp_variance = 16 * variance / (1 - fidelity)**2
        amp_thrifty_variance = 16 * thrifty_variance / (1 - fidelity)**2
        infid_inv = 1/(1 - fidelity)
        sample = int(16 * 68 * np.log(200) * variance / (1 - fidelity)**2)
        sample_thr = int(16 * 68 * np.log(200) * thrifty_variance / (1 - fidelity)**2)

        results.append([N_m, variance, amp_variance, sample, thrifty_variance, amp_thrifty_variance, sample_thr,infid_inv])
    
    # 写入 CSV
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N_M', 'Variance', 'Variance(amp)', 'Sample#', 'Variance_thrifty', 'Variance_thrifty(amp)', 'Sample#_thrifty','infid^{-1}'])
        writer.writerows(results)

    print(f"结果写入 {full_path}")
    return full_path


if __name__ == "__main__":
    k = 0
    full_path = main_varying_Nm(nqubit=7, k=k)# 画

    variance_amp_values = []

    with open(full_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  

        index = header.index('Variance(amp)')

        for row in reader:
            try:
                variance_amp_values.append(float(row[index]))  
            except ValueError:
                pass  

