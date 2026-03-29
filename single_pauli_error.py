"""
single_pauli_error.py

This module provides functions for calculating (cross) characteristic functions for single-qubit pauli error in Clifford/4-design CRM (fig. 4).
Author: YZY
"""

import time
import numpy as np
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
from paddle_quantum.state import State
from scipy.interpolate import interp1d

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


def generate_fixed_pauli_channel(nqubit, magnitude=0.5):
    """Generate a fixed Pauli channel: the Z-error on the first site"""
    if nqubit < 1:
        raise ValueError("nqubit must be at least 1")
    
    channelinfo = []
    coefficient = magnitude  
    pauli_string = f'z{nqubit-1}'     
    
    channelinfo.append([coefficient, pauli_string])

    
    save_pauli_channel(channelinfo, 'pauli_channel.pkl')  
    return channelinfo, magnitude




def caculate_characters_CRM(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
   
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    delta = state-sigma
    delta_vec = []
    sigma_vec = []
    cross_vec = []
    for i in range(len(pauli_group)):
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
   
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    sigma_vec = []
    cross_vec = []
    state_vec = []
    d = 2**nqubit
    I = np.identity(d)
    ob = sigma-I/d
    ob_vec = []
    for i in range(len(pauli_group)):
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        state_vec.append(np.trace(state@ele[1]))
        ob_vec.append(np.trace(ob@ele[1]))
        cross_vec.append(np.trace(state@ele[1]@ob@ele[1]))
    ob_vec = np.array(ob_vec)
    cross_vec = np.array(cross_vec)
    state_vec = np.array(state_vec)
    cross_character = np.sum(state_vec*ob_vec*cross_vec)

    character = np.dot(state_vec**2,ob_vec**2)
    return character,cross_character


import math
def compute_for_magnitude(m,magnow, nqubit,N_m =100):
    print(rf'current m:{m}')
    state = generate_random_sector_state(nqubit=nqubit,m=m)
    state = _type_transform(state, "density_matrix").numpy()

    d = 2**nqubit
    channel, mag = generate_fixed_pauli_channel(nqubit, magnitude=0.5)
    state_noised, l1_norm = generate_channel(state, nqubit, channel)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()

    delta = state_noised-state
    character, cross_character = caculate_characters_CRM(state_noised, state, nqubit)
    fidelity = np.trace(state_noised @ state)
    variance_u = ((d+1)/(d*(d+2)))*(np.real(character)+np.real(cross_character))-(1 - np.real(fidelity))**2/(d+2)
    character2, cross_character2 = caculate_characters_THR(state_noised,state,nqubit)
    amp_character = np.real(character/(1-fidelity)**2)
    amp_cross = np.real(cross_character/(1-fidelity)**2)


    variance_0 = (d+1)/(d*(d+2))*(np.real(character2)+np.real(cross_character2))-(np.trace(delta@state))**2/(d+2)
    if (1 - fidelity) != 0:
        N_m = math.ceil(10 / (1 - fidelity) ** 2)
    else:
        N_m = 1e12  
    variance = variance_u

    thrifty_variance = variance_0+1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    amp_thrifty_variance = np.real(thrifty_variance/(1-fidelity)**2)
    amp_variance= np.real(variance*16/(1-fidelity)**2)
    sample = int(68*np.log(200)*variance*16/(1-fidelity)**2)
    sample_thr = int(68*np.log(200)*thrifty_variance/(1-fidelity)**2)
    r = 1/(m**4*(1-m**2)**2)

    

    return [1/m**4*(1-m**2),m,r,1 / (1 - np.real(fidelity)), np.real(character), np.real(cross_character), sample,np.real(variance),amp_variance,amp_character,amp_cross,np.real(thrifty_variance),amp_thrifty_variance,sample_thr]


def compute_for_magnitude_4design(m,magnow, nqubit,N_m =100):

    state = generate_random_sector_state(nqubit=nqubit,m=m)
    state = _type_transform(state, "density_matrix").numpy()

    d = 2**nqubit
    channel, mag = generate_fixed_pauli_channel(nqubit, magnitude=0.5)
    state_noised, l1_norm = generate_channel(state, nqubit, channel)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    delta = state_noised-state
    fidelity = np.trace(state_noised @ state)
    I = np.identity(d)
    operator = state-I/d    #traceless part
    term1 = np.real((d**2+3*d+4)/(d*(d+2)*(d+3))*(np.trace(delta @ operator))**2)
    term2 = np.real(2*(1+d)**2/(d*(d+2)*(d+3))*np.trace(delta@delta@operator@operator))
    term3 = np.real((d+1)/(d*(d+3))*np.trace(delta@delta)*np.trace(operator@operator))
    term4 = np.real(-2*(1+d)/(d*(d+2)*(d+3))*np.trace(delta@operator@delta@operator))
    variance_u = term1+term2+term3+term4
    r = 1/(m**4*(1-m**2)**2)
    variance = variance_u 
    amp_variance= np.real(variance*16/(1-fidelity)**2)
    sample = int(68*np.log(200)*variance*16/(1-fidelity)**2)

    #thrifty ones
    
    return [1/m**4*(1-m**2),m,r,1 / (1 - np.real(fidelity)),'nan', 'nan', sample,np.real(variance),amp_variance,'nan','nan','nan','nan','nan']


def generate_random_sector_state(nqubit, m):
    """Generate the random target states described in Appendix E in the paper arXiv:2511.22509"""
    if abs(m) > 1:
        raise ValueError("m must satisfy |m| ≤ 1")

    dim = 2 ** nqubit
    state_vector = np.zeros((dim, 1), dtype=complex)

    indices_0 = [i for i in range(dim) if (i & 1) == 0]
    num_sector_0 = len(indices_0)

    real_part_0 = np.random.randn(num_sector_0)
    imag_part_0 = np.random.randn(num_sector_0)
    random_amplitudes_0 = real_part_0 + 1j * imag_part_0


    norm_0 = np.linalg.norm(random_amplitudes_0)
    random_amplitudes_0 = random_amplitudes_0 / norm_0 * np.sqrt(m**2)

    indices_1 = [i for i in range(dim) if (i & 1) == 1]
    num_sector_1 = len(indices_1)


    real_part_1 = np.random.randn(num_sector_1)
    imag_part_1 = np.random.randn(num_sector_1)
    random_amplitudes_1 = real_part_1 + 1j * imag_part_1

    
    norm_1 = np.linalg.norm(random_amplitudes_1)
    random_amplitudes_1 = random_amplitudes_1 / norm_1 * np.sqrt(1 - abs(m)**2)


    for idx, amp in zip(indices_0, random_amplitudes_0):
        state_vector[idx, 0] = amp
    for idx, amp in zip(indices_1, random_amplitudes_1):
        state_vector[idx, 0] = amp

    state = State(state_vector)

    return state

def main(nqubit,fdesign=True):
    current_time = datetime.now().strftime("%m%d_%H%M")
    N_m = 10000000
    if fdesign:

        folder_path = rf"xxxx"    # User defined


        filename = f'one_error_shadow_norm_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'
        full_path = f"{folder_path}\\{filename}"

        mlist = np.exp(np.linspace(-5, 0, 10))        # A parameter to control the range of infidelity


        with Pool(processes=10) as pool: 
            results = pool.starmap(compute_for_magnitude_4design, [(m,0.5, nqubit,N_m) for m in mlist])

        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['1/1-phi0',
            'm','r','infid^{-1}', 'Character', 'Cross_Character', 'Sample#','Variance','Variance(amp)',"Character(amp)","Cross_Character(amp)",'Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty'])
            writer.writerows(results)

        print(f'Data written to {full_path}')
        return filename
    else:
        folder_path = rf"yyyy"        # User defined
        filename = f'one_error_shadow_norm_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'
        full_path = f"{folder_path}\\{filename}"

        mlist = np.exp(np.linspace(-5, 0, 10))         # A parameter to control the range of infidelity


        with Pool(processes=10) as pool:  
            results = pool.starmap(compute_for_magnitude, [(m,0.5, nqubit,N_m) for m in mlist])


        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['1/1-phi0',
            'm','r','infid^{-1}', 'Character', 'Cross_Character', 'Sample#','Variance','Variance(amp)',"Character(amp)","Cross_Character(amp)",'Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty'])
            writer.writerows(results)

        print(f'Data written to {full_path}')
        return full_path

if __name__ == "__main__":
    full_path = main(nqubit=7,fdesign = True)# 画图

