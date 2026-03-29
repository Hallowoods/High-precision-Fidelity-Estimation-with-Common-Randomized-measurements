"""


This module provides functions for calculating Delta norms, characteristic functions, cross-characteristic functions with infidelities in random instances. (random single-error Pauli channel, random Pauli channel and random coherent channel)

coherent noise is assumed.

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
import sys, os
sys.path.append(r"c:\Users\Yzy\Desktop\TEMP0820")
import numpy as np
from scipy.linalg import expm

def random_single_qubit_rotation():
"""
Generates a random single-qubit rotation matrix U (2x2 unitary).

The rotation is defined by a random unit vector serving as the rotation axis, 
with the rotation angle sampled uniformly from the interval [0, π].
"""
    phi = np.random.uniform(0, 2*np.pi)    
    costheta = np.random.uniform(-1, 1)    
    sintheta = np.sqrt(1 - costheta**2)
    axis = np.array([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta])
    theta = np.random.uniform(0, np.pi)

    # Pauli 矩阵
    X = np.array([[0,1],[1,0]],dtype=complex)
    Y = np.array([[0,-1j],[1j,0]],dtype=complex)
    Z = np.array([[1,0],[0,-1]],dtype=complex)
    paulis = [X,Y,Z]
    H = axis[0]*X + axis[1]*Y + axis[2]*Z
    U = expm(-1j * theta/2 * H)

    return U
import functools

def generate_random_unitary_noise(nqubit):
"""
Generates random rotation noise across `nqubit`s.

Returns:
    The overall (2^n x 2^n) unitary matrix U representing the noise.
"""

    single_qubit_rotations = [random_single_qubit_rotation() for _ in range(nqubit)]
    U = functools.reduce(np.kron, single_qubit_rotations)
    return U
def apply_unitary_noise(rho, U):
    return U @ rho @ U.conj().T
def generate_pauli_group(nqubit):
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

    with open(json_path, "w") as f:
        json.dump(pauli_group, f, indent=4)
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

def generate_random_state(nqubit,type='random',k=0):
    if type == 'random':
        state = pq.state.random_state(nqubit)
    elif type == 'w_state':
        state = pq.state.w_state(nqubit)
    elif type == 'ghz_state':
        state = pq.state.ghz_state(nqubit)
    elif type =='bell_state':
        state = pq.state.bell_state(nqubit)
    elif type == 'zero_state':
        state = pq.state.zero_state(nqubit)
    elif type == 's_nk_state':
        if k !=0:
            cir = Circuit(nqubit)
            cir.h(range(k))
            cir.t(range(k))
            state = cir()
        else:
            state = pq.state.zero_state(nqubit)
    state = _type_transform(state, "density_matrix").numpy()
    np.save('state.npy', state)      
    return state


def caculate_characters_CRM(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
   
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    delta = state-sigma
    d = 2**nqubit
    I = np.identity(d)
    delta_vec = []
    sigma_vec = []
    cross_vec = []
    ob = sigma-I/d
    for i in range(len(pauli_group)):

        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        delta_vec.append(np.real(np.trace(delta@ele[1])))  
        sigma_vec.append(np.real(np.trace(ob@ele[1])))
        cross_vec.append(np.real(np.trace(delta@ele[1]@ob@ele[1])))
    delta_vec = np.array(delta_vec)
    sigma_vec = np.array(sigma_vec)
    cross_vec = np.array(cross_vec)
    cross_character = np.sum(delta_vec*sigma_vec*cross_vec)
    character = np.dot(delta_vec**2,sigma_vec**2)
    overlap = np.trace(delta@state)
    return character,cross_character,overlap


def caculate_characters_THR(rho,sigma,nqubit):
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
        state_vec.append(np.real(np.trace(rho@ele[1])))
        ob_vec.append(np.real(np.trace(ob@ele[1])))
        cross_vec.append(np.real(np.trace(rho@ele[1]@ob@ele[1])))
    ob_vec = np.array(ob_vec)
    cross_vec = np.array(cross_vec)
    state_vec = np.array(state_vec)
    cross_character = np.sum(state_vec*ob_vec*cross_vec)

    character = np.dot(state_vec**2,ob_vec**2)
    return character,cross_character



def compute_for_magnitude(channel, state, nqubit):
    d = 2**nqubit
    unitary  = generate_random_unitary_noise(nqubit)
    state_noised = apply_unitary_noise(state,unitary)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    
    character, cross_character,overlap = caculate_characters_CRM(state_noised, state, nqubit)
    character2, cross_character2 = caculate_characters_THR(state_noised, state, nqubit)

    return [character, cross_character,character2, cross_character2]


import os
import csv
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm   # pip install tqdm


def generate_haar_state(nqubit):
    """generate a haar random pure state"""
    state = pq.state.random_state(nqubit)
    state_dm = _type_transform(state, "density_matrix").numpy()
    return state_dm


def analyze_rho_sigma(rho, sigma,nqubit):
  """calculate the cross charcteristic functions,  twisted cross charcteristic functionsDelta with inverse infidelity"""
    Delta = rho - sigma
    eigvals = np.linalg.eigvalsh(Delta)  
    min_eig = np.min(eigvals)
    min_eig_term = 4 * (min_eig ** 2)

    norm2_sq = np.trace(Delta @ Delta).real
    overlap = np.trace(rho @ sigma).real
    infid = (1.0 - overlap) 
    infid2 = infid**2
    character, cross_character,overlap = caculate_characters_CRM(rho,sigma,nqubit)
    character2, cross_character2 = caculate_characters_THR(rho, sigma, nqubit)
    return infid,infid2,min_eig_term, norm2_sq,character, cross_character,character2, cross_character2


def worker(args):
    idx, nqubit, chan_num = args
    sigma = generate_haar_state(nqubit)
    unitary  = generate_random_unitary_noise(nqubit)
    rho = apply_unitary_noise(sigma,unitary)
    rho = _type_transform(rho, "density_matrix").numpy()
    infid, infid2, min_eig_term, norm2_sq, character, cross_character, character2, cross_character2 = analyze_rho_sigma(
        rho, sigma, nqubit
    )
    return [
        idx,
        infid,
        infid2,
        min_eig_term,
        norm2_sq,
        np.real(character),
        np.real(cross_character),
        np.real(character2),
        np.real(cross_character2),
    ]



def main(nqubit=2, num_samples=2000, nproc=10,num_chan=0):
    out_dir = rf"xxxx"         # User defined
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"data_statstic_n={nqubit}_{timestamp}_unitary_num_sample_{num_samples}.csv")
    args_list = [(i,nqubit,num_chan) for i in range(num_samples)]

    results = []
    with Pool(processes=nproc) as pool:
        for res in tqdm(pool.imap(worker, args_list), total=num_samples, desc=f"n={nqubit}"):

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "infid", "infid2", "4*min_eig^2", "||Delta||_2^2",'character','cross_character','character2', 'cross_character2'])

        writer.writerows(results)


if __name__ == "__main__":
    nqubit_list = [1,2,3,4]
    num_samples = 20000          # Instance number
    magnitude = 1
    nproc = 7
    chan_num = 1  

    for nqubit in nqubit_list:
        main(nqubit=nqubit, num_samples=num_samples, nproc=nproc,num_chan=chan_num)




