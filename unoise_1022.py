"""
unitary_noise_upperboundterms.py

This module provides functions for calculating upper bound terms(including cross-characteristic, characteristic, variance, and their amplified version, sample complexity) for snk states with different k. Both for CRM and THR protocol.

Local unitary noise is assumed.

The code to caulate the data for CRM/THR based on Clifford measurements in Fig. 2. The columns 'Sample#', 'Sample#_thrifty', 'infid^{-1}' is the result.
Author: YZY
"""

import math
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
from paddle_quantum.channel.common import MixedUnitaryChannel


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


import numpy as np
def generate_theta_vector(nqubit, mag):
    """
    Generates a vector of `nqubit` theta values. 
    Each theta_i = arccos(sqrt(v_i)), subject to v_i >= 0 and sum(v_i) = mag.
    Returns a vector of 2 * theta_i.
    """
    # Step 1: Randomly generate `nqubit` numbers in the range [0, 1)
    v = np.random.rand(nqubit)
    beta = np.log2(mag)
    
    # Step 2: Normalize the vector so that sum(v) = mag (using beta as the scaling factor)
    v = v / np.sum(v) * beta
    
    # Step 3: Take the square root of each component and then apply arccos
    theta = np.arccos(2 * 2**v - 1)

    # Step 4: Return 2 * theta
    return 2 * theta

def generate_random_unitary_channel_rotation2(nqubit, mag, num_sites=1):
    """Generate random local rotations on random sites, the noise is parametrized by "mag". This returns a channel info indicating the chosen sites, angles and axis"""
    channel_info = []
    sites = random.sample(range(0, nqubit), num_sites)
    thetalist = generate_theta_vector(num_sites, mag)
    for i  in range(len(sites)):
        site = sites[i]
        theta = thetalist[i]
        vec = np.random.randn(3)
        channel_info.append([site, theta, axis.tolist()])  

    filename = 'unitary_channel(local_rotation).pkl'
    with open(filename, 'wb') as file:
        pickle.dump(channel_info, file)
    
    return channel_info    


def axis_to_u3_params(theta, axis):
    """
    Convert rotation (axis, theta) -> U3 parameters (θ', φ, λ)
    so that U3(θ', φ, λ) = exp(-i theta/2 * (n·σ))
    consistent with Paddle Quantum's U3 definition:
        U3(θ, φ, λ) = [[cos θ/2, -exp(i λ) sin θ/2],
                        [exp(i φ) sin θ/2, exp(i(φ+λ)) cos θ/2]]
    
    Args:
        theta: rotation angle
        axis: array-like, 3D unit vector of rotation axis [nx, ny, nz]
    
    Returns:
        tuple (theta_u3, phi, lam)
    """
    axis = np.array(axis, dtype=float)
    if np.linalg.norm(axis) == 0:
        raise ValueError("Rotation axis cannot be zero vector.")
    axis = axis / np.linalg.norm(axis)
    nx, ny, nz = axis

    # Build rotation matrix R = exp(-i theta/2 * n·σ)
    cos_term = np.cos(theta/2)
    sin_term = np.sin(theta/2)
    R = np.array([[cos_term - 1j * nz * sin_term, -1j * (nx - 1j * ny) * sin_term],
                  [-1j * (nx + 1j * ny) * sin_term, cos_term + 1j * nz * sin_term]])

    # Extract U3 parameters from R
    theta_u3 = 2 * np.arccos(np.abs(R[0,0]))
    
    # Avoid division by zero
    if np.sin(theta_u3/2) < 1e-12:
        phi = 0.0
        lam = np.angle(R[1,1]) - np.angle(R[0,0])
    else:
        phi = np.angle(R[1,0])
        lam = np.angle(-R[0,1])

    return [theta_u3, phi, lam]

def generate_unitary_channel_rotation2(state, nqubit, channelinfo):
    """Applies coherent noise to the target quantum state using the previously generated random `channel_info`."""
    cir = Circuit(nqubit)
    state = to_state(state, backend=Backend.DensityMatrix)
    prodtest = 1
    statep  = _type_transform(state, "density_matrix").numpy()
    for info in channelinfo:
        site = info[0]
        theta = info[1]
        axis = info[2]
        print('axis::',axis,'   theta:',theta)
        u3_params = axis_to_u3_params(theta, axis)
        cir.u3([site], param=u3_params)  
    state_after = cir.forward(state)
    state_afterp = _type_transform(state_after, "density_matrix").numpy()
    print('fid_inst:',np.trace(statep@state_afterp),'  fid_theory:',prodtest)
    return state_after

def caculate_characters_CRM(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    delta = state-sigma
    delta_vec = []
    ob_vec = []
    cross_vec=[]
    d= 2**nqubit
    I = np.identity(d)
    ob = sigma-I/d
    for i in range(len(pauli_group)):
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        delta_vec.append(np.trace(delta@ele[1]))  
        ob_vec.append(np.trace(ob@ele[1]))
        cross_vec.append(np.trace(delta@ele[1]@ob@ele[1]))
    delta_vec = np.array(delta_vec)
    ob_vec = np.array(ob_vec)
    cross_vec = np.array(cross_vec)

    
    cross_character = np.sum(delta_vec*ob_vec*cross_vec)

    character = np.dot(delta_vec**2,ob_vec**2)
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

    


def compute_for_magnitude(magnow ,state, nqubit, noisetype='rotation', N_m=None, errorsites=1):
    d = 2**nqubit
    # Generate noised state
    if noisetype == 'rotation':
        channel = generate_random_unitary_channel_rotation2(nqubit, magnow, num_sites=errorsites)
        state_noised = generate_unitary_channel_rotation2(state, nqubit, channel)

    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    
    # fidelity
    fidelity = np.real(np.trace(state_noised @ state))     #This is what really matters

    # Circuit Reusing Number R
    if (1 - fidelity) != 0:
        N_m = math.ceil(20 / (1 - fidelity) ** 2)
    else:
        N_m = 1e20  # avoid being too large 

    # CRM
    character, cross_character = caculate_characters_CRM(state_noised, state, nqubit)
    variance_u = ((d+1)/(d*(d+2))) * (np.real(character) + np.real(cross_character)) - (1 - fidelity)**2 / (d+2)

    # THR
    character2, cross_character2 = caculate_characters_THR(state_noised, state, nqubit)
    variance_0 = (d+1)/(d*(d+2)) * (np.real(character2) + np.real(cross_character2)) - (fidelity - 1/d)**2 / (d+2)

    amp_character = np.real(character / (1 - fidelity) ** 2)
    amp_cross = np.real(cross_character / (1 - fidelity) ** 2)

    variance = variance_u + 1 / N_m * (-fidelity**2 + d*(2*fidelity + 1) / (d+2) - variance_0)
    thrifty_variance = variance_0 + 1 / N_m * (-fidelity**2 + d*(2*fidelity + 1) / (d+2) - variance_0)

    amp_variance = np.real(variance / (1 - fidelity) ** 2)
    amp_thrifty_variance = np.real(thrifty_variance / (1 - fidelity) ** 2)

    sample = int(68 * np.log(200) * 16* variance / (1 - fidelity) ** 2) if (1 - fidelity) ** 2 != 0 else -1    #This is what really matters

    
    sample_thr = int(68 * np.log(200) * 16*thrifty_variance / (1 - fidelity) ** 2) if (1 - fidelity) ** 2 != 0 else -1    #This is what really matters
    partone_crm = np.real(variance_u)
    parttwo  = np.real(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    partone_thr = np.real(variance_0)
    print(r'infid^-1={}'.format(1 / (1 - np.real(fidelity))))
    return [
        1 / (1 - fidelity),
        np.real(character),
        np.real(cross_character),
        sample,
        np.real(variance),
        amp_variance,
        amp_character,
        amp_cross,
        np.real(thrifty_variance),
        amp_thrifty_variance,
        sample_thr,N_m,partone_crm,parttwo,partone_thr
    ]



def main(nqubit, k,errorsites,theta,statetype,noisetype='rotation',):

   
    folder_path = r"xxxxx"    # User defined
    current_time = datetime.now().strftime("%m%d_%H%M")
    N_m = None
    # N_m = 10000

    noisetype = 'rotation'
    if statetype == 'snk':
        filename = f'Unoise_nqubit={nqubit}_k={k}_NM={N_m}_errorsites={errorsites}_type_{noisetype}_{current_time}.csv'
        if k != 0:
            cir = Circuit(nqubit)
            cir.h(range(k))
            cir.t(range(k))
            state = cir()
        else:
            state = pq.state.zero_state(nqubit)

    N = 200     # Instance number

    bx = np.linspace(0,3,N)
    magnitudelist = 1-10**(-bx)        # A parameter to control the range of infidelity
    with Pool(processes=10) as pool:  
        results = pool.starmap(compute_for_magnitude, [(magnitudelist[i], state, nqubit,noisetype,N_m,errorsites) for i in range(len(magnitudelist)-1)])

    # 统一写入 CSV 文件
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['infid^{-1}', f'{k}_Character', f'{k}_Cross_Character', 'Sample#','Variance','Variance(amp)',f"{k}_Character(amp)",f"{k}_Cross_Character(amp)",'Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty', 'N_m', 'PartOne_CRM', 'PartTwo', 'PartOne_THR'])
        writer.writerows(results)

    print(f'Data written to {full_path}')
    

    return full_path

if __name__ == "__main__":
    #target state generation |S_{n,k}\>
    klist = [1,7]
    n = 7
    for k in klist:
        filename = main(nqubit=n, k=k,theta=0,errorsites=n,noisetype='rotation',statetype='snk')#
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


