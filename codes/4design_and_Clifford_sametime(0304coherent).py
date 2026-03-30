"""

This module provides functions for calculating upper bound terms(including cross-characteristic, characteristic, variance, and their amplified version, sample complexity) for snk states with different k. Both for CRM and THR protocol.

coherent noise is assumed.

The code to caulate the data for CRM/THR based on 4-design/Clifford measurements in Fig. 2. The columns 'Sample#', 'Sample#_thrifty', 'infid^{-1}' is the result.
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

def generate_unitary_channel_rotation2(state1, state2,nqubit, channelinfo):
    cir = Circuit(nqubit)
    state1 = to_state(state1, backend=Backend.DensityMatrix)
    state2 = to_state(state2, backend=Backend.DensityMatrix)
    prodtest = 1
    statep1  = _type_transform(state1, "density_matrix").numpy()
    statep2  = _type_transform(state2, "density_matrix").numpy()
    for info in channelinfo:
        site = info[0]
        theta = info[1]
        axis = info[2]
        print('axis::',axis,'   theta:',theta)
        u3_params = axis_to_u3_params(theta, axis)
        cir.u3([site], param=u3_params)  # 使用 u3 门
    state_after1 = cir.forward(state1)
    state_after2 = cir.forward(state2)
    # /statep  = _type_transform(state, "density_matrix").numpy()
    state_afterp1 = _type_transform(state_after1, "density_matrix").numpy()
    state_afterp2 = _type_transform(state_after2, "density_matrix").numpy()
    return state_after1,state_after2




def save_pauli_channel(channelinfo, filename):
    '''write-in Pauli Channel file'''
    with open(filename, 'wb') as file:
        pickle.dump(channelinfo, file)  # 将 channelinfo 保存到文件中

def load_pauli_channel(filename):
    '''Read the Pauli Channel file'''

    with open(filename, 'rb') as file:
        channelinfo = pickle.load(file)  # 从文件中读取 channelinfo
    return channelinfo




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




def caculate_characters_CRM(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
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





def compute_for_magnitude(magnow, state1,state2, nqubit,noisetype='rotation',errorsites=1):
    d = 2**nqubit
    if noisetype =='rotation':
        channel = generate_random_unitary_channel_rotation2(nqubit,magnow,num_sites=errorsites)
        state_noised1,state_noised2 = generate_unitary_channel_rotation2(state1,state2,nqubit,channel)
    state_noised1 = _type_transform(state_noised1, "density_matrix").numpy()
    state_noised2 = _type_transform(state_noised2, "density_matrix").numpy()



    vu1 = calcualte_terms_vu(state_noised1, state1, nqubit)
    fidelity1 = np.real(np.trace(state_noised1 @ state1))
    vstar1 = calcualte_terms_vstar(state_noised1,state1,nqubit)

    vu2 = calcualte_terms_vu(state_noised2, state2, nqubit)
    fidelity2 = np.real(np.trace(state_noised2 @ state2))
    vstar2 = calcualte_terms_vstar(state_noised2,state2,nqubit)

    N_m1 = 20*np.real(1/(1-fidelity1)**2)
    N_m2 = 20*np.real(1/(1-fidelity2)**2)



    variance_41= vu1 + 1/N_m1*(-fidelity1**2+d*(2*fidelity1+1)/(d+2)-vstar1)
    N_m_term1 = (-fidelity1**2+d*(2*fidelity1+1)/(d+2)-vstar1)
    thrifty_variance_41 = vstar1+1/N_m1*(-fidelity1**2+d*(2*fidelity1+1)/(d+2)-vstar1)
    sample_41 = int(68*np.log(200)*variance_41*16/(1-fidelity1)**2)
    sample_thr_41 = int(68*np.log(200)*thrifty_variance_41*16/(1-fidelity1)**2)
    remain1 = (-fidelity1**2+d*(2*fidelity1+1)/(d+2)-vstar1)

    variance_42= vu2 + 1/N_m2*(-fidelity2**2+d*(2*fidelity2+1)/(d+2)-vstar2)
    N_m_term2 = (-fidelity2**2+d*(2*fidelity2+1)/(d+2)-vstar2)
    thrifty_variance_42 = vstar2+1/N_m2*(-fidelity2**2+d*(2*fidelity2+1)/(d+2)-vstar2)
    sample_42 = int(68*np.log(200)*variance_42*16/(1-fidelity2)**2)
    sample_thr_42 = int(68*np.log(200)*thrifty_variance_42*16/(1-fidelity2)**2)
    remain2 = (-fidelity2**2+d*(2*fidelity2+1)/(d+2)-vstar2)

    # CRM
    character1, cross_character1 = caculate_characters_CRM(state_noised1, state1, nqubit)
    variance_u1 = ((d+1)/(d*(d+2))) * (np.real(character1) + np.real(cross_character1)) - (1 - fidelity1)**2 / (d+2)

    character2, cross_character2 = caculate_characters_CRM(state_noised2, state2, nqubit)
    variance_u2 = ((d+1)/(d*(d+2))) * (np.real(character2) + np.real(cross_character2)) - (1 - fidelity2)**2 / (d+2)


    # THR
    character21, cross_character21 = caculate_characters_THR(state_noised1, state1, nqubit)
    variance_01 = (d+1)/(d*(d+2)) * (np.real(character21) + np.real(cross_character21)) - (fidelity1 - 1/d)**2 / (d+2)

    character22, cross_character22 = caculate_characters_THR(state_noised2, state2, nqubit)
    variance_02 = (d+1)/(d*(d+2)) * (np.real(character22) + np.real(cross_character22)) - (fidelity2 - 1/d)**2 / (d+2)


    variance1 = variance_u1 + 1 / N_m1 * (-fidelity1**2 + d*(2*fidelity1 + 1) / (d+2) - variance_01)
    thrifty_variance1 = variance_01 + 1 / N_m1 * (-fidelity1**2 + d*(2*fidelity1 + 1) / (d+2) - variance_01)

    variance2 = variance_u2 + 1 / N_m2 * (-fidelity2**2 + d*(2*fidelity2 + 1) / (d+2) - variance_02)
    thrifty_variance2 = variance_02 + 1 / N_m2 * (-fidelity2**2 + d*(2*fidelity2 + 1) / (d+2) - variance_02)


    sample1 = int(68 * np.log(200) * 16* variance1 / (1 - fidelity1) ** 2) if (1 - fidelity1) ** 2 != 0 else -1
    sample_thr1 = int(68 * np.log(200) * 16*thrifty_variance1 / (1 - fidelity1) ** 2) if (1 - fidelity1) ** 2 != 0 else -1
    partone_crm1 = np.real(variance_u1)
    parttwo1  = np.real(-fidelity1**2+d*(2*fidelity1+1)/(d+2)-variance_01)
    partone_thr1 = np.real(variance_01)

    sample2 = int(68 * np.log(200) * 16* variance2 / (1 - fidelity2) ** 2) if (1 - fidelity2) ** 2 != 0 else -1
    sample_thr2 = int(68 * np.log(200) * 16*thrifty_variance2 / (1 - fidelity2) ** 2) if (1 - fidelity2) ** 2 != 0 else -1
    partone_crm2 = np.real(variance_u2)
    parttwo2  = np.real(-fidelity2**2+d*(2*fidelity2+1)/(d+2)-variance_02)
    partone_thr2 = np.real(variance_02)


    
    results1 = [1 / (1 - np.real(fidelity1)), sample_41,sample_thr_41,np.real(vu1),N_m_term1,vstar1,remain1,np.real(character1),
        np.real(cross_character1),
        sample1,sample_thr1,N_m1,partone_crm1,parttwo1,partone_thr1]
    results2 = [1 / (1 - np.real(fidelity2)), sample_42,sample_thr_42,np.real(vu2),N_m_term2,vstar2,remain2,np.real(character2),
        np.real(cross_character2),
        sample2,sample_thr2,N_m1,partone_crm2,parttwo2,partone_thr2]
    return (results1,results2)



def main(nqubit, errorsites,theta,statetype,noisetype='rotation',):
    
    current_time = datetime.now().strftime("%m%d_%H%M")
    noisetype = 'rotation'
    folder_path = rf"xxx"    # User defined

    filename1 = f'same_axis_changing_snk_state_n={nqubit}_k=1_errorsites={errorsites}_{current_time}.csv'
    full_path1 = f"{folder_path}\\{filename1}"

    filename2 = f'same_axis_changing_snk_state_n={nqubit}_k=7_errorsites={errorsites}_{current_time}.csv'
    full_path2 = f"{folder_path}\\{filename2}"

    N = 550
    bx = np.linspace(0,4,N)
    magnitudelist = 1-10**(-bx)



    cir1 = Circuit(nqubit)
    cir1.h(range(1))
    cir1.t(range(1))
    state1 = cir1()
    state1 = _type_transform(state1, "density_matrix").numpy()

    cir7 = Circuit(nqubit)
    cir7.h(range(7))
    cir7.t(range(7))
    state7 = cir7()
    state7 = _type_transform(state7, "density_matrix").numpy()


    with Pool(processes=10) as pool:
        results = pool.starmap(
    compute_for_magnitude,
    [(magnitudelist[i], state1, state7, nqubit, noisetype, errorsites)
     for i in range(len(magnitudelist)-1)]
)

        results1, results2 = zip(*results)

        results1 = list(results1)
        results2 = list(results2)

    with open(full_path1, mode='w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerow(['infid^{-1}', 'Sample#_4','Sample#_thrifty_4','vu','N_mterm','vstar','remain','1_Character', '1_Cross_Character', 'Sample#','Sample#_thrifty', 'N_m', 'PartOne_CRM', 'PartTwo', 'PartOne_THR'])
        writer.writerows(results1)

    print(f'Data written to {full_path1}')


    with open(full_path2, mode='w', newline='') as file2:
        writer = csv.writer(file2)
        writer.writerow(['infid^{-1}', 'Sample#_4','Sample#_thrifty_4','vu','N_mterm','vstar','remain','7_Character', '7_Cross_Character', 'Sample#','Sample#_thrifty', 'N_m', 'PartOne_CRM', 'PartTwo', 'PartOne_THR'])
        writer.writerows(results2)
    
    print(f'Data written to {full_path1}')
    return full_path1,full_path2


if __name__ == "__main__":

    filename1,filename2 = main(nqubit=7,theta=0,errorsites=7,noisetype='rotation',statetype='snk')
    print('Finished')






