"""
4design_upperbound_k.py

This module provides functions for calculating terms in the variance of 4-design CRM and thrifty.

Author: YZY
Date: 2025-04-21
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
    """生成 nqubit 的 Pauli Group，并将其写入 JSON 和 Pickle 文件"""
    pauli_labels = ['x', 'y', 'z']
    pauli_group = []
    json_path = f"pauli_group_{nqubit}qubit.json"
    pkl_path = f"pauli_group_{nqubit}qubit.pkl"
    
    for pauli_ops in product(['I'] + pauli_labels, repeat=nqubit):
        formatted_pauli = [1]  # 系数固定为 1
        pauli_str = []
        
        # 格式化 Pauli 操作，并添加对应 qubit 的位置
        for idx, op in enumerate(pauli_ops):
            if op != 'I':
                pauli_str.append(f"{op}{idx}")
        
        # 如果有非 'I' 操作，则将其加入
        if pauli_str:
            formatted_pauli.append(','.join(pauli_str))  # Pauli 操作按逗号分隔
        
        # 如果没有 Pauli 操作 (只有 'I' 操作)，则跳过
        elif not pauli_str and pauli_ops != tuple(['I'] * nqubit):
            continue
        
        # 将每个操作加入 Pauli group
        pauli_group.append(formatted_pauli)

    # 写入 JSON 文件
    with open(json_path, "w") as f:
        json.dump(pauli_group, f, indent=4)

    # 写入 Pickle 文件
    with open(pkl_path, "wb") as f:
        pickle.dump(pauli_group, f)
    del pauli_group[0]
    return pauli_group  # 返回生成的列表

def load_pauli_group(nqubit):
    """从 JSON 和 Pickle 文件中读取 Pauli Group"""
    json_path = f"pauli_group_{nqubit}qubit.json"
    pkl_path = f"pauli_group_{nqubit}qubit.pkl"
    
    with open(json_path, "r") as f:
        pauli_group_json = json.load(f)

    with open(pkl_path, "rb") as f:
        pauli_group_pickle = pickle.load(f)
    
    return pauli_group_json, pauli_group_pickle

def generate_random_unitary_channel_phase(nqubit,manitude=1,num_sites=1):
    channel_info = []
    sites=random.sample(range(0, nqubit), num_sites)
    for site in sites:
        theta = random.uniform(0,manitude)*2*np.pi     #random phase gate at each site
        # theta = np.pi/6
        channel_info.append([site,theta])
    filename = 'unitary_channel(local).pkl'
    with open(filename, 'wb') as file:
        pickle.dump(channel_info, file) 
    # print(channel_info)
    return channel_info

def generate_random_unitary_channel_rotation(nqubit,thetamag,thetamag2,num_sites=1):
    channel_info = []
    sites=random.sample(range(0, nqubit), num_sites)
    for site in sites:
        theta = random.uniform(thetamag,thetamag2)*np.pi 
        phi = random.uniform(thetamag,thetamag2)*np.pi 
        lamb = random.uniform(thetamag,thetamag2)*np.pi 
        # phi = random.uniform(0,1)*2*np.pi 
        # lamb = random.uniform(0,1)*2*np.pi 
        channel_info.append([site,theta,phi,lamb])
        filename = 'unitary_channel(local rotation).pkl'
    with open (filename,'wb') as file:
        pickle.dump(channel_info, file) 
    # print(channel_info)
    return channel_info

def generate_unitary_channel_phase(state, nqubit, channelinfo):
    cir = Circuit(nqubit)
    # print(f'initial state:{state}')
    state = to_state(state,backend=Backend.DensityMatrix)
    for i in range(len(channelinfo)):
        site = channelinfo[i][0]
        theta = channelinfo[i][1]
        # theta = 2
        cir.p([site],param=theta)
    # print(f'circuit:{cir}')
    state_after = cir.forward(state)
    
    # print(f'state_noised:{state_after}')
    # print(f'fidelity:{np.trace(state_after@state)}')
    return state_after

def generate_unitary_channel_rotation(state, nqubit, channelinfo):
    cir = Circuit(nqubit)
    # print(f'initial state:{state}')
    state = to_state(state,backend=Backend.DensityMatrix)
    for i in range(len(channelinfo)):
        site = channelinfo[i][0]
        theta = channelinfo[i][1]
        phi = channelinfo[i][2]
        lamb = channelinfo[i][3]
        cir.u3([site],param=[theta,phi,lamb])
        # print(f'circuit:{cir}')
    state_after = cir.forward(state)
    
    # print(f'state_noised:{state_after}')
    # print(f'fidelity:{np.trace(state_after@state)}')
    return state_after

def generate_theta_vector(nqubit, mag):
    """
    生成 nqubit 个 theta 值的向量，每个 theta_i = arccos(sqrt(v_i))，v_i >=0, sum(v_i)=mag
    返回 2*theta_i 的向量
    """
    # Step 1: 随机生成 nqubit 个 [0,1] 的数
    v = np.random.rand(nqubit)
    beta = np.log2(mag)
    # Step 2: 归一化使得 sum(v) = mag
    v = v / np.sum(v) * beta
    
    # Step 3: 每个分量开根号再 arccos
    theta = np.arccos(2**v)

    # Step 4: 返回 2*theta
    return 2 * theta
def generate_random_unitary_channel_rotation2(nqubit, mag, num_sites=1):
    channel_info = []
    sites = random.sample(range(0, nqubit), num_sites)
    # vec = np.random.randn(3)
    # axis = vec / np.linalg.norm(vec)
    # axis = fix_axis
    # 在 [thetamag, thetamag2] * pi 范围内随机选择旋转角

    thetalist = generate_theta_vector(num_sites, mag)
    for i  in range(len(sites)):
        # axis = collection_axis[i]
        site = sites[i]
        theta = thetalist[i]
        # 随机生成旋转轴（单位向量）
        vec = np.random.randn(3)
        axis = vec / np.linalg.norm(vec)
        
        # 在 [thetamag, thetamag2] * pi 范围内随机选择旋转角
        # theta = random.uniform(thetamag, thetamag2) 
        # print('axis:', axis)
        # print('theta:', theta)
        # theta = random.uniform(0, 1) * 2 * np.pi
        # 保存信息：[qubit index, rotation angle, rotation axis]
        channel_info.append([site, theta, axis.tolist()])  # axis 转成 list 方便保存

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
    s_x = 1/np.sqrt(2)
    s_y = 1/np.sqrt(2)
    s_z = 0
    s_vec = np.array([s_x, s_y, s_z])
    for info in channelinfo:
        site = info[0]
        theta = info[1]
        axis = info[2]
        print('axis::',axis,'   theta:',theta)
        u3_params = axis_to_u3_params(theta, axis)
        cir.u3([site], param=u3_params)  # 使用 u3 门
        p_j = np.cos(theta/2)**2 + (np.dot(axis, s_vec))**2 * np.sin(theta/2)**2
        prodtest *= p_j
    
    state_after1 = cir.forward(state1)
    state_after2 = cir.forward(state2)
    # /statep  = _type_transform(state, "density_matrix").numpy()
    state_afterp1 = _type_transform(state_after1, "density_matrix").numpy()
    state_afterp2 = _type_transform(state_after2, "density_matrix").numpy()
    print('fid_inst1:',np.trace(statep1@state_afterp1),'  fid_theory:',prodtest)
    print('fid_inst2:',np.trace(statep2@state_afterp2),'  fid_theory:',prodtest)
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
        # print(i)
        # print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        # pauli =pq.qinfo.pauli_str_to_matrix([[1,pauli_group[i][1]]],nqubit)
        # print(pauli)
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
    # print('pauligroup:',pauli_group)
    sigma_vec = []
    cross_vec = []
    state_vec = []
    d = 2**nqubit
    I = np.identity(d)
    ob = sigma-I/d
    ob_vec = []
    for i in range(len(pauli_group)):
        # print(i)
        # print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        # pauli =pq.qinfo.pauli_str_to_matrix([[1,pauli_group[i][1]]],nqubit)
        # print(pauli) 
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
    """单个进程的计算任务"""
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
    # amp_thrifty_variance_41 = np.real(thrifty_variance_41/(1-fidelity1)**2)
    # amp_variance_41= np.real(variance_41/(1-fidelity1)**2)
    sample_41 = int(68*np.log(200)*variance_41*16/(1-fidelity1)**2)
    sample_thr_41 = int(68*np.log(200)*thrifty_variance_41*16/(1-fidelity1)**2)
    remain1 = (-fidelity1**2+d*(2*fidelity1+1)/(d+2)-vstar1)

    variance_42= vu2 + 1/N_m2*(-fidelity2**2+d*(2*fidelity2+1)/(d+2)-vstar2)
    N_m_term2 = (-fidelity2**2+d*(2*fidelity2+1)/(d+2)-vstar2)
    thrifty_variance_42 = vstar2+1/N_m2*(-fidelity2**2+d*(2*fidelity2+1)/(d+2)-vstar2)
    # amp_thrifty_variance_42 = np.real(thrifty_variance_42/(1-fidelity2)**2)
    # amp_variance_42 = np.real(variance_42/(1-fidelity2)**2)
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
    start_time = time.time()
    current_time = datetime.now().strftime("%m%d_%H%M")
    noisetype = 'rotation'

    # 修改为你的目标路径
    # folder_path = rf"C:\Users\Administrator\Desktop\plottemp\4d_equal_spacing_Unoise k"

    folder_path = rf"C:\Users\Administrator\Desktop\plottemp\same_axis_equal_spacing_Unoise_k_0304"

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
        # results2 = pool.starmap(compute_for_magnitude, [(magnitudelist[i], state7, nqubit,noisetype,errorsites) for i in range(len(magnitudelist)-1)])
    # 写入 CSV 文件到指定目录
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'运行时间: {hours}时{minutes}分{seconds}秒')
    return full_path1,full_path2


if __name__ == "__main__":
    # klist = [1,4,7]
    # klist=[1,2,3]
    # Nm = 1e7
    # N=50
    # log_inv_x = np.linspace(-3, 0, N)

    # # 计算对应的 x 值
    # magnitudelist = 10 ** log_inv_x
    # l = range(2, 11,2)
    # sample_list = [10**int(r) for r in l ]
    # for k in klist:
        # filename = main(nqubit=7, k=k,theta=0,errorsites=7,noisetype='rotation',statetype='snk')# 画
    filename1,filename2 = main(nqubit=7,theta=0,errorsites=7,noisetype='rotation',statetype='snk')# 画
    print('Finished')
        # plot_results(filename,k)
        # plot_results('shadow_norm_snk_state_nqubit=5_k=5_Nm=1000000_0325_1931.csv',k)
    # variance_amp_values = []

    # # 读取 CSV 文件
    # with open(filename, mode='r', newline='') as file:
    #     reader = csv.reader(file)
    #     header = next(reader)  # 读取表头
        
    #     # 找到 'Variance(amp)' 在第几列
    #     index = header.index('Variance(amp)')

    #     for row in reader:
    #         try:
    #             variance_amp_values.append(float(row[index]))  # 读取并转换为浮点数
    #         except ValueError:
    #             pass  # 如果转换失败（例如空值），就跳过

    # # 计算平均值
    # if variance_amp_values:
    #     mean_variance_amp = np.mean(variance_amp_values)
    #     print(f"Variance(amp) 平均值: {mean_variance_amp}")
    # else:
    #     print("没有有效的 Variance(amp) 数据")





