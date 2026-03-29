"""
unitary_noise_upperboundterms.py

This module provides functions for calculating upper bound terms(including cross-characteristic, characteristic, variance, and their amplified version, sample complexity) for snk states with different k. Both for CRM and THR protocol.

Local unitary noise(phase gate) is assumed.
和1022的区别：这里是用的是特殊生成方法（Euler angles with same values）
Author: YZY
Date: 2025-04-01
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
from paddle_quantum.channel.common import MixedUnitaryChannel


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


def generate_theta_vector_uni(nqubit, mag):
    """
    生成 nqubit 个 theta 值的向量，每个 theta_i = arccos(sqrt(v_i))，v_i >=0, sum(v_i)=mag
    返回 2*theta_i 的向量
    """

    v = np.array([1 for _ in range(nqubit)])        #相同转角
    beta = np.log2(mag)
    # Step 2: 归一化使得 sum(v) = mag
    v = v / np.sum(v) * beta
    
    # Step 3: 每个分量开根号再 arccos
    theta = np.arccos(2**v)

    # Step 4: 返回 2*theta
    return 2 * theta



def generate_random_unitary_channel_rotation(m,nqubit,num_sites=1):
    channel_info = []
    sites=random.sample(range(0, nqubit), num_sites)
    thetalist = generate_theta_vector_uni(num_sites, m)
    for i  in range(len(sites)):
        site = sites[i]
        theta = thetalist[i]
        phi = theta 
        lamb = theta
        # phi = random.uniform(0,1)*2*np.pi 
        # lamb = random.uniform(0,1)*2*np.pi 
        channel_info.append([site,theta,phi,lamb])
        filename = 'unitary_channel(local rotation).pkl'
    with open (filename,'wb') as file:
        pickle.dump(channel_info, file) 
    # print(channel_info)
    return channel_info
    
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

    


def compute_for_magnitude( m,state, nqubit, noisetype='rotation', N_m=None, errorsites=1):
    """单个进程的计算任务"""
    d = 2**nqubit
    if noisetype == 'rotation':
        # channel = generate_random_unitary_channel_rotation2(m,nqubit, num_sites=errorsites)
        # state_noised = generate_unitary_channel_rotation2(state, nqubit, channel)
        channel = generate_random_unitary_channel_rotation(m,nqubit, num_sites=errorsites)
        state_noised = generate_unitary_channel_rotation(state, nqubit, channel)
    
    state_noised = _type_transform(state_noised, "density_matrix").numpy()
    
    # fidelity
    fidelity = np.real(np.trace(state_noised @ state))

    # 在这里动态设置 N_m
    if (1 - fidelity) != 0:
        N_m = math.ceil(10 / (1 - fidelity) ** 2)
    else:
        N_m = 1e12  # 防止除零，可以给一个很大的数

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

    sample = int(68 * np.log(200) * 16* variance / (1 - fidelity) ** 2) if (1 - fidelity) ** 2 != 0 else -1
    sample_thr = int(68 * np.log(200) * 16*thrifty_variance / (1 - fidelity) ** 2) if (1 - fidelity) ** 2 != 0 else -1
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


# def plot_results(filename,k):
#     """从 CSV 读取数据并绘制曲线"""
#     data = np.loadtxt(filename, delimiter=",", skiprows=1)  # 跳过表头
#     x = data[:, 0]  # 第一列 infid^{-1} 作为横坐标
#     y1 = data[:, 1]  # k_Character
#     y2 = data[:, 2]  # k_Cross_Character
#     y3 = data[:,7]  # k_Character(amp)
#     y4 = data[:,6]  # k_Cross_Character(amp)
#     y5 = data[:,4]  # Real variance
#     y6 = data[:,3]    #sample complexity
#     y7 = data[:,5]    #Real variance(amp)
#     y8 = data[:,8]      #thrifty_variance
#     y9 = data[:,9]      #amp_thrifty_variance]
#     y10 = data[:,10]        ##sample complexity(thrifty)
#     plt.figure(figsize=(8, 6))
#     # plt.scatter(x, y1, marker="o",  label=f"{k}_Character", color="b")
#     # plt.scatter(x, y2, marker="s", label=f"{k}_Cross_Character", color="r")
#     # plt.scatter(x, y3, marker="*",  label=f"{k}_Character(amp)", color="g")
#     # plt.scatter(x, y4, marker="^", label=f"{k}_Cross_Character(amp)", color="orange")
#     # plt.scatter(x, y5, marker="o",  label=f"{k}_Variance", color="b")
#     # plt.scatter(x, y8, marker="x",  label=f"{k}_Variance_thrifty", color="r")
#     plt.scatter(x, y6, marker="x", label=f"{k}_NU", color="r")
#     plt.scatter(x, y10, marker="o", label=f"{k}_NU_THR", color="b")
#     # plt.scatter(x, y7, marker="x",  label=f"{k}_Variance(amp)", color="r")
#     # plt.scatter(x, y9, marker="o",  label=f"{k}_Variance_thrifty(amp)", color="b")


#     plt.xlabel("infid^{-1}")
#     plt.ylabel("Character Values")
#     plt.title("Character vs infid^{-1}")
#     plt.legend()
#     plt.grid(True)
#     # plt.savefig(filename.replace(".csv", ".png"))  # 保存图片
#     plt.show()


def main(nqubit, k,errorsites,theta,statetype,noisetype='rotation',):
    # folder_path = rf"C:\Users\Administrator\Desktop\plottemp\equal_spacing_Unoise k"
    # folder_path = rf'C:\Users\Administrator\Desktop\plottemp\equal_spacing_Unoise k\storage'
    folder_path = rf'C:\Users\Administrator\Desktop\0808test2\test'
    # filename = f'{k}-haar_shadow_norm_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'

    
    # full_path = f"{folder_path}\\{filename}"
    # errorsites = nqubit
    start_time = time.time()
    current_time = datetime.now().strftime("%m%d_%H%M")
    N_m = None
    # N_m = 10000

    noisetype = 'rotation'
    if statetype == 'snk':
        filename = f'Unoise_nqubit={nqubit}_k={k}_NM={N_m}_errorsites={errorsites}_type_{noisetype}_{current_time}_test.csv'
        if k != 0:
            cir = Circuit(nqubit)
            cir.h(range(k))
            cir.t(range(k))
            state = cir()
        else:
            state = pq.state.zero_state(nqubit)
    
    elif statetype=='ghz':
        filename = f'Unoise_nqubit={nqubit}_ghz_errorsites={errorsites}_type_{noisetype}_{current_time}.csv'
        state = pq.state.ghz_state(nqubit)
    elif statetype =='w':
        filename = f'Unoise_nqubit={nqubit}_W_errorsites={errorsites}_type_{noisetype}_{current_time}.csv'
        state = pq.state.w_state(nqubit)
    elif statetype == 'snk_theta':
        filename = f'Unoise_nqubit={nqubit}_k={k}_NM={N_m}_theta={theta}_errorsites={errorsites}_type_{noisetype}_{current_time}.csv'
        if k != 0:
            cir = Circuit(nqubit)
            cir.h(range(k))
            cir.p(range(k), param=theta,param_sharing=True)
            state = cir()
        else:
            state = pq.state.zero_state(nqubit)
    full_path = f"{folder_path}\\{filename}"
    state = _type_transform(state, "density_matrix").numpy()
    # x = np.linspace(1, 10000, 400)
    # magnitudelist = 1/x
    # magnitudelist = 1 / x  # 1/x spacing
    # magnitudelist = np.linspace(0.0001,0.001,100)
    # magnitudelist = np.linspace(0.001,0.01,100)
    # magnitudelist = np.linspace(0.01,0.1,100)
    # magnitudelist = [0.1]*50
    # magnitudelist=[1]
    N = 150 # 样本数量

    # 1/x 在 10^-5 到 1 之间对数均匀分布
    bx = np.linspace(0,4,N)
    magnitudelist = 1-10**(-bx)


    # log_inv_x = np.linspace(-5, 0, N)
    # magnitudelist = 10 ** log_inv_x*np.pi


    
    # log_inv_x = np.linspace(-5, 0, N)
    # log_inv_x = np.array([-5.0 for _ in range(N)])

    # 计算对应的 x 值
    # magnitudelist = 10 ** log_inv_x
    # magnitudelist = np.linspace(0.1,1,N)*np.pi
    # state = _type_transform(state, "density_matrix").numpy()

    # 使用 Pool 进行并行计算
    with Pool(processes=10) as pool:  # 10 个进程并行
        results = pool.starmap(compute_for_magnitude, [(magnitudelist[i],state, nqubit,noisetype,N_m,errorsites) for i in range(len(magnitudelist)-1)])

    # 统一写入 CSV 文件
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['infid^{-1}', f'{k}_Character', f'{k}_Cross_Character', 'Sample#','Variance','Variance(amp)',f"{k}_Character(amp)",f"{k}_Cross_Character(amp)",'Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty', 'N_m', 'PartOne_CRM', 'PartTwo', 'PartOne_THR'])
        writer.writerows(results)

    print(f'Data written to {full_path}')
    
    # 画图
    # plot_results(filename,k)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'运行时间: {hours}时{minutes}分{seconds}秒')
    return full_path

if __name__ == "__main__":
    klist = [1,4,7]
    n = 7
    for k in klist:
        filename = main(nqubit=n, k=k,theta=0,errorsites=n,noisetype='rotation',statetype='snk')# 画
        # plot_results(filename,k)
        # plot_results('shadow_norm_snk_state_nqubit=5_k=5_Nm=1000000_0325_1931.csv',k)
        variance_amp_values = []

        # 读取 CSV 文件
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # 读取表头
            
            # 找到 'Variance(amp)' 在第几列
            index = header.index('Variance(amp)')

            for row in reader:
                try:
                    variance_amp_values.append(float(row[index]))  # 读取并转换为浮点数
                except ValueError:
                    pass  # 如果转换失败（例如空值），就跳过

        # 计算平均值
        if variance_amp_values:
            mean_variance_amp = np.mean(variance_amp_values)
            print(f"Variance(amp) 平均值: {mean_variance_amp}")
            # print(collection_axis)

        else:
            print("没有有效的 Variance(amp) 数据")
            # print(collection_axis)

