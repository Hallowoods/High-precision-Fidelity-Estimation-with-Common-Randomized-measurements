"""
single_pauli_error.py

This module provides functions for calculating (cross) characteristic functions for single-qubit pauli error in Clifford CRM
Author: YZY
Date: 2025-04-28
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
from paddle_quantum.state import State
from scipy.interpolate import interp1d

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

# def load_pauli_group(nqubit):
#     """从 JSON 和 Pickle 文件中读取 Pauli Group"""
#     json_path = f"pauli_group_{nqubit}qubit.json"
#     pkl_path = f"pauli_group_{nqubit}qubit.pkl"
    
#     with open(json_path, "r") as f:
#         pauli_group_json = json.load(f)

#     with open(pkl_path, "rb") as f:
#         pauli_group_pickle = pickle.load(f)
    
#     return pauli_group_json, pauli_group_pickle




def save_pauli_channel(channelinfo, filename):
    '''write-in Pauli Channel file'''
    with open(filename, 'wb') as file:
        pickle.dump(channelinfo, file)  # 将 channelinfo 保存到文件中

def load_pauli_channel(filename):
    '''Read the Pauli Channel file'''

    with open(filename, 'rb') as file:
        channelinfo = pickle.load(file)  # 从文件中读取 channelinfo
    return channelinfo


def generate_fixed_pauli_channel(nqubit, magnitude=0.5):
    if nqubit < 1:
        raise ValueError("nqubit must be at least 1")
    
    channelinfo = []
    coefficient = magnitude  # 就是 magnitude 本身
    pauli_string = f'z{nqubit-1}'      # 固定字符串
    
    channelinfo.append([coefficient, pauli_string])

    
    save_pauli_channel(channelinfo, 'pauli_channel.pkl')  # 保存到文件
    return channelinfo, magnitude




def caculate_characters_CRM(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
   
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
    delta = state-sigma
    delta_vec = []
    sigma_vec = []
    cross_vec = []
    for i in range(len(pauli_group)):
        # print(i)
        # print(pauli_group[i])
        ele = pauli_group[i]
        ele[1] = pq.qinfo.pauli_str_to_matrix([[1,ele[1]]],nqubit).numpy()
        # pauli =pq.qinfo.pauli_str_to_matrix([[1,pauli_group[i][1]]],nqubit)
        # print(pauli)
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


import math
def compute_for_magnitude(m,magnow, nqubit,N_m =100):
    """单个进程的计算任务"""
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
    # variance = variance_u       #this is wrong! just as a reference!
    thrifty_variance = variance_0+1/N_m*(-fidelity**2+d*(2*fidelity+1)/(d+2)-variance_0)
    amp_thrifty_variance = np.real(thrifty_variance/(1-fidelity)**2)
    amp_variance= np.real(variance*16/(1-fidelity)**2)
    sample = int(68*np.log(200)*variance*16/(1-fidelity)**2)
    sample_thr = int(68*np.log(200)*thrifty_variance/(1-fidelity)**2)
    r = 1/(m**4*(1-m**2)**2)
    #thrifty ones
    

    return [1/m**4*(1-m**2),m,r,1 / (1 - np.real(fidelity)), np.real(character), np.real(cross_character), sample,np.real(variance),amp_variance,amp_character,amp_cross,np.real(thrifty_variance),amp_thrifty_variance,sample_thr]


def compute_for_magnitude_4design(m,magnow, nqubit,N_m =100):
    """单个进程的计算任务"""
    print(rf'current m_{m}')
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
    # variance = variance_u       #this is wrong! just as a reference!
    amp_variance= np.real(variance*16/(1-fidelity)**2)
    sample = int(68*np.log(200)*variance*16/(1-fidelity)**2)

    #thrifty ones
    
    return [1/m**4*(1-m**2),m,r,1 / (1 - np.real(fidelity)),'nan', 'nan', sample,np.real(variance),amp_variance,'nan','nan','nan','nan','nan']


def plot_results(filename,k):
    """从 CSV 读取数据并绘制曲线"""
    data = np.loadtxt(filename, delimiter=",", skiprows=1)  # 跳过表头
    x = data[:, 0]  # 第一列 infid^{-1} 作为横坐标
    y1 = data[:, 1]  # k_Character
    y2 = data[:, 2]  # k_Cross_Character
    y3 = data[:,7]  # k_Character(amp)
    y4 = data[:,6]  # k_Cross_Character(amp)
    y5 = data[:,4]  # Real variance
    y6 = data[:,3]    #sample complexity
    y7 = data[:,5]    #Real variance(amp)
    y8 = data[:,8]      #thrifty_variance
    y9 = data[:,9]      #amp_thrifty_variance]
    y10 = data[:,10]        ##sample complexity(thrifty)
    plt.figure(figsize=(8, 6))
    # plt.scatter(x, y1, marker="o",  label=f"{k}_Character", color="b")
    # plt.scatter(x, y2, marker="s", label=f"{k}_Cross_Character", color="r")
    # plt.scatter(x, y3, marker="*",  label=f"{k}_Character(amp)", color="g")
    # plt.scatter(x, y4, marker="^", label=f"{k}_Cross_Character(amp)", color="orange")
    # plt.scatter(x, y5, marker="o",  label=f"{k}_Variance", color="b")
    # plt.scatter(x, y8, marker="x",  label=f"{k}_Variance_thrifty", color="r")
    plt.scatter(x, y6, marker="x", label=f"{k}_NU", color="r")
    plt.scatter(x, y10, marker="o", label=f"{k}_NU_THR", color="b")
    # plt.scatter(x, y7, marker="x",  label=f"{k}_Variance(amp)", color="r")
    # plt.scatter(x, y9, marker="o",  label=f"{k}_Variance_thrifty(amp)", color="b")


    plt.xlabel("infid^{-1}")
    plt.ylabel("Character Values")
    plt.title("Character vs infid^{-1}")
    plt.legend()
    plt.grid(True)
    # plt.savefig(filename.replace(".csv", ".png"))  # 保存图片
    plt.show()

def generate_random_sector_state(nqubit, m):
    if abs(m) > 1:
        raise ValueError("m must satisfy |m| ≤ 1")

    dim = 2 ** nqubit
    state_vector = np.zeros((dim, 1), dtype=complex)

    # 第一个分量
    # state_vector[0, 0] = m
        # sector: 最后一个 qubit = 1 的索引
    indices_0 = [i for i in range(dim) if (i & 1) == 0]
    num_sector_0 = len(indices_0)

    # 随机复数振幅
    real_part_0 = np.random.randn(num_sector_0)
    imag_part_0 = np.random.randn(num_sector_0)
    random_amplitudes_0 = real_part_0 + 1j * imag_part_0

    # 归一化这些随机振幅，再乘上 sqrt(1 - |m|^2)
    norm_0 = np.linalg.norm(random_amplitudes_0)
    random_amplitudes_0 = random_amplitudes_0 / norm_0 * np.sqrt(m**2)

    # sector: 最后一个 qubit = 1 的索引
    indices_1 = [i for i in range(dim) if (i & 1) == 1]
    num_sector_1 = len(indices_1)

    # 随机复数振幅
    real_part_1 = np.random.randn(num_sector_1)
    imag_part_1 = np.random.randn(num_sector_1)
    random_amplitudes_1 = real_part_1 + 1j * imag_part_1

    # 归一化这些随机振幅，再乘上 sqrt(1 - |m|^2)
    norm_1 = np.linalg.norm(random_amplitudes_1)
    random_amplitudes_1 = random_amplitudes_1 / norm_1 * np.sqrt(1 - abs(m)**2)

    # 填入 state_vector
    for idx, amp in zip(indices_0, random_amplitudes_0):
        state_vector[idx, 0] = amp
    for idx, amp in zip(indices_1, random_amplitudes_1):
        state_vector[idx, 0] = amp

    # 直接用 state_vector 创建 State 实例
    state = State(state_vector)

    return state

def main(nqubit,fdesign=True):
    start_time = time.time()
    current_time = datetime.now().strftime("%m%d_%H%M")
    N_m = 10000000
    if fdesign:
        # folder_path = rf"C:\Users\Yzy\Desktop\probability-expectation-project\src\one_error_Clifford"
        folder_path = rf"D:\probability-expectation-project\src\one_error_4design"
        # filename = f'{k}-haar_shadow_norm_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'

        filename = f'one_error_shadow_norm_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'
        full_path = f"{folder_path}\\{filename}"
        # r = 4*m**4
    
        # filename = f'shadow_norm_ghz_state_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'
        # m_raw = np.linspace(1-0.999**2, 1-0.001**2, 100)
        # u = np.linspace(0, 1, 100)
        # m_raw = np.linspace(0.01, 0.1, 500)
        # mlist = m_raw
        # mlist = np.logspace(np.log10(0.02), 0, 200)
        mlist = np.exp(np.linspace(-5, 0, 10))



        # 使用 Pool 进行并行计算
        with Pool(processes=10) as pool:  # 10 个进程并行
            results = pool.starmap(compute_for_magnitude_4design, [(m,0.5, nqubit,N_m) for m in mlist])

        # 统一写入 CSV 文件
        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['1/1-phi0',
            'm','r','infid^{-1}', 'Character', 'Cross_Character', 'Sample#','Variance','Variance(amp)',"Character(amp)","Cross_Character(amp)",'Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty'])
            writer.writerows(results)

        print(f'Data written to {full_path}')
        
        # 画图
        # plot_results(filename,k)

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f'运行时间: {hours}时{minutes}分{seconds}秒')
        return filename
    else:
        folder_path = rf"D:\probability-expectation-project\src\one_error_Clifford"
        # folder_path = rf"C:\Users\Yzy\Desktop\probability-expectation-project\src\one_error_4design"
        # filename = f'{k}-haar_shadow_norm_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'

        filename = f'one_error_shadow_norm_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'
        full_path = f"{folder_path}\\{filename}"
        # r = 4*m**4
    
        # filename = f'shadow_norm_ghz_state_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'
        # m_raw = np.linspace(1-0.999**2, 1-0.001**2, 100)
        # u = np.linspace(0, 1, 100)
        # m_raw = np.linspace(0.01, 0.1, 500)
        # mlist = m_raw
        # mlist = np.logspace(-5, 0, 200)
        mlist = np.exp(np.linspace(-5, 0, 10))



        # 使用 Pool 进行并行计算
        with Pool(processes=10) as pool:  # 10 个进程并行
            results = pool.starmap(compute_for_magnitude, [(m,0.5, nqubit,N_m) for m in mlist])

        # 统一写入 CSV 文件
        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['1/1-phi0',
            'm','r','infid^{-1}', 'Character', 'Cross_Character', 'Sample#','Variance','Variance(amp)',"Character(amp)","Cross_Character(amp)",'Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty'])
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
    full_path = main(nqubit=7,fdesign = True)# 画图
    # plot_results(filename,k)
    # plot_results('shadow_norm_snk_state_nqubit=5_k=5_Nm=1000000_0325_1931.csv',k)
    variance_amp_values = []

    # 计算平均值
    if variance_amp_values:
        mean_variance_amp = np.mean(variance_amp_values)
        print(f"Variance(amp) 平均值: {mean_variance_amp}")
    else:
        print("没有有效的 Variance(amp) 数据")

# if __name__ == "__main__":
#     main(nqubit=3,k=1)
