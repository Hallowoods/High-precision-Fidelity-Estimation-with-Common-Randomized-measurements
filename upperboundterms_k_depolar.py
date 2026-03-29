"""
upperboundterms_k.py

This module provides functions for calculating upper bound terms(including cross-characteristic, characteristic, variance, and their amplified version, sample complexity) for snk states with different k. Both for CRM and THR protocol.

Pauli noise is assumed.

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


def save_pauli_channel(channelinfo, filename):
    '''write-in Pauli Channel file'''
    with open(filename, 'wb') as file:
        pickle.dump(channelinfo, file)  # 将 channelinfo 保存到文件中

def load_pauli_channel(filename):
    '''Read the Pauli Channel file'''

    with open(filename, 'rb') as file:
        channelinfo = pickle.load(file)  # 从文件中读取 channelinfo
    return channelinfo


def generate_random_pauli_channel(nqubit,magnitude = 0.1,channel_num = 0):
    channelinfo = []
    existing_pauli_strings = set()  # 用于跟踪已生成的 Pauli 字符串
    maxallowed = 4**nqubit - 1
    if channel_num == 0:    #default:random
        number_of_string = random.randint(1, maxallowed)
    else:
        number_of_string = channel_num
    

    while len(channelinfo) < number_of_string:  # 生成指定数量的通道信息
        coefficient = random.uniform(0, 1)  # 在 [0, 1] 区间生成实数
        num_pauli = random.randint(1, nqubit)  # 随机生成 Pauli 字符串的数量
        indices = random.sample(range(nqubit), num_pauli)  # 随机选择不重复的索引
        pauli_string = ','.join(f'{random.choice("xyz")}{i}' for i in sorted(indices))  # 生成 Pauli 字符串
        
        if pauli_string not in existing_pauli_strings:  # 检查字符串是否已存在
            existing_pauli_strings.add(pauli_string)  # 添加到集合中
            channelinfo.append([coefficient, pauli_string])  # 添加到 channelinfo 列表
    sum_p = sum(ele[0] for ele in channelinfo)
    for ele in channelinfo:
        ele[0] = ele[0]/sum_p*magnitude  
        save_pauli_channel(channelinfo, 'pauli_channel.pkl')  # 保存到文件
    return channelinfo,magnitude





def caculate_characters_CRM(state,sigma,nqubit):
    '''Calculate cross chatacteristic functions of rho and sigma and charcteristic functions, output as a product of them'''
   
    pauli_group = generate_pauli_group(nqubit=nqubit) #loading Pauli group first
    # print('pauligroup:',pauli_group)
    delta = state-sigma
    delta_vec = []
    sigma_vec = []
    cross_vec = []
    for i in range(len(pauli_group)):
        print(i)
        print(pauli_group[i])
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


def generate_depolarizing_channel(nqubit, magnitude=0.1):
    """生成退极化通道，返回等概率作用的所有非恒等 Pauli 字符串"""
    pauli_group = generate_pauli_group(nqubit)
    depolarizing_channel = []
    d = 2 ** nqubit
    p = magnitude

    # 平均分配所有非恒等Pauli操作
    for ele in pauli_group:
        depolarizing_channel.append([p / (d**2 - 1), ele[1]])  # 保证总和为 p

    return depolarizing_channel, p

def apply_depolarizing_noise(state, p):
    nqubit = int(np.log2(state.shape[0]))
    dim = 2 ** nqubit
    identity = np.eye(dim) / dim
    return (1 - p) * state + p * identity

def compute_for_magnitude(magnow, state, nqubit, N_m=None):
    """使用退极化通道计算"""
    d = 2**nqubit
    state_noised = apply_depolarizing_noise(state, p=magnow)
    state_noised = _type_transform(state_noised, "density_matrix").numpy()

    character, cross_character = caculate_characters_CRM(state_noised, state, nqubit)
    fidelity = np.trace(state_noised @ state)
    if (1 - fidelity) != 0:
        N_m = math.ceil(10 / (1 - fidelity) ** 2)
    else:
        N_m = 1e12  # 防止除零，可以给一个很大的数
    character2, cross_character2 = caculate_characters_THR(state_noised, state, nqubit)

    amp_character = np.real(character / (1 - fidelity)**2) * 16
    amp_cross = np.real(cross_character / (1 - fidelity)**2) * 16

    variance_u = ((d + 1) / (d * (d + 2))) * (np.real(character) + np.real(cross_character)) - (1 - np.real(fidelity))**2 / (d + 2)
    variance_0 = ((d + 1) / (d * (d + 2))) * (np.real(character2) + np.real(cross_character2)) - (1 - np.real(fidelity))**2 / (d + 2)
    variance = variance_u + 1 / N_m * (-fidelity**2 + d * (2 * fidelity + 1) / (d + 2) - variance_0)
    thrifty_variance = variance_0 + 1 / N_m * (-fidelity**2 + d * (2 * fidelity + 1) / (d + 2) - variance_0)

    amp_variance = np.real(variance / (1 - fidelity)**2) * 16
    amp_thrifty_variance = np.real(thrifty_variance / (1 - fidelity)**2) * 16

    sample = int(16 * 68 * np.log(200) * variance / (1 - fidelity)**2)
    sample_thr = int(16 * 68 * np.log(200) * thrifty_variance / (1 - fidelity)**2)

    return [1 / (1 - np.real(fidelity)), np.real(character), np.real(cross_character), sample,
            np.real(variance), amp_variance, amp_character, amp_cross,
            np.real(thrifty_variance), amp_thrifty_variance, sample_thr]

    #thrifty ones
    

    return [1 / (1 - np.real(fidelity)), np.real(character), np.real(cross_character), sample,np.real(variance),amp_variance,amp_character,amp_cross,np.real(thrifty_variance),amp_thrifty_variance,sample_thr]

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


def main(nqubit, k,):
    start_time = time.time()
    current_time = datetime.now().strftime("%m%d_%H%M")
    # N_m = 100000000000000
    N_m = None
    # N_m = 1000
    folder_path = rf"C:\Users\Administrator\Desktop\depolar08"
    # filename = f'{k}-haar_shadow_norm_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'

    filename = f'shadow_norm_snk_state_nqubit={nqubit}_k={k}_Nm={N_m}_{current_time}.csv'
    full_path = f"{folder_path}\\{filename}"
    # filename = f'shadow_norm_snk_state_nqubit={nqubit}_k={k}_Nm={N_m}_{current_time}.csv'
    # filename = f'shadow_norm_ghz_state_nqubit={nqubit}_Nm={N_m}_{current_time}.csv'
    N = 50  # 样本数量

    # 1/x 在 10^-5 到 1 之间对数均匀分布
    log_inv_x = np.linspace(-3, 0, N)

    # 计算对应的 x 值
    magnitudelist = 10 ** log_inv_x
    

    # 生成 snk 量子态
    if k != 0:
        cir = Circuit(nqubit)
        cir.h(range(k))
        cir.t(range(k))
        state = cir()
    else:
        state = pq.state.zero_state(nqubit)
    state = _type_transform(state, "density_matrix").numpy()
    # state = pq.state.ghz_state(nqubit)
    # state = _type_transform(state, "density_matrix").numpy()

    # 使用 Pool 进行并行计算
    with Pool(processes=10) as pool:  # 10 个进程并行
        results = pool.starmap(compute_for_magnitude, [(magnow, state, nqubit,N_m) for magnow in magnitudelist])

    # 统一写入 CSV 文件
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['infid^{-1}', f'{k}_Character', f'{k}_Cross_Character', 'Sample#','Variance','Variance(amp)',f"{k}_Character(amp)",f"{k}_Cross_Character(amp)",'Variance_thrifty','Variance_thrifty(amp)','Sample#_thrifty'])
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
    # ks = [1, 3, 5, 7]
    ks = [0]

    nqubit = 7
    all_results = {}

    for k in ks:
        print(f"\nRunning for k = {k}...")
        full_path = main(nqubit=nqubit, k=k)
        
        # 读取 Variance(amp) 数据
        variance_amp_values = []
        with open(full_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            index = header.index('Variance(amp)')

            for row in reader:
                try:
                    variance_amp_values.append(float(row[index]))
                except ValueError:
                    continue

        if variance_amp_values:
            mean_variance_amp = np.mean(variance_amp_values)
            print(f"Variance(amp) 平均值 for k={k}: {mean_variance_amp}")
            all_results[k] = mean_variance_amp
        else:
            print(f"No valid Variance(amp) data found for k={k}")
            all_results[k] = None

    # 输出汇总结果
    print("\n=== Variance(amp) 平均值汇总 ===")
    for k, value in all_results.items():
        print(f"k = {k}: {value if value is not None else '无有效数据'}")


# if __name__ == "__main__":
#     main(nqubit=3,k=1)
