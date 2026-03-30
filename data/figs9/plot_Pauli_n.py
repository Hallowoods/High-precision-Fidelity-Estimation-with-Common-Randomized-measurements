import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
n=15
scale = 1.5
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 31.6,
    'legend.fontsize': 28.6,
    'font.family': 'Times New Roman',
})
# ------------------------------
# 配置字体与 LaTeX 渲染
# ------------------------------
# plt.rcParams.update({
#     'text.usetex': True,
#     'font.size': 28*scale,
#     'legend.fontsize': 26*scale,
#     'font.family': 'Times New Roman',
#     'text.latex.preamble': r'\usepackage{amsmath,amssymb}'
# })

# ---------- 参数 ----------
n_values = range(1, 15)
folder_path = r"C:\Users\Administrator\Desktop\Pauli CRM"
# 初始化图形
l = 10.5*0.898
R = 100
fig, ax = plt.subplots(figsize=(l, l*0.8))
# ---------- 读取 Pauli 数据 ----------
x_vals, y_pauli = [2,4,6,8,10,12,14], []
color_map = plt.get_cmap('tab10')
color_clifford = color_map(0)   # 蓝
color_pauli = color_map(1)      # 橙
color_shadow = color_map(2)     # 绿

# ---------- 计算 Clifford 理论值 ----------
y_clifford = []
for i in x_vals:
    d = 2 ** n
    val = np.ceil(3600700* (d+(d+1)/R))
    y_clifford.append(val)

# ---------- 计算 4D 理论值 ----------
y_4D = []
for i in x_vals:
    d = 2 ** n
    # coeff = ((d**2 + 3*d + 4) / (d*(d+2)*(d+3))
    #          + (2*(d+1)**2)*(1+2/d**2-2/d) / (d*(d+2)*(d+3))
    #          + 2*(d+1)*(1-1/d)/ (d*(d+3))
    #          - (2*(d+1))*(1+2/d**2-2/d) / (d*(d+2)*(d+3)))
    coeff = (2+8/d-4/(d+3))/4+1/R*((2*d+1)/(d+2)-1/4*(1+3/d-2/(d+3)))
    y_4D.append(np.ceil(3600700 * coeff))


# ---------- 计算 Pauli 理论值 ----------
for w in x_vals:
    d = 2 ** n
    coeff = 3**w-1+3**w/R
    y_pauli.append(np.ceil(3600700 * coeff))

# ---------- 绘图 ----------
# fig, ax = plt.subplots(figsize=(10*scale, 6*scale))

# markers = ['o', 's', '^']
colors = ['b', 'r', 'g']

# Pauli
# ax.scatter(x_vals, y_pauli, facecolors='none', edgecolors=colors[0],
#            s=120*scale, marker=markers[0],  label="Pauli")
ax.plot(x_vals, y_pauli, linestyle='-', marker='o', color=color_pauli, markersize=10,  label="Pauli")

# Clifford
# ax.scatter(x_vals, y_clifford, facecolors='none', edgecolors=colors[1],
#            s=120*scale, marker=markers[1], label="Clifford")
ax.plot(x_vals, y_clifford, linestyle='--', marker='s', color=color_clifford, markersize=10, label="Clifford")

# 4D
# ax.scatter(x_vals, y_4D, facecolors='none', edgecolors=colors[2],
#            s=120*scale, marker=markers[2], label="4D")
ax.plot(x_vals, y_4D, linestyle='-.', marker='^', color=color_shadow, markersize=10, label="4-design")

# ---------- 坐标轴 ----------
ax.set_xlabel("$w$")
ax.set_ylabel(r"$N_U$")
ax.set_yscale("log")
# plt.ylim(5e3,3e10)
plt.xlim(1.8,14.2)
# 固定横坐标刻度
ax.set_xticks(x_vals)

ax.legend(loc='best', labelspacing=0.2,)
ax.grid(False)
plt.subplots_adjust(left=0.135, right=0.99, bottom=0.12, top=0.99)
save_path = os.path.join(folder_path, "Sample_mean_vs_n_with_4D_scatter.pdf")
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
plt.show()