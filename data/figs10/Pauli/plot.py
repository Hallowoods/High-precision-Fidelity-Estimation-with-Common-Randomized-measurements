# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# import re

# # 文件夹路径
# folder = r"C:\Users\22968\Desktop\TFIM\PauliCRM_EDGS"

# # 获取所有 CSV 文件
# file_list = [f for f in os.listdir(folder) if f.endswith(".csv")]

# # 自动从文件名提取 nqubit
# pattern = re.compile(r"n=(\d+)_")

# plt.figure(figsize=(8,5))
# colors = plt.cm.viridis_r(np.linspace(0,1,len(file_list)))  # 不同颜色

# for i, fname in enumerate(sorted(file_list)):
#     fpath = os.path.join(folder, fname)
#     df = pd.read_csv(fpath)
#     # 提取 nqubit
#     match = pattern.search(fname)
#     if match:
#         n = int(match.group(1))
#     else:
#         n = i+1
#     plt.scatter(df['h'], df['V_star'], label=f"n={n}", color=colors[i], linestyle='-')
#     plt.plot(df['h'], df['V_star'], color=colors[i], linestyle='-') # 连线

# plt.xlabel("h")
# plt.ylabel("V_star")
# plt.yscale('log')
# plt.title("V_star vs h for different n")
# plt.legend()
# plt.grid(False)  # 去掉 grid
# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import re
scale=6/7
R = 1e5
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"
# 设置 Matplotlib 使用 LaTeX
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 31.6,
    'legend.fontsize': 28.6
})

l = 15*scale
plt.figure(figsize=(l, l*0.618))

# 文件夹路径
folder = r"C:\Users\Administrator\Desktop\TFIM\PauliCRM_EDGS"

# 获取所有 CSV 文件
file_list = [f for f in os.listdir(folder) if f.endswith("p=0.01.csv")]

# 自动从文件名提取 nqubit
pattern = re.compile(r"n=(\d+)_")

# 使用 tab10 调色板
colors = plt.cm.get_cmap("tab10")(np.arange(len(file_list)) % 10)

for i, fname in enumerate(sorted(file_list)):
    fpath = os.path.join(folder, fname)
    df = pd.read_csv(fpath)
    # 提取 nqubit
    match = pattern.search(fname)
    if match:
        n = int(match.group(1))
        d = 2**n
    else:
        n = i+1
    NU = 68*np.log2(200)*16*(df['V_starDelta']+(df['VarianceStd']-df['V_starrho'])/R)/(0.01*(1-1/d))**2
    plt.plot(
        df['h'], NU,
        'o-', color=colors[i],
        alpha=0.7, markersize=10,
        label=fr"$n={n}$"
    )

plt.xlabel(r"$h$")
plt.ylabel(r"$N_U$")
plt.ylim(1e4,9e4)
plt.xlim(-0.1,4.1)
# plt.yscale('log')
from matplotlib.ticker import ScalarFormatter

# ...
plt.ylabel(r"$N_U$")

# 使用科学计数法显示纵坐标
ax = plt.gca()  # 获取当前坐标轴
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(26)  # 调整科学计数法指数的字体大小

plt.legend(ncol=2,loc='upper left')
plt.grid(False)
plt.tight_layout(pad=0.05)
plt.show()