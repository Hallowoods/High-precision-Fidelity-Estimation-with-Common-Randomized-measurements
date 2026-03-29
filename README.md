

# High-precision-Fidelity-Estimation-with-Common-Randomized-measurements

[![arXiv](https://img.shields.io/badge/arXiv-2511.22509-b31b1b.svg)](https://arxiv.org/abs/2511.22509)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The code is primarily designed to:
* Compute various quantum state fidelities, cross characteristic functions, and twisted cross characteristic functions and other relevant quantities discussed in the paper https://arxiv.org/pdf/2511.22509.
* Simulate the numerical values of these quantities under different noise models.
* Evaluate and compare the variance and circuit cost of different estimation protocols.

## Notice
These source codes are only responsible for generating the data in the paper, and therefore need further reasonable selections to obtain the figures in the paper. 

## Requirements

This project is written in Python. We recommend using `conda` or `pip` to create a virtual environment.

Core dependencies:
* Python >= 3.11.5
* Numpy
* Scipy
* Matplotlib
* Pandas
* Paddlepaddle    
* Paddle-quantum
* Mpi4py

## Citation
```bibtex
@misc{yourlastname2026fidelity,
      title={High-precision Fidelity Estimation with Common Randomized measurements}, 
      author={Firstname Lastname and Firstname Lastname and Firstname Lastname},
      year={2026},
      eprint={2603.xxxxx},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## Reproducing the Figures (Figure-Code Mapping)

Here is a detailed guide on how to reproduce the core figures in our paper. 

### Figure 1: Variance of Different Estimation Protocols
This figure compares the variance of our proposed protocol against traditional methods under depolarizing noise.

<p align="center">
  <img src="./figures/fig1_variance.png" width="600" alt="Variance Comparison">
</p>

* **Corresponding Script**: `src/variance_eval.py`
* **How to run**:
  ```bash
  python src/variance_eval.py --noise_type depolarizing --protocol common_randomized
  ```
* **Note**: The script will generate a `.csv` file. You can then use `scripts/plot_variance.py` to plot this exact figure.

---

### Figure 2: Verification of the Quantum Zeno Effect
This figure illustrates the system dynamics under high-frequency measurements, verifying the mean-field Hamiltonian approximation.

<p align="center">
  <img src="./figures/fig2_zeno_effect.png" width="600" alt="Quantum Zeno Effect">
</p>

* **Corresponding Script**: `src/zeno_effect.py`
* **How to run**:
  ```bash
  python src/zeno_effect.py --n_tot 12 --n_a 4 --k 2
  ```

---

### Figure 3: MPI Parallel Computation of $V^*$
This figure shows the computational scaling of $V^*$ as the number of qubits ($n$) increases, computed using the MPI parallelized script.

<p align="center">
  <img src="./figures/fig3_vstar_scaling.png" width="600" alt="V* Scaling">
</p>

* **Corresponding Script**: `src/compute_V_star_mpi.py`
* **How to run**:
  ```bash
  mpiexec -n 4 python src/compute_V_star_mpi.py
  ```
