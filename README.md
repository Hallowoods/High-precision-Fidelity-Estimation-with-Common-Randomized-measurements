

# High-precision-Fidelity-Estimation-with-Common-Randomized-measurements

[![arXiv](https://img.shields.io/badge/arXiv-2511.22509-b31b1b.svg)](https://arxiv.org/abs/2511.22509)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The code is primarily designed to:
* Compute various quantum state fidelities, cross characteristic functions, and twisted cross characteristic functions and other relevant quantities discussed in the paper https://arxiv.org/pdf/2511.22509.
* Simulate the numerical values of these quantities under different noise models.
* Evaluate and compare the variance and circuit cost of different estimation protocols.

## Notice
These source code is only responsible for generating the data in the paper, which needs further reasonable selections to obtain the figures in the paper. 

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
