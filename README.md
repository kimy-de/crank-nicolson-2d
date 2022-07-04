# Crank-Nicolson method in 2D
This repository provides the Crank-Nicolson method to solve the heat equation in 2D. Basically, The numerical method is processed by CPUs, but it can be implemented on GPUs if the CUDA is installed.

### 1. Requirements
* numpy, torch, tqdm, matplotlib

### 2. Default setting
* Initial condition: a star shape (modifiable)
* Boundary condition: zero Neumann

* crank_nicolson.py: iteration
* disc.py: discretization matrices
* utils.py: initial condition, result plot

### 3. Execution
```python
python crank_nicolson.py
```

### 4. Result
<p align="center">
<img width="800" alt="r1" src="https://user-images.githubusercontent.com/52735725/177200432-ac4c26bf-fc36-4f6e-87ff-5de1393987de.png">
</p>
