# Neural Operators for 1D PDE Solving

This repository contains implementations and comparative studies of various Neural Operator architectures, including **Fourier Neural Operators (FNO)**, **Deep Operator Networks (DeepONet)**, and **Physics-Informed Neural Operators (PINO)**.

The focus is on solving 1D physics problems, specifically the **1D Heat Equation** and the **1D Bar (Elasticity) problem**.

## Project Overview

The goal of this project is to benchmark the performance and accuracy of data-driven and physics-informed approaches for solving differential equations. The repository explores:
- **DeepONet:** Learning operators for mapping input functions to solution functions.
- **FNO:** Using Fourier transforms to learn resolution-invariant operators.
- **PINO:** Combining the data efficiency of operators with the physical constraints of PINNs.

## File Structure & Guide

### 1. 1D Heat Equation Experiments
Scripts focused on solving the heat diffusion equation.
* `1D Heat Deep 1.py` / `1D Heat Deep 2.py`: Implementation of DeepONet for the heat equation.
* `1D Heat Deep Training.py`: Training loop and configuration for the DeepONet heat model.
* `1D_Heat Pino.py` / `Heat_1D_PINO.py`: Implementation of Physics-Informed Neural Operators for the heat equation.
* `1D Heat compare.py`: **Key Script.** Compares the results of the different models (likely DeepONet vs. FNO/PINO).

### 2. 1D Bar (Elasticity) Experiments
Scripts focused on the 1D elastic bar deformation problem.
* `1D_Bar_PINO.py`: Solving the bar problem using PINO.
* `1D_Bar_PiDeepONet.py`: Solving the bar problem using Physics-Informed DeepONet.
* `1DBARDEEPONET.py`: Standard DeepONet implementation for the bar.
* `1DBAR DEEP vs FNO.py`: **Key Script.** Comparative analysis between DeepONet and FNO performance on the bar topology.

### 3. Wave Equation & Utilities
* `1D_Wave.py`: Preliminary experiments with the 1D Wave equation.
* `PINO Error.py`: Error analysis and metric calculation for PINO models.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Neural-Operators-1D-Physics.git](https://github.com/YourUsername/Neural-Operators-1D-Physics.git)