# Continuous-Time Echo State Networks for Predicting Power System Dynamics

This repository contains the Julia code used in the paper: C. Roberts, J. D. Lara, R. Henriquez-Auba, M. Bossart, R. Anantharaman, C. Rackauckas, B.-M. Hodge and D. Callaway, "Continuous-Time Echo State Networks for Predicting Power System Dynamics", presented in the Power Systems Computation Conference (PSCC) 2022.

The code is organized in different scripts that allow to replicate the results obtained in the paper. The user must run Julia in this project environment to ensure correct versions of each package used. The code is properly commented to ensure easy understanding of the different scripts and functions.

### Accuracy of the Surrogate

Results for Section IV.A are in the file `src/trainSize.jl` (Figures 4 and 5) that compares the different train sizes with the correction errors.

### Power System Dynamic Behavior

Results for Section IV.B are in the files `src/sampleTraces.jl` (Figure 6) and `src/parameterSweep.jl` (Figures 7, 8 and 9).

### Scalability and Computation Time

Results for Section IV.C are in the files `src/144BusAccuracy.jl` (Tables I and II) and `src/accelerationBenchmark.jl` (Figure 10).
