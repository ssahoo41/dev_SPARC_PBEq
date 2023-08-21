## Input and output files of systems used for assessment of PBEq15 functional

### (1) Brief:
This test suite consists of molecular, metallic and chemisorption systems used to assess the performance of class of XC functionals, PBEq in SPARC. There are three directories, corresponding to molecules, metals and adsorption systems. All the input files and output files are stored.

A brief description of systems in each directory is given below:

### (2) Molecules:
A small set consisting of geometries and energies of 217 small organic molecules made of H, C, N and O atoms are taken from the [Computational Chemistry Comparison and Benchmark Database](https://cccbdb.nist.gov/introx.asp) (CCCBDB). These calculations are done at CCSD(T)-ccPVTZ level of theory.[1]

### (3) Metals:
The solid state reference data used in this study is obtained from data curated by [Wellendorff et al.](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.85.235149),[2] which contains lattice constants and cohesive energies of cubic lattices. We also provide the geometries for single metal atoms required for calculating cohesive energies. 

### (4) Adsorption systems:
For chemisorption, we select CO adsorbed on three transition metal surfaces: Pt(111), Rh(111) and Cu(111). The surfaces are constructed using PBE lattice constants obtained from a benchmark dataset for transition metal surfaces by Wellendorf et. al at low coverage.[2]

### Citations:
If you publish work using this version of code, please cite the following:
[![DOI](https://zenodo.org/badge/671987553.svg)](https://zenodo.org/badge/latestdoi/671987553)
### References:
[1] NIST Computational Chemistry Comparison and Benchmark Database, NIST Standard Reference Database Number 101, Release 22, May 2022, Editor: Russell D. Johnson III 

[2] Wellendorff, Jess, et al. "Density functionals for surface science: Exchange-correlation model development with Bayesian error estimation." Physical Review B 85.23 (2012): 235149.
