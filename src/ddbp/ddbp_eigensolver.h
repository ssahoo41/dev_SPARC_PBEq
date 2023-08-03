/**
 * @file    ddbp_eigensolver.h
 * @brief   This file contains the function declarations for the Discrete
 *          Discontinuous Basis Projection (DDBP) eigensolver routines for the Kohn-Sham
 *          problem.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef _DDBP_EIGENSOLVER_H
#define _DDBP_EIGENSOLVER_H

#include "isddft.h"


void Lanczos_DDBP(
    DDBP_HAMILTONIAN *H_DDBP, const DDBP_ARRAY *X0, double *eigmin,
    double *eigmax, double TOL_min, double TOL_max, int MAXIT, int k,
    int spn_i, MPI_Comm comm);


void Chebyshevfilter_constants_DDBP(
    SPARC_OBJ *pSPARC, DDBP_ARRAY *X0, double *lambda_cutoff,
    double *eigmin, double *eigmax, double *lambda_prev, int count,
    int k, int spn_i
);

/**
 * @brief Chebyshev filtering on the orbitals for the DDBP Hamiltonian.
 *
 *        This routine finds
 *             Y = Pm(H_DDBP) X = Tm((H_DDBP - c)/e) X,
 *        where Tm is the Chebyshev polynomial of the first kind,
 *        c = (a+b)/2, e = (b-a)/2.
 * 
 * @param H_DDBP DDBP Hamiltonian.
 * @param X Orbital (in DDBP basis) to be filtered.
 * @param Y Resulting filtered orbital (in DDBP basis).
 * @param m Chebyshev polynomial degree.
 * @param a Filter bound. a -> -1.
 * @param b Filter bound. b -> +1.
 * @param a0 Filter scaling factor, Pm(a0) = 1.
 * @param kpt K-point index. 
 * @param spn_i Spin index.
 * @param comm Communicator where the orbitals are distributed (bandcomm).
 */
void ChebyshevFiltering_DDBP(
    DDBP_HAMILTONIAN *H_DDBP, DDBP_ARRAY *X, DDBP_ARRAY *Y,
    const int m, const double a, const double b, const double a0,
    const int k, const int spn_i, const MPI_Comm comm);


/**
 * @brief Find the projected Hamiltonian Hp and mass matrix Mp, where
 *
 *          Hp = Y' * H_DDBP * Y.
 *          Mp = Y' * Y.
 *
 * @param DDBP_info DDBP_info object.
 * @param H_DDBP DDBP Hamiltonian.
 * @param n Global size of Hp and Mp.
 * @param Y DDBP KS orbital.
 * @param descY Descriptor for Y.
 * @param HY DDBP Array for storing H_DDBP*Y.
 * @param descHY Descriptor for HY.
 * @param Hp Subspace Hamiltonian (output).
 * @param descHp Descriptor for the resulting Hp (output distributed version).
 * @param Mp Subspace mass matrix (output).
 * @param descMp Descriptor for the resulting Mp (output distributed version).
 * @param kpt K-point index (local).
 * @param spn_i Spin index (local).
 * @param elemcomm Element comm for distributing rows of orbitals (by elements).
 * @param bandcomm Band comm for distributing columns of orbitals (by bands).
 */
void Project_Hamiltonian_DDBP(
    DDBP_INFO *DDBP_info, DDBP_HAMILTONIAN *H_DDBP, int n, const DDBP_ARRAY *Y,
    int *descY, DDBP_ARRAY *HY, int *descHY, double *Hp, int *descHp, double *Mp,
    int *descMp, const int kpt, const int spn_i, MPI_Comm elemcomm, MPI_Comm bandcomm);


// void Solve_Subspace_EigenProblem_DDBP(
//     int n, double *Hp, int *descHp, double *Mp, int *descMp,
//     double *lambda, double *Q, int *descQ, char *typ, int isSerial,
//     MPI_Comm rowcomm, MPI_Comm colcomm);

/**
 * @brief Solve subspace eigenproblem (DDBP).
 * 
 * @param n Global size of Hp and Mp.
 * @param Hp Subspace Hamiltonian.
 * @param descHp Descriptor for the subspace Hamiltonian Hp.
 * @param Mp Subspace mass matrix (not referenced for standard eigenproblem).
 * @param descMp Descriptor for the subspace mass Mp.
 * @param lambda Eigenvalues (output).
 * @param Q Eigenvectors (output).
 * @param descQ Descriptor for the eigenvectors.
 * @param typ Type of eigenproblem.
 *            if typ = "gen": generalized eigenproblem;
 *            if typ = "std": standard eigenproblem.
 * @param isSerial Flag to specify whether to solve the eigenproblem in serial.
 * @param rowcomm Communicator where the subspace eigenproblem is distributed.
 * @param blksz Block size for paral eigensolver (not reference if isSerial = 1).
 * @param maxnp Maximum number of processors to be used for paral eigensolver (not
 *              reference if isSerial = 1).
 * @param proc_active Flag to indicate if the process is active.
 */
void Solve_Subspace_EigenProblem_DDBP(
    int n, double *Hp, int *descHp, double *Mp, int *descMp, double *lambda,
    double *Q, int *descQ, char *typ, int isSerial, MPI_Comm rowcomm,
    int blksz, int maxnp, int proc_active);


void Subspace_Rotation_DDBP(
    int n, DDBP_ARRAY *Y, int *descY, double *Q, int *descQ, DDBP_ARRAY *X,
    int *descX, MPI_Comm comm
);


void recover_orbitals_on_grid(
    DDBP_INFO *DDBP_info, const DDBP_ARRAY *X, int *descX, double **psi,
    int **desc_psi, const int nkpt, const int kpt, const int spn_i);


/**
 * @brief Calculate electron density from given KS orbitals on the
 *        finite-difference grid, which are distributed over DDBP
 *        elements.
 *
 * @param nelem Number of elements.
 * @param elem_list Element list.
 * @param psi Orbitals for all local elements.
 * @param rho Electron density for all local elements (output).
 * @param occ Occupations.
 * @param dV Integration weights associated to each grid point.
 * @param isGammaPoint Flag indicating if it's gamma-point.
 * @param spin_typ Spin type. 0 - spin-unpolarized, 1 - spin polarized.
 * @param Nspin Number of local spin.
 * @param Nkpt Number of local kpoints.
 * @param Nstates Number of states.
 * @param spin_start_index Local start index for spin.
 * @param band_start_indx Local start index for band/orbitals.
 * @param band_end_indx Local end index for band/orbitals.
 * @param rowcomm Row communicator where the bands are distributed column-wisely.
 */
void Calculate_density_psi_DDBP(
    int nelem, DDBP_ELEM *elem_list, double ****psi, double **rho,
    double *occ, double dV, int isGammaPoint, int spin_typ, int Nspin,
    int Nkpt, int Nstates, int spin_start_index, int band_start_indx,
    int band_end_indx, MPI_Comm rowcomm
);

#endif // _DDBP_EIGENSOLVER_H
