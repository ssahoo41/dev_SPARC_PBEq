/**
 * @file    cs.h
 * @brief   This file contains the function declarations for the complementary subspace method.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef CS_H
#define CS_H 

#include "isddft.h"


/**
 * @brief Find the bounds for the dense Chebyshev filtering on the subspace 
 *        Hamiltonian.
 *
 * @param lambda_cutoff  The cutoff parameter used for the original Hamiltonian.
 * @param eigmin         Minimum eigenvalue of the original Hamiltonian.
 * @param eigmin_calc    The previous minimum calculated eigenvalue of the 
 *                       subspace Hamiltonian, not referenced if isFirstIt = 0.
 * @param isFirstIt      Flag to check if this is the first Chebyshev iteration.
 */
void Chebyshevfilter_dense_constants(
  const SPARC_OBJ *pSPARC, const double lambda_cutoff, const double eigmin, 
  const double eigmin_calc, const int isFirstIt, double *a, double *b, double *a0);



/**
 * @brief Find Y = (Hs + cI) * X, where X is distributed column-wisely.
 *
 * @param Hs   Symmetric dense matrix, all proc have a copy of the full matrix.
 * @param c    A constant shift.
 * @param X    A bunch of vectors, to be multiplied.
 * @param ncol Number of columns of X distributed in current process. 
 */
void mat_vectors_mult(
  SPARC_OBJ *pSPARC, const int N, const double *Hs, const int ncol, 
  const double c, const double *X, double *Y, const MPI_Comm comm);

/**
 * @brief   Initialize CS emthod.
 */
void init_CS(SPARC_OBJ *pSPARC);

/**
 * @brief   Initialize complementary subspace eigensolver.
 */
void init_CS_CheFSI(SPARC_OBJ *pSPARC);

/**
 * @brief   Free complementary subspace eigensolver.
 */
void free_CS(SPARC_OBJ *pSPARC);

/**
 * @brief   Chebyshev-filtered subspace iteration eigensolver.
 */
void CheFSI_dense_eig(
  SPARC_OBJ *pSPARC, int Ns, int Nt, double lambda_cutoff, int repeat_chefsi, 
  int count, int k, int spn_i);


/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for a symmetric dense matrix.
 */
void Lanczos_dense_seq(
  SPARC_OBJ *pSPARC, const double *A, const int N, 
  double *eigmin, double *eigmax, double *x0, const double TOL_min, 
  const double TOL_max, const int MAXIT, int k, int spn_i, MPI_Comm comm);


/**
 * @brief Perform Chebyshev filtering.
 *        Y = Pm(Hs) X = Tm((Hs - c)/e) X, where Tm is the Chebyshev polynomial 
 *        of the first kind, c = (a+b)/2, e = (b-a)/2.
 *
 * @param N  Matrix size of Hs.
 * @param m  Chebyshev polynomial degree.
 * @param a  Filter bound. a -> -1.
 * @param b  Filter bound. b -> +1.
 * @param a0 Filter scaling factor, Pm(a0) = 1.
 */
void ChebyshevFiltering_dense(
  SPARC_OBJ *pSPARC, const double *Hs, const int N, double *X, double *Y, 
  const int ncol, const int m, const double a, const double b, const double a0, 
  const int k, const int spn_i, const MPI_Comm comm, double *time_info);


/**
 * @brief   Solve subspace eigenproblem Hp * x = lambda * x for the top Nt 
 *          eigenvalues/eigenvectors using the CheFSI algorithm.
 *
 *          Note: Hp = Psi' * H * Psi, where Psi' * Psi = I. 
 *          
 */
void Solve_partial_standard_EigenProblem_CheFSI(
  SPARC_OBJ *pSPARC, double lambda_cutoff, int Nt, int repeat_chefsi, int k, 
  int count, int spn_i);


/**
 * @brief Call the pdsygvx_ routine with an automatic workspace setup.
 *
 *        The original pdsygvx_ routine asks uses to provide the size of the 
 *        workspace. This routine calls a workspace query and automatically sets
 *        up the memory for the workspace, and then call the pdsygvx_ routine to
 *        do the calculation.
 */
void automem_pdsygvx_ ( 
	int *ibtype, char *jobz, char *range, char *uplo, int *n, double *a, int *ia,
	int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *vl, 
	double *vu, int *il, int *iu, double *abstol, int *m, int *nz, double *w,
	double *orfac, double *z, int *iz, int *jz, int *descz, int *ifail, int *info);


#ifdef USE_DP_SUBEIG

#endif  // End of "#ifdef USE_GTMATRIX"


#endif // EIGENSOLVER_H 
