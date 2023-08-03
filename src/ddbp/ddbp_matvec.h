/**
 * @file    ddbp_matvec.h
 * @brief   This file contains the function declarations for the Discrete
 *          Discontinuous Basis Projection (DDBP) Hamiltonian matrix-vector
 *          multiplication routines.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef _DDBP_MATVEC_H
#define _DDBP_MATVEC_H

#include "isddft.h"
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

void adaptive_dsymm(
    const CBLAS_ORDER order, const CBLAS_SIDE side,
    const CBLAS_UPLO uplo, const int m, const int n, const double alpha,
    const double*a, const int lda, const double *b, const int ldb,
    const double beta, double *c, const int ldc);


void adaptive_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const int m, const int n, const int k,
    const double alpha, const double *a, const int lda, const double *b,
    const int ldb, const double beta, double *c, const int ldc);


/**
 * @brief DDBP Hamiltonian matrix-vector multiplication routine.
 *
 *        This routine finds
 *            HX = alpha * (H_DDBP + c*I) X + beta * HX,
 *        where H_DDBP is the DDBP Hamiltonian, alpha, beta, c
 *        are constants.
 *
 * @details This routine calls the kernel and does the mat-vec
 *          in bunches of a certain number. Since if ncol is too
 *          large the efficiency of halo exchange can become bad.
 *          Moreover, the memory required for the buffer will be
 *          larger. If ncol is too small, say ncol = 1, not only
 *          the overhead for halo exchange is large, the mat-vec
 *          is also not as efficient as BLAS 3. Therefore, there
 *          is a sweat spot where we get the best efficiency for
 *          the matrix multiplication of H_DDBP and vectors.
 *
 * @param alpha Scalar alpha.
 * @param H_DDBP DDBP Hamiltonian object.
 * @param c Scalar c.
 * @param X Vector to be multiplied.
 * @param beta Scalar beta.
 * @param HX Resulting vector (output).
 * @param comm Communicator that includes all the processes that owns
 *             pieces of X, and HX.
 */
void DDBP_Hamiltonian_vectors_mult(
    const double alpha, DDBP_HAMILTONIAN *H_DDBP,
    const double c, const DDBP_ARRAY *X, const double beta, DDBP_ARRAY *HX,
    MPI_Comm comm);


/**
 * @brief Calculate selected columns of DDBP Hamiltonian matvec.
 *
 * @details This routine finds the selected columns of HX, i.e.,
 *                        HX(is:is+ncol-1),
 *          where HX = alpha * (H_DDBP + c*I) X + beta * HX, H_DDBP
 *          is the DDBP Hamiltonian, alpha, beta, c are constants.
 *
 * @param alpha Scalar alpha.
 * @param H_DDBP DDBP Hamiltonian object.
 * @param c Scalar c.
 * @param X Vector to be multiplied.
 * @param beta Scalar beta.
 * @param HX Resulting vector (output).
 * @param comm Communicator that includes all the processes that owns
 *             pieces of X, and HX.
 * @param is Starting column of HX to be computed.
 * @param ncol Number of columns of HX to be computed.
 */
void DDBP_Hamiltonian_vectors_mult_selectcol(
    const double alpha, DDBP_HAMILTONIAN *H_DDBP,
    const double c, const DDBP_ARRAY *X, const double beta, DDBP_ARRAY *HX,
    MPI_Comm comm, const int is, const int ncol);


#endif // _DDBP_MATVEC_H
