/**
 * @file    ddbp_eigensolver.c
 * @brief   This file contains the functions for the Discrete Discontinuous
 *          Basis Projection (DDBP) eigensolver routines for the Kohn-Sham
 *          problem.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <limits.h>
/* BLAS and LAPACK routines */
#ifdef USE_MKL
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif
/* ScaLAPACK routines */
#ifdef USE_MKL
    #include "blacs.h"     // Cblacs_*
    #include <mkl_blacs.h>
    #include <mkl_pblas.h>
    #include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
    #include "blacs.h"     // Cblacs_*
    #include "scalapack.h" // ScaLAPACK functions
#endif

#include "parallelization.h"
#include "linearAlgebra.h"
#include "nlocVecRoutines.h"
#include "eigenSolver.h"
#include "finalization.h"
#include "isddft.h"
#include "tools.h"
#include "cs.h"
#include "ddbp.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define TEMP_TOL 1e-14


void Lanczos_DDBP(
    DDBP_HAMILTONIAN *H_DDBP, const DDBP_ARRAY *X0, double *eigmin,
    double *eigmax, double TOL_min, double TOL_max, int MAXIT, int k,
    int spn_i, MPI_Comm comm)
{
    double t1, t2;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #ifdef DEBUG
        if (rank == 0) printf("\nStart Lanczos_DDBP ...\n");
    #endif

    double vscal, err_eigmin, err_eigmax, eigmin_pre, eigmax_pre;
    // double *V_j, *V_jm1, *V_jp1, 
    double *a, *b, *d, *e;
    // int i, j;
    DDBP_ARRAY Xj, Xjm1, Xjp1;
    DDBP_ARRAY *V_j   = &Xj;
    DDBP_ARRAY *V_jm1 = &Xjm1;
    DDBP_ARRAY *V_jp1 = &Xjp1;
    duplicate_DDBP_Array_template(X0, V_j);
    duplicate_DDBP_Array_template(X0, V_jm1);
    duplicate_DDBP_Array_template(X0, V_jp1);

    a = (double*)malloc((MAXIT+1) * sizeof(double));
    b = (double*)malloc((MAXIT+1) * sizeof(double));
    d = (double*)malloc((MAXIT+1) * sizeof(double));
    e = (double*)malloc((MAXIT+1) * sizeof(double));
    assert(a != NULL && b != NULL && d != NULL && e != NULL);

    // V_jm1 = 1.0 * X0 + 0.0 * V_jm1 = X0
    axpby_DDBP_Array(1.0, X0, 0.0, V_jm1);

    // find norm of V_jm1
    Norm_DDBP_Array(V_jm1, 1, &vscal, comm);

    vscal = 1.0 / vscal;
    // scale the random guess vector s.t. V_jm1 has unit 2-norm
    scale_DDBP_Array(vscal, V_jm1);

    t1 = MPI_Wtime();
    // V_j = H * V_jm1
    DDBP_Hamiltonian_vectors_mult(1.0, H_DDBP, 0.0, V_jm1, 0.0, V_j, comm);
    t2 = MPI_Wtime();
#ifdef DEBUG
    if(!rank) printf("rank = %2d, One H_DDBP*x took %.3f ms\n", rank, (t2-t1)*1e3);   
#endif
    // find dot product of V_jm1 and V_j, and store the value in a[0]
    // VectorDotProduct(V_jm1, V_j, DMnd, &a[0], comm);
    DotProduct_DDBP_Array(V_jm1, V_j, 1, &a[0], comm);

    // orthogonalize V_jm1 and V_j
    // V_j = V_j - a[0] * V_jm1
    axpby_DDBP_Array(-a[0], V_jm1, 1.0, V_j);
    
    // find norm of V_j
    Norm_DDBP_Array(V_j, 1, &b[0], comm);
    
    if (!b[0]) {
        // if ||V_j|| = 0, pick an arbitrary vector with unit norm that's orthogonal to V_jm1
        randomize_DDBP_Array(V_j, comm);
        // orthogonalize V_j and V_jm1
        DotProduct_DDBP_Array(V_j, V_jm1, 1, &a[0], comm);
        // V_j = V_j - a[0] * V_jm1
        axpby_DDBP_Array(-a[0], V_jm1, 1.0, V_j);
        // find norm of V_j
        Norm_DDBP_Array(V_j, 1, &b[0], comm);
    }

    // scale V_j
    vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
    scale_DDBP_Array(vscal, V_j);

    eigmin_pre = *eigmin = 0.0;
    eigmax_pre = *eigmax = 0.0;
    err_eigmin = TOL_min + 1.0;
    err_eigmax = TOL_max + 1.0;
    int j = 0;
    // while ((err_eigmin > TOL_min || err_eigmax > TOL_max) && j < MAXIT) 
    for (j = 0; j < MAXIT; j++)
    {
        if (err_eigmin <= TOL_min && err_eigmax <= TOL_max) break;
        // V_{j+1} = H * V_j
        DDBP_Hamiltonian_vectors_mult(1.0, H_DDBP, 0.0, V_j, 0.0, V_jp1, comm);

        // a[j+1] = <V_j, V_{j+1}>
        DotProduct_DDBP_Array(V_j, V_jp1, 1, &a[j+1], comm);

        // V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
        axpbypcz_DDBP_Array(-b[j], V_jm1, -a[j+1], V_j, 1.0, V_jp1);
        // update V_{j-1}, i.e., V_{j-1} := V_j
        axpby_DDBP_Array(1.0, V_j, 0.0, V_jm1);

        Norm_DDBP_Array(V_jp1, 1, &b[j+1], comm);
        if (!b[j+1]) break;

        vscal = 1.0 / b[j+1];
        // update V_j := V_{j+1} / ||V_{j+1}||
        axpby_DDBP_Array(vscal, V_jp1, 0.0, V_j);

        // solve for eigenvalues of the (j+2) x (j+2)
        // tridiagonal matrix T = tridiag(b,a,b)
        for (int i = 0; i < j+2; i++) {
            d[i] = a[i];
            e[i] = b[i];
        }

        if (!LAPACKE_dsterf(j+2, d, e)) {
            *eigmin = d[0];
            *eigmax = d[j+1];
        } else {
            if (rank == 0) printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");
            break;
        }

        err_eigmin = fabs(*eigmin - eigmin_pre);
        err_eigmax = fabs(*eigmax - eigmax_pre);
        eigmin_pre = *eigmin;
        eigmax_pre = *eigmax;
    }

#ifdef DEBUG
    if (rank == 0) {
        printf("    Lanczos (H_DDBP) iter %d, eigmin  = %.9f, eigmax = %.9f, err_eigmin = %.3e, err_eigmax = %.3e\n",j,*eigmin, *eigmax,err_eigmin,err_eigmax);
    }
#endif

    delete_DDBP_Array(V_j);
    delete_DDBP_Array(V_jm1);
    delete_DDBP_Array(V_jp1);
    free(a); free(b); free(d); free(e);
}



// find chebyshev fiter bounds for H_DDBP
void Chebyshevfilter_constants_DDBP(
    SPARC_OBJ *pSPARC, DDBP_ARRAY *X0, double *lambda_cutoff,
    double *eigmin, double *eigmax, double *lambda_prev, int count,
    int k, int spn_i
)
{
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
    MPI_Comm bandcomm = DDBP_info->bandcomm;
    int chefsibound_flag = pSPARC->chefsibound_flag;
    int rhoTrigger = pSPARC->rhoTrigger;
    int elecgs_Count = pSPARC->elecgs_Count;
    int Ns = pSPARC->Nstates;
    // double *lambda_prev = pSPARC->lambda_sorted;

    // tolerances for Lanczos
    double tol_eigmin = 1e10;
    double tol_eigmax = pSPARC->TOL_LANCZOS;
    // eigmin is only needed for the very first SCF step
    if (count == 0) {
        tol_eigmin = tol_eigmax;
    }

    double lanczos_eigmin = *eigmin; 
    double lanczos_eigmax = *eigmax;

    if (count == 0 || (count >= rhoTrigger && chefsibound_flag == 1) || 1) // ! forcing always true!
    // if (count == 0 || count-rhoTrigger == 0 || (count >= rhoTrigger && chefsibound_flag == 1))
    {
        DDBP_HAMILTONIAN *H_DDBP = &DDBP_info->H_DDBP;
        // use Lanczos to find the extreme eigenvalues
        Lanczos_DDBP(H_DDBP, X0, &lanczos_eigmin, &lanczos_eigmax,
            tol_eigmin, tol_eigmax, 1000, k, spn_i, bandcomm);
        // TODO: consider bcasting to make sure all processes have the same value
        lanczos_eigmin -= 0.10;
        lanczos_eigmax *= 1.01;
        
        lanczos_eigmax += 1.0; // TODO: remove after check

    }

    // set up outputs: eigmin, eigmax, lambda_cutoff
    // 1. eigmin, 2. eigmax
    if (count == 0) {
        *eigmin = lanczos_eigmin;
        *eigmax = lanczos_eigmax;
    } else if (count >= rhoTrigger) {
        // take previous eigmin
        *eigmin = lambda_prev[spn_i*Ns];
        *eigmax = lanczos_eigmax;
    }

    // 3. lambda_cutoff
    if (elecgs_Count == 0 && count == 0) {
        *lambda_cutoff = 0.5 * (*eigmin + *eigmax);
    } else{
        //*lambda_cutoff = pSPARC->Efermi + log(1e6-1) / pSPARC->Beta + 0.1;
        *lambda_cutoff = lambda_prev[(spn_i+1)*Ns-1] + 0.1;
    }
}



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
    const int kpt, const int spn_i, const MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;
    if (X->ncol <= 0 || Y->ncol <= 0) return;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    #ifdef DEBUG   
    if(!rank) printf("Start DDBP Chebyshev filtering routine ... \n");
    #endif

    double e, c, sigma, sigma1, sigma2, gamma, vscal, vscal2;  
    e = 0.5 * (b - a);
    c = 0.5 * (b + a);
    sigma = sigma1 = e / (a0 - c);
    gamma = 2.0 / sigma1;

    double t1, t2, time_info;
    t1 = MPI_Wtime();
    // find Y = (H - c*I)X
    DDBP_Hamiltonian_vectors_mult(1.0, H_DDBP, -c, X, 0.0, Y, comm);
    t2 = MPI_Wtime();
    time_info += t2 - t1;

    // scale Y by (sigma1 / e)
    vscal = sigma1 / e;
    // Y *= vscal;
    scale_DDBP_Array(vscal, Y);

    // Ynew = (double *)malloc(len_tot * sizeof(double));
    DDBP_ARRAY *Ynew = malloc(sizeof(*Ynew));
    assert(Ynew != NULL);
    duplicate_DDBP_Array_template(Y, Ynew); // set up Ynew (mem,haloX,params)

    for (int j = 1; j < m; j++) {
        sigma2 = 1.0 / (gamma - sigma);

        t1 = MPI_Wtime();
        // Ynew = (H - c*I)Y
        DDBP_Hamiltonian_vectors_mult(1.0, H_DDBP, -c, Y, 0.0, Ynew, comm);
        t2 = MPI_Wtime();
        time_info += t2 - t1;

        // Ynew = (2*sigma2/e) * Ynew - (sigma*sigma2) * X, then update X and Y
        vscal = 2.0 * sigma2 / e; vscal2 = sigma * sigma2;
        // Ynew = vscal * Ynew - vscal2 * X
        axpby_DDBP_Array(-vscal2, X, vscal, Ynew);
        // X = Y
        axpby_DDBP_Array(1.0, Y, 0.0, X);
        // Y = Ynew
        axpby_DDBP_Array(1.0, Ynew, 0.0, Y);
        sigma = sigma2;
    }
    delete_DDBP_Array(Ynew);
    free(Ynew);
}


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
    int *descMp, const int kpt, const int spn_i, MPI_Comm elemcomm, MPI_Comm bandcomm)
{
    if (DDBP_info->elemcomm_index == -1 || DDBP_info->basiscomm_index == -1 ||
        DDBP_info->dmcomm == MPI_COMM_NULL) {
        return;
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    #ifdef DEBUG   
    if(!rank) printf("Start DDBP projection routine ... \n");
    #endif

    // find Mp = Y^T * Y
    Hermitian_Multiply_DDBP_Array(
        n, 1.0, Y, descY, Y, descY, 0.0, Mp, descMp, elemcomm, bandcomm);

    // find HY = H_DDBP * Y
    DDBP_Hamiltonian_vectors_mult(1.0, H_DDBP, 0.0, Y, 0.0, HY, bandcomm);

    // find Hp = Y^T * HY
    Hermitian_Multiply_DDBP_Array(
        n, 1.0, Y, descY, HY, descHY, 0.0, Hp, descHp, elemcomm, bandcomm);    
}

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
    int blksz, int maxnp, int proc_active)
{
    if (proc_active == 0) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    #ifdef DEBUG   
    if(!rank) printf("Start DDBP subspace eigenproblem routine ... \n");
    #endif

    int rank_rowcomm, rank_colcomm;
    MPI_Comm_rank(rowcomm, &rank_rowcomm);
    // MPI_Comm_rank(colcomm, &rank_colcomm);
    double t1, t2;
    int ONE = 1, info = 0;

// #define PRINT_MAT
#ifdef PRINT_MAT
    // MPI_Barrier(rowcomm);
    char fname[128] = "subspace_matrices.txt";
    FILE *output_fp = fopen(fname,"w");
    if (output_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",fname);
        exit(EXIT_FAILURE);
    }
    if (rank == 0) {
        fprintf(output_fp,"Hp = [\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                fprintf(output_fp,"%.16e ", Hp[j*n+i]);
            }
            fprintf(output_fp,"\n");
        }
        fprintf(output_fp,"];\n");
    }

    if (rank == 0) {
        fprintf(output_fp,"Mp = [\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                fprintf(output_fp,"%.16e ", Mp[j*n+i]);
            }
            fprintf(output_fp,"\n");
        }
        fprintf(output_fp,"];\n");
    }
    fclose(output_fp);
    // if (rank == 0) {
    //     printf("subspace eigenproblem (DDBP): local Hp in rank %d\n", rank);
    //     int ncol = 5; // hard-coded
    //     for (int i = 0; i < min(ncol,10); i++) {
    //         for (int j = 0; j < min(ncol,12); j++) {
    //             printf("%15.8e ", Hp[j*n+i]);
    //         }
    //         printf("\n");
    //     }
    // }

    // if (rank == 0) {
    //     printf("subspace eigenproblem (DDBP): local Mp in rank %d\n", rank);
    //     int ncol = 5; // hard-coded
    //     for (int i = 0; i < min(ncol,10); i++) {
    //         for (int j = 0; j < min(ncol,12); j++) {
    //             printf("%15.8e ", Mp[j*n+i]);
    //         }
    //         printf("\n");
    //     }
    // }
#endif

    if (isSerial) {
        t1 = MPI_Wtime();
        // only root process in rowcomm has the matrix
        if (rank_rowcomm == 0) {
            if (strcmpi(typ, "gen") == 0) {
                info = LAPACKE_dsygvd(
                    LAPACK_COL_MAJOR,1,'V','U', n, Hp, n, Mp, n, lambda);
            } else if (strcmpi(typ, "std") == 0) {
                info = LAPACKE_dsyevd(LAPACK_COL_MAJOR,'V','U', n, Hp,n, lambda);
            }
            if (info) printf("info = %d\n", info);
            assert(info == 0);
        }
        // MPI_Barrier(rowcomm);
        t2 = MPI_Wtime();
        if (rank == 0) printf("rank = %d, LAPACK eigensolver: %.3f ms\n", rank, (t2-t1)*1e3);
        
        t1 = MPI_Wtime();
#if defined(USE_MKL) || defined(USE_SCALAPACK)
        int ictxt = descQ[1];
        if (descQ[1] < 0) {
            ictxt = descHp[1];
        }

        // find the larger context that contains both contexts
        // int nprow_Q, npcol_Q, myrow_Q, mycol_Q;
        // Cblacs_gridinfo(descQ[1], &nprow_Q, &npcol_Q, &myrow_Q, &mycol_Q);
        // int nprow_Hp, npcol_Hp, myrow_Hp, mycol_Hp;
        // Cblacs_gridinfo(descHp[1], &nprow_Hp, &npcol_Hp, &myrow_Hp, &mycol_Hp);
        // ictxt = (nprow_Hp * npcol_Hp >= nprow_Q * npcol_Q) ? descHp[1] : descQ[1];
        // printf("rank = %d, np_Q = %d, np_Hp = %d, descQ[1] = %d, descHp[1] = %d, ictxt = %d\n",
        //     rank, nprow_Hp*npcol_Hp, nprow_Q*npcol_Q, descQ[1], descHp[1], ictxt);

        // distribute eigenvectors to block cyclic format
        if (descQ[1] >= 0 || descHp[1] >= 0) {
            pdgemr2d_(&n, &n, Hp, &ONE, &ONE, descHp, Q, &ONE, &ONE, descQ, &ictxt);
        }
        t2 = MPI_Wtime();
        if (!rank) printf("rank = %d, redistributing eigenvectors took: %.3f ms\n", rank, (t2 - t1)*1e3);
#else
        // TODO: implement corresponding action without MKL/ScaLAPACK
        assert(0);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    } else {
        int gridsizes[2] = {n,n}, ierr = 1, size_rowcomm, subdims[2];
        MPI_Comm_size(rowcomm, &size_rowcomm);
        SPARC_Dims_create(min(size_rowcomm, maxnp), 2, gridsizes, 1, subdims, &ierr);
        if (ierr) subdims[0] = subdims[1] = 1;
        if (!rank)
            printf("rank = %d, for subspace eigensolver: process grid = (%d,%d).\n",rank,subdims[0],subdims[1]);
/*
// subdims[0] = subdims[1] = 1; // TODO: remove after check

printf("rank = %d, descHp = [%d,%d,%d,%d,%d,%d,%d,%d,%d], break 1!\n",rank,
    descHp[0],
    descHp[1],
    descHp[2],
    descHp[3],
    descHp[4],
    descHp[5],
    descHp[6],
    descHp[7],
    descHp[8]);
printf("rank = %d, descMp = [%d,%d,%d,%d,%d,%d,%d,%d,%d], break 1!\n",rank,
    descMp[0],
    descMp[1],
    descMp[2],
    descMp[3],
    descMp[4],
    descMp[5],
    descMp[6],
    descMp[7],
    descMp[8]);
printf("rank = %d, descQ  = [%d,%d,%d,%d,%d,%d,%d,%d,%d], break 1!\n",rank,
    descQ[0],
    descQ[1],
    descQ[2],
    descQ[3],
    descQ[4],
    descQ[5],
    descQ[6],
    descQ[7],
    descQ[8]);
*/
        int il = 1, iu = 1, M, NZ;
        double vl = 0.0, vu = 0.0, orfac = 0.0;
		double abstol = -1.0; // ask the program to determine
        int *ifail = (int *)malloc(n * sizeof(int));
        if (strcmpi(typ, "gen") == 0) {
            // automem_pdsygvx_ ( 
            //     &ONE, "V", "A", "U", &n, Hp, &ONE, &ONE, descHp, 
            //     Mp, &ONE, &ONE, descMp, &vl, &vu, &il, &iu, &abstol, &M, &NZ,
            //     lambda, &orfac, Q, &ONE, &ONE, descQ, ifail, &info);
            pdsygvx_subcomm_ (
                &ONE, "V", "A", "U", &n, Hp, &ONE, &ONE, descHp,
                Mp, &ONE, &ONE, descMp, &vl, &vu, &il, &iu, &abstol, &M, &NZ,
                lambda, &orfac, Q, &ONE, &ONE, descQ, ifail, &info,
                rowcomm, subdims, blksz);
        } else if (strcmpi(typ, "std") == 0) {
            
        }
        if (info) printf("info = %d, ifail[0] = %d\n", info, ifail[0]);
        free(ifail);
    }

#ifdef PRINT_MAT
    // MPI_Barrier(rowcomm);
    // char fname[128] = "subspace_matrices.txt";
    output_fp = fopen(fname,"a");
    if (output_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",fname);
        exit(EXIT_FAILURE);
    }
    if (rank == 0) {
        fprintf(output_fp,"Q = [\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                fprintf(output_fp,"%.16e ", Q[j*n+i]);
            }
            fprintf(output_fp,"\n");
        }
        fprintf(output_fp,"];\n");
    }
    fclose(output_fp);
#endif



}


// X = Y * Q
void Subspace_Rotation_DDBP(
    int n, DDBP_ARRAY *Y, int *descY, double *Q, int *descQ, DDBP_ARRAY *X,
    int *descX, MPI_Comm comm
)
{
    DDBP_Array_Matrix_Multiply(
        n, 1.0, Y, descY, Q, descQ, 0.0, X, descX, comm);
}


// given the coefficients of the orbitals expressed in DDBP basis, recover
// the original orbital on the FD grid
void recover_orbitals_on_grid(
    DDBP_INFO *DDBP_info, const DDBP_ARRAY *X, int *descX, double **psi,
    int **desc_psi, const int nkpt, const int kpt, const int spn_i)
{
    // given a DDBP array, find it's representation on the FD grid
    int nelem = X->nelem;
    assert(nelem == DDBP_info->n_elem_elemcomm);

    for (int k = 0; k < nelem; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        // matrix size of basis in element E_k
        int nrow = E_k->nd_d;
        int ncol = DDBP_info->n_basis_basiscomm;
        int size_k = nrow * ncol;
        int size_s = size_k * nkpt;
        double *vk = E_k->v + spn_i*size_s + kpt*size_k;
        double *xk = X->array[k];
        double *psi_k = psi[k];
        int *desc_psi_k = desc_psi[k];
        DDBP_Element_Basis_Coeff_Multiply(
            vk, E_k->desc_v, xk, descX,
            psi_k, desc_psi_k, DDBP_info->elemcomm
        );
    }
}


// given orbitals on the grid within an element E_k, calculate
// the corresponding density (not reduced) within this element
// rho_k = sum_{n=nstart}^nend alpha * occ(n) * psi_k(:,n)^2
void calculate_element_density_psi(
    int nd_k, double alpha, double *occ, double *psi_k, int beta,
    double *rho_k, int nstart, int nend)
{
    assert(beta == 0 || beta == 1);

    for (int n = nstart; n <= nend; n++) {
        double g_n = alpha * occ[n];
        double *psi_n = psi_k + (n-nstart) * nd_k;
        if (beta == 0) {
            for (int i = 0; i < nd_k; i++) {
                rho_k[i] = g_n * psi_n[i] * psi_n[i];
            }
        } else if (beta == 1) {
            for (int i = 0; i < nd_k; i++) {
                rho_k[i] += g_n * psi_n[i] * psi_n[i];
            }
        }
    }
}


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
)
{
    // init rho to 0.0
    for (int spn_i = 0; spn_i < Nspin; spn_i++) {
        int sg = spn_i + spin_start_index;
        for (int k = 0; k < nelem; k++) {
            DDBP_ELEM *E_k = &elem_list[k];
            int nd_k = E_k->nd_d;
            double *rho_k = rho[k];
            if (spin_typ == 1) rho_k += nd_k * (sg+1);
            for (int i = 0; i < nd_k; i++) {
                rho_k[i] = 0.0;
            }
        }
    }

    double alpha = (spin_typ == 1) ? 1.0 : 2.0;
    alpha /= dV; // scale the density by 1/dV

    for (int spn_i = 0; spn_i < Nspin; spn_i++) {
        int sg = spn_i + spin_start_index;
        for (int kpt_i = 0; kpt_i < Nkpt; kpt_i++) {
            double *occ_ks = occ + spn_i*Nkpt*Nstates+kpt_i*Nstates;
            for (int k = 0; k < nelem; k++) {
                DDBP_ELEM *E_k = &elem_list[k];
                int nd_k = E_k->nd_d;
                // psi in element E_k
                double *psi_k = psi[spn_i][kpt_i][k];
                // rho in element E_k
                double *rho_k = rho[k];
                // for spin, there're 3 cols, 1st: total density, 2nd: spin-up, 3rd: spin-down
                if (spin_typ == 1) rho_k += nd_k * (sg+1);
                int beta = 1;
                calculate_element_density_psi(
                    nd_k, alpha, occ_ks, psi_k, beta,
                    rho_k, band_start_indx, band_end_indx);
            }
        }
    }

    // do an allreduce over all bands
    // TODO: another way is to allocate the memory for all elem's in one chunk
    // TODO: and then point rho[k] to the corresponding location, that way we
    // TODO: can do allreduce just once
    int nproc_rowcomm;
    MPI_Comm_size(rowcomm, &nproc_rowcomm);
    if (nproc_rowcomm > 1) {
        for (int k = 0; k < nelem; k++) {
            DDBP_ELEM *E_k = &elem_list[k];
            int nd_k = E_k->nd_d;
            MPI_Allreduce(MPI_IN_PLACE, rho[k], nd_k, MPI_DOUBLE, MPI_SUM, rowcomm);
        }
    }

    // TODO: do an allreduce over spin (spin_bridge_comm)
    // TODO: do an allreduce over all kpoints (kpt_bridge_comm)
}


void transfer_orbitals_E2D(SPARC_OBJ *pSPARC, double ****psi_E, double *Psi_D)
{
    // transfter density from elem distribution to domain distribution
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
    int Nstates = pSPARC->Nstates;
    int nspin = pSPARC->Nspin_spincomm;
    int nkpt = pSPARC->Nkpts_kptcomm;
    int DMnd = pSPARC->Nd_d_dmcomm;
    
    // element distribution to domain distribution
    int gridsizes[3] = {pSPARC->Nx, pSPARC->Ny, pSPARC->Nz};
    int BCs[3] = {pSPARC->BCx, pSPARC->BCy, pSPARC->BCz};
    int dmcomm_dims[3] = {pSPARC->npNdx, pSPARC->npNdy, pSPARC->npNdz};
    int send_ns = DDBP_info->band_start_index;
    int send_ncol = DDBP_info->n_band_bandcomm;
    int recv_ns = pSPARC->band_start_indx;
    int recv_ncol = pSPARC->Nband_bandcomm;
    int Edims[3] = {DDBP_info->Nex, DDBP_info->Ney, DDBP_info->Nez};
    
    int size_s = recv_ncol * DMnd;

    E2D_INFO E2D_info;
    for (int spn_i = 0; spn_i < pSPARC->Nspin; spn_i++) {
        E2D_Init(&E2D_info, Edims, DDBP_info->n_elem_elemcomm, DDBP_info->elem_list,
            gridsizes, BCs, Nstates,
            send_ns, send_ncol, DDBP_info->elemcomm, DDBP_info->npband, DDBP_info->elemcomm_index,
            DDBP_info->bandcomm, DDBP_info->npelem, DDBP_info->bandcomm_index,
            recv_ns, recv_ncol, pSPARC->DMVertices_dmcomm, pSPARC->blacscomm, pSPARC->npband, pSPARC->dmcomm,
            &dmcomm_dims[0], pSPARC->bandcomm_index, pSPARC->kptcomm
        );
/*
        // TODO: remove after check
        int nproc_wrldcomm;
        int rank_wrldcomm;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_wrldcomm);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc_wrldcomm);
        int rank = rank_wrldcomm;
        for (int i = 0; i < nproc_wrldcomm; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == rank) {
                printf("rank = %2d, ranks to send [n = %2d]: ", rank_wrldcomm, E2D_info.nproc_to_send);
                if (E2D_info.is_sender) {
                    print_array(E2D_info.ranks_to_send, E2D_info.nproc_to_send, sizeof(int));
                } else {
                    printf("\n");
                }
                // printf("rank = %2d, sendcounts    [n = %2d]: ", rank_wrldcomm, E2D_info.nproc_to_send);
                // if (E2D_info.is_sender) {
                //     print_array(E2D_info.sendcounts, E2D_info.nproc_to_send, sizeof(int));
                // } else {
                //     printf("\n");
                // }
            }
            // usleep(100000);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        extern void sleep();
        extern void usleep();
        usleep(300000);
        // sleep(1);

        for (int i = 0; i < nproc_wrldcomm; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == rank) {
                printf("rank = %2d, ranks to recv [n = %2d]: ", rank_wrldcomm, E2D_info.nproc_to_recv);
                if (E2D_info.is_recver) {
                    print_array(E2D_info.ranks_to_recv, E2D_info.nproc_to_recv, sizeof(int));
                } else {
                    printf("\n");
                }
                // printf("rank = %2d, recvcounts    [n = %2d]: ", rank_wrldcomm, E2D_info.nproc_to_recv);
                // if (E2D_info.is_recver) {
                //     print_array(E2D_info.recvcounts, E2D_info.nproc_to_recv, sizeof(int));
                // } else {
                //     printf("\n");
                // }
            }
            // usleep(100000);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // sleep(1);
        // exit(1);
        usleep(300000);
        for (int i = 0; i < nproc_wrldcomm; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == rank) {
                int rank_dmcomm = -1;
                int coords[3] = {-1,-1,-1};
                if (pSPARC->dmcomm != MPI_COMM_NULL) {
                    MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);
                    MPI_Cart_coords(pSPARC->dmcomm, rank_dmcomm, 3, coords);
                }
                printf("band+domain: bandcomm_index = %2d, coords = (%2d,%2d,%2d), rank_dmcomm = %2d, rank = %2d\n",
                    pSPARC->bandcomm_index, coords[0], coords[1], coords[2], rank_dmcomm, rank_wrldcomm);
            }
            // usleep(100000);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        usleep(300000);
        MPI_Barrier(MPI_COMM_WORLD);
        


        for (int i = 0; i < nproc_wrldcomm; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == rank) {
                int rank_elemcomm = -1;
                if (DDBP_info->elemcomm != MPI_COMM_NULL) {
                    MPI_Comm_rank(DDBP_info->elemcomm, &rank_elemcomm);
                }
                printf("elem+band: bandcomm_index = %2d, rank_elemcomm = %2d, npband = %d, rank = %2d\n",
                    DDBP_info->bandcomm_index, rank_elemcomm, DDBP_info->npband, rank_wrldcomm);
            }
            // usleep(100000);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        usleep(300000);
        MPI_Barrier(MPI_COMM_WORLD);
*/
        E2D_Iexec(&E2D_info, (const void **) psi_E[spn_i][0]);
        
        E2D_Wait(&E2D_info, Psi_D + spn_i*size_s);

        E2D_Finalize(&E2D_info);
    }
}

