/**
 * @file    cs.c
 * @brief   This file contains the functions for complementary subspace.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
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

#include "eigenSolver.h"
#include "tools.h" 
#include "linearSolver.h" // Lanczos
#include "lapVecRoutines.h"
#include "hamiltonianVecRoutines.h"
#include "occupation.h"
#include "isddft.h"
#include "parallelization.h"
#include "eigenSolver.h"
#include "cs.h"
#include "linearAlgebra.h"

#define TEMP_TOL 1e-12

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#ifdef USE_EVA_MODULE
#include "ExtVecAccel/ExtVecAccel.h"
int CheFSI_use_EVA = -1;
#endif


// #ifndef PRINT_MAT
// 	#define PRINT_MAT
// #endif

// #ifndef CHECK_CS_EIG
// 	#define CHECK_CS_EIG
// #endif

/**
 * @brief Find the bounds for the dense Chebyshev filtering on the subspace 
 *        Hamiltonian.
 *
 * @param eigmax_calc    The maximum calculated eigenvalue of the 
 *                       subspace Halmotonian.
 * @param eigmin         Minimum eigenvalue of the original Hamiltonian.
 * @param eigmin_calc    The previous minimum calculated eigenvalue of the 
 *                       subspace Hamiltonian, not referenced if isFirstIt = 0.
 * @param isFirstIt      Flag to check if this is the first Chebyshev iteration.
 */
void Chebyshevfilter_dense_constants(
	const SPARC_OBJ *pSPARC, const double eigmax_calc, const double eigmin, 
	const double eigmin_calc, const int isFirstIt, double *a, double *b, double *a0)
{
	// *a0 = eigmax_calc - 0.1;
	*a0 = eigmax_calc;
	*b = eigmin - 0.05;
	if (isFirstIt) {
		*a = (*a0 + *b) * 0.5;
		// *a = *b + 2.0; // assuming spectrum width is 2.0 (this is useful if a0 is very wrong)
	} else {
		*a = eigmin_calc - 0.05;
	}
}



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
	const double c, const double *X, double *Y, const MPI_Comm comm)
{
	double alpha, beta;
	alpha = 1.0;
	beta  = 0.0; 
	if (ncol < 5) {
		for (int i = 0; i < ncol; i++)
			cblas_dsymv(CblasColMajor, CblasUpper, N, alpha, Hs, N, X+i*N, 1, beta, Y+i*N, 1);
	} else {
		cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, N, ncol, alpha, Hs, N,
			X, N, beta, Y, N);
	}
	// add shift c * X
	if (fabs(c) > 1e-14) {
		const int nele = N * ncol;
		for (int i = 0; i < nele; i++) {
			Y[i] += c * X[i];
		}
	}
}

void init_CS(SPARC_OBJ *pSPARC) 
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // init CS
    int Nt = pSPARC->CS_Nt;
    int nv_local;
    if (pSPARC->bandcomm_index == -1) {
        nv_local = 0;
    } else {
        int NB =  ceil(Nt / (double)pSPARC->npband); 
        nv_local = pSPARC->bandcomm_index < (Nt / NB) ? NB :
        (pSPARC->bandcomm_index == (Nt / NB) ? (Nt % NB) : 0);
    }
    pSPARC->CS_nv_local = nv_local; // local number of subspace bands
    pSPARC->CS_Qt = (double *)malloc(pSPARC->Nstates * nv_local * sizeof(double));
    // initialize Qt
    srand(rank+1);
    for (int i = 0; i < pSPARC->Nstates * nv_local; i++) { // TODO: find actual distributed size of Qt in each process
        pSPARC->CS_Qt[i] = (double) rand() / RAND_MAX - 0.5;
    }

    // find the descriptor for Qt
    int mb_v, nb_v, m_loc, n_loc, lld_v, ZERO = 0, ONE = 1, info;
    int Ns = pSPARC->Nstates;
    int *descV = pSPARC->CS_descQt;
    int ictxt = pSPARC->ictxt_blacs;
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    nprow = 1;
    npcol = pSPARC->npband;
    mb_v = Ns; // block size
    nb_v = ceil(Nt / (double)pSPARC->npband);
    m_loc = numroc_(&Ns, &mb_v, &myrow, &ZERO, &nprow);
    n_loc = numroc_(&Nt, &nb_v, &mycol, &ZERO, &npcol);
    lld_v = max(1, m_loc);
    descinit_(descV, &Ns, &Nt, &mb_v, &nb_v, &ZERO, &ZERO, &ictxt, 
    &lld_v, &info);

	if (pSPARC->isGammaPoint){
        if (pSPARC->bandcomm_index != -1 && pSPARC->dmcomm != MPI_COMM_NULL) {
            pSPARC->Xorb_BLCYC = (double *)malloc(pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double));
            pSPARC->Yorb_BLCYC = (double *)malloc(pSPARC->nr_orb_BLCYC * pSPARC->nc_orb_BLCYC * sizeof(double));
        } else {
            pSPARC->Xorb_BLCYC = (double *)malloc(1 * sizeof(double));
            pSPARC->Yorb_BLCYC = (double *)malloc(1 * sizeof(double));
        }
    } else {
        assert(0);
    }
#endif // (USE_MKL or USE_SCALAPACK)  
}

/**
 * @brief   Free complementary subspace eigensolver.
 */
void free_CS(SPARC_OBJ *pSPARC)
{
	free(pSPARC->CS_Qt);
	#if defined(USE_MKL) || defined(USE_SCALAPACK)
    if (pSPARC->isGammaPoint) {
        free(pSPARC->Xorb_BLCYC);
        free(pSPARC->Yorb_BLCYC);
    }
#endif
}

struct CS_CheFSI_s
{
	double   *Mt_local;         // Local Mt result
	double   *Ht_local;         // Local Ht result
	double   *eig_vecs;         // Eigen vectors from solving generalized eigenproblem
	// MPI_Comm kpt_comm;          // MPI communicator that contains all active processes in pSPARC->kptcomm
};
typedef struct CS_CheFSI_s* CS_CheFSI_t;



/**
 * @brief   Initialize domain parallelization data structures for calculating projected Hamiltonian,  
 *          solving generalized eigenproblem, and performing subspace rotation in CheFSI().
 */
void init_CS_CheFSI(SPARC_OBJ *pSPARC)
{
	int proc_active = (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) ? 0 : 1;
	CS_CheFSI_t CS_CheFSI = (CS_CheFSI_t) malloc(sizeof(struct CS_CheFSI_s));
	free(CS_CheFSI);
#ifdef COMMENT
	#if defined(USE_MKL) || defined(USE_SCALAPACK)
	int ZERO = 0, info;
	// descriptors for Hp_local, Mp_local and eig_vecs (set block size to Ns_dp)
	descinit_(CS_CheFSI->desc_Hp_local, &Ns_dp, &Ns_dp, &Ns_dp, &Ns_dp, 
		&ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns_dp, &info);
	assert(info == 0);
	descinit_(CS_CheFSI->desc_Mp_local, &Ns_dp, &Ns_dp, &Ns_dp, &Ns_dp, 
		&ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns_dp, &info);
	assert(info == 0);
	descinit_(CS_CheFSI->desc_eig_vecs, &Ns_dp, &Ns_dp, &Ns_dp, &Ns_dp, 
		&ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns_dp, &info);
	assert(info == 0);
	#endif
#endif
}


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
	const int k, const int spn_i, const MPI_Comm comm, double *time_info)
{
	if (comm == MPI_COMM_NULL || ncol <= 0) return;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	#ifdef DEBUG   
	if(!rank) printf("Start dense Chebyshev filtering routine ... \n");
	#endif

	double t1, t2;
	*time_info = 0.0;

	double e, c, sigma, sigma1, sigma2, gamma, vscal, vscal2, *Ynew;
	int i, j, len_tot;
	len_tot = N * ncol;    
	e = 0.5 * (b - a);
	c = 0.5 * (b + a);
	sigma = sigma1 = e / (a0 - c);
	gamma = 2.0 / sigma1;

	t1 = MPI_Wtime();
	// find Y = (H - c*I)X
	int sg  = pSPARC->spin_start_indx + spn_i;
	mat_vectors_mult(pSPARC, N, Hs, ncol, -c, X, Y, comm);
	t2 = MPI_Wtime();
	*time_info += t2 - t1;

	// scale Y by (sigma1 / e)
	vscal = sigma1 / e;
	for (i = 0; i < len_tot; i++) Y[i] *= vscal;
	Ynew = (double *)malloc(len_tot * sizeof(double));

	for (j = 1; j < m; j++) {
		sigma2 = 1.0 / (gamma - sigma);
		
		t1 = MPI_Wtime();
		// Ynew = (H - c*I)Y
		mat_vectors_mult(pSPARC, N, Hs, ncol, -c, Y, Ynew, comm);
		t2 = MPI_Wtime();
		*time_info += t2 - t1;

		// Ynew = (2*sigma2/e) * Ynew - (sigma*sigma2) * X, then update X and Y
		vscal = 2.0 * sigma2 / e; vscal2 = sigma * sigma2;

		for (i = 0; i < len_tot; i++) {
			//Ynew[i] = vscal * Ynew[i] - vscal2 * X[i];
			Ynew[i] *= vscal;
			Ynew[i] -= vscal2 * X[i];
			X[i] = Y[i];
			Y[i] = Ynew[i];
		}
		sigma = sigma2;
	} 
	free(Ynew);

#ifdef PRINT_MAT
	if (rank == 0) {
		printf("local Y in rank %d\n", rank);
		for (int i = 0; i < min(N,12); i++) {
			for (int j = 0; j < min(ncol,10); j++) {
				printf("%15.8e ", Y[j*N+i]);
			}	
			printf("\n");
		}
	}
#endif

}


/**
 * @brief  Orthogonalize the subspace V using Cholesky factorization.
 *
 *         The CS Rayleight-Ritz process consists of the following steps:
 *           1. M = V'*V.
 *           2. Cholesky factorization M = R' * R.
 *           3. V = V * R^-1. 
 */
void orthChol(
	SPARC_OBJ *pSPARC, int Ns, int Nt, double *V, int ncol, int *descV, 
	MPI_Comm rowcomm)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// generate a (subset) process grid within rowcomm // TODO: move to CS init!
	extern int Csys2blacs_handle(MPI_Comm comm); // TODO: move this to scalapack.h
	int bhandle = Csys2blacs_handle(rowcomm); // create a context out of rowcomm
	int ictxt = bhandle;
	int nproc, gridsizes[2] = {Ns,Nt}, dims[2], ierr;
	MPI_Comm_size(rowcomm, &nproc);
	// for square matrices of size < 20000, it doesn't scale well beyond 64 proc
	ierr = 1; 
	int ishift = 8;
	// while (ierr && ishift) {
		SPARC_Dims_create(min(nproc,64), 2, gridsizes, 1<<ishift, dims, &ierr); // best min size 256
	// 	ishift--;
	// }
	if (ierr) dims[0] = dims[1] = 1;
	// TODO: swap dim[0] and dim[1] value, since SPARC_Dims_create tends to give larger dim for dim[1] on a tie situation
	#ifdef DEBUG
	if (rank == 0) printf("OrthChol: process grid = (%d, %d)\n", dims[0], dims[1]);
	#endif
	Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);

	int ictxt_rowcomm = descV[1];

	if (ictxt >= 0) {
		int nproc_grid, nprow, npcol, myrow, mycol;
		Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
		nproc_grid = nprow * npcol;

		// define new BLCYC distribution of V // TODO: move to CS init!
		int mb, nb, m_loc, n_loc, llda, ZERO = 0, ONE = 1, info, 
			descV_BLCYC[9], descVQ_BLCYC[9];
		mb = max(1, Ns/nprow); // block size
		nb = max(1, Nt/npcol);
		m_loc = numroc_(&Ns, &mb, &myrow, &ZERO, &nprow);
		n_loc = numroc_(&Nt, &nb, &mycol, &ZERO, &npcol);
		llda = max(1, m_loc);
		descinit_(descV_BLCYC, &Ns, &Nt, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info);
		assert(info == 0);
		for (int i = 0; i < 9; i++) descVQ_BLCYC[i] = descV_BLCYC[i];
		double *V_BLCYC  = (double *)malloc(m_loc*n_loc*sizeof(double));
		double *VQ_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double)); 

#ifdef PRINT_MAT
		if (rank == 0) {
			printf("orthChol (before): local V in rank %d\n", rank);
			for (int i = 0; i < Ns; i++) {
				for (int j = 0; j < min(ncol,12); j++) {
					printf("%15.8e ", V[j*Ns+i]);
				}	
				printf("\n");
			}
		}
#endif

		double t1, t2;

		t1 = MPI_Wtime();
		// convert V from BP to BLCYC
		pdgemr2d_(&Ns, &Nt, V, &ONE, &ONE, descV, V_BLCYC, &ONE, &ONE, 
			descV_BLCYC, &ictxt_rowcomm);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("OrthChol on V: V -> V_BLCYC: %.3f ms\n", (t2-t1)*1e3);
		#endif

		// find mass matrix
		// define BLCYC distribution of Mt // TODO: move to CS init!
		int descMt_BLCYC[9];
		mb = nb = max(1, Nt/max(nprow,npcol)); // block size mb must equal nb
		m_loc = numroc_(&Nt, &mb, &myrow, &ZERO, &nprow);
		n_loc = numroc_(&Nt, &nb, &mycol, &ZERO, &npcol);
		llda = max(1, m_loc);
		descinit_(descMt_BLCYC, &Nt, &Nt, &mb, &nb, &ZERO, &ZERO, &ictxt, 
			&llda, &info);
		assert(info == 0);

		double *Mt_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));
		assert(Mt_BLCYC != NULL);
		

		t1 = MPI_Wtime();
		// Mt = V' * V
		double alpha = 1.0, beta = 0.0;
		pdsyrk_("U", "T", &Nt, &Ns, &alpha, V_BLCYC, &ONE, &ONE, descV_BLCYC,
			&beta, Mt_BLCYC, &ONE, &ONE, descMt_BLCYC);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("OrthChol on V: Mt = V' * V: %.3f ms\n", (t2-t1)*1e3);
		#endif

		t1 = MPI_Wtime();
		// Mt = R' * R (Cholesky factorization)
		pdpotrf_("U", &Nt, Mt_BLCYC , &ONE, &ONE, descMt_BLCYC, &info);
		assert(info == 0);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("OrthChol on V: Mt = R' * R: %.3f ms\n", (t2-t1)*1e3);
		#endif

		t1 = MPI_Wtime();
		// V = V / R
		pdtrsm_ ("R", "U", "N", "N", &Ns, &Nt, &alpha, Mt_BLCYC, &ONE, &ONE, 
			descMt_BLCYC, V_BLCYC, &ONE, &ONE, descV_BLCYC);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("OrthChol on V: V = V / R: %.3f ms\n", (t2-t1)*1e3);
		#endif
		t1 = MPI_Wtime();
		// convert V_BLCYC back to BP
		pdgemr2d_(&Ns, &Nt, V_BLCYC, &ONE, &ONE, descV_BLCYC, V, &ONE, &ONE, descV, &ictxt_rowcomm);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("OrthChol on V: V_BLCYC -> V: %.3f ms\n", (t2-t1)*1e3);
		#endif

#ifdef PRINT_MAT
		if (rank == 0) {
			printf("orthChol (after): local V in rank %d\n", rank);
			for (int i = 0; i < Ns; i++) {
				for (int j = 0; j < min(ncol,12); j++) {
					printf("%15.8e ", V[j*Ns+i]);
				}	
				printf("\n");
			}
		}
#endif
        free(V_BLCYC);
        free(VQ_BLCYC);
        free(Mt_BLCYC);
		Cblacs_gridexit(ictxt);
	} else {
		int ONE = 1, descV_BLCYC[9], descHV_BLCYC[9], descVQ_BLCYC[9];
		double *V_BLCYC, *VQ_BLCYC;
		for (int i = 0; i < 9; i++) 
			descV_BLCYC[i] = descHV_BLCYC[i] = descVQ_BLCYC[i] = -1;
		pdgemr2d_(&Ns, &Nt, V, &ONE, &ONE, descV, V_BLCYC, &ONE, &ONE, descV_BLCYC, &ictxt_rowcomm);
		pdgemr2d_(&Ns, &Nt, V_BLCYC, &ONE, &ONE, descV_BLCYC, V, &ONE, &ONE, descV, &ictxt_rowcomm);
	}
#endif // (USE_MKL or USE_SCALAPACK)    
}






/**
 * @brief  Perform a Rayleigh-Ritz step to find the approx. eigen pair of the 
 *         subspace Hamiltonian Hs in the subspace V.
 *
 *         The CS Rayleight-Ritz process consists of the following steps:
 *           1. Ht = V'*Hs*V (assuming V'*V = I).
 *           2. Ht * Qt_s = Qt_s * Lambda_t.
 *           3. VQ = Y * Qt_s.
 */
void CS_Rayleigh_Ritz_process(
	SPARC_OBJ *pSPARC, int Ns, int Nt, double *Hs, double *V, int ncol, 
	double *w, double *VQ, int *descV, MPI_Comm colcomm, MPI_Comm rowcomm)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double t1, t2;

	int descHV[9];
	for (int i = 0; i < 9; i++) descHV[i] = descV[i];
	double *HV = (double *)malloc(Ns * ncol * sizeof(double));
	assert(HV != NULL);

	t1 = MPI_Wtime();
	mat_vectors_mult(pSPARC, Ns, Hs, ncol, 0.0, V, HV, colcomm);
	t2 = MPI_Wtime();
	#ifdef DEBUG
	if (!rank) printf("CS_Rayleigh_Ritz_process: H * V: %.3f ms\n", (t2-t1)*1e3);
	#endif

	// generate a (subset) process grid within rowcomm // TODO: move to CS init!
	extern int Csys2blacs_handle(MPI_Comm comm); // TODO: move this to scalapack.h
	int bhandle = Csys2blacs_handle(rowcomm); // create a context out of rowcomm
	int ictxt = bhandle;
	int nproc, gridsizes[2] = {Ns,Nt}, dims[2], ierr;
	MPI_Comm_size(rowcomm, &nproc);
	// for square matrices of size < 20000, it doesn't scale well beyond 64 proc
	SPARC_Dims_create(min(nproc,64), 2, gridsizes, 256, dims, &ierr); // TODO: set min size to 256
	if (ierr) dims[0] = dims[1] = 1;
	// TODO: swap dim[0] and dim[1] value, since SPARC_Dims_create tends to give larger dim for dim[1] on a tie situation

	#ifdef DEBUG
	if (rank == 0) 
		printf("CS_Rayleigh_Ritz_process: process grid = (%d, %d)\n", dims[0], dims[1]);
	#endif
	Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);

	int ictxt_rowcomm = descV[1];
	// perform Rayleigh step 
	if (ictxt >= 0) {
		int nproc_grid, nprow, npcol, myrow, mycol;
		Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
		nproc_grid = nprow * npcol;

		// define new BLCYC distribution of V // TODO: move to CS init!
		int mb, nb, m_loc, n_loc, llda, ZERO = 0, ONE = 1, info, 
			descV_BLCYC[9], descHV_BLCYC[9], descVQ_BLCYC[9];
		mb = max(1, Ns/nprow); // block size
		nb = max(1, Nt/npcol);
		m_loc = numroc_(&Ns, &mb, &myrow, &ZERO, &nprow);
		n_loc = numroc_(&Nt, &nb, &mycol, &ZERO, &npcol);
		llda = max(1, m_loc);
		descinit_(descV_BLCYC, &Ns, &Nt, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info);
		assert(info == 0);
		for (int i = 0; i < 9; i++) 
			descHV_BLCYC[i] = descVQ_BLCYC[i] = descV_BLCYC[i];
		double *V_BLCYC  = (double *)malloc(m_loc*n_loc*sizeof(double));
		double *HV_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));
		double *VQ_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double)); 

#ifdef PRINT_MAT
		if (rank == 0) {
			printf("local V in rank %d\n", rank);
			for (int i = 0; i < Ns; i++) {
				for (int j = 0; j < min(ncol,12); j++) {
					printf("%15.8e ", V[j*Ns+i]);
				}	
				printf("\n");
			}
		}
#endif

		t1 = MPI_Wtime();
		// convert V from BP to BLCYC
		pdgemr2d_(&Ns, &Nt, V, &ONE, &ONE, descV, V_BLCYC, &ONE, &ONE, 
			descV_BLCYC, &ictxt_rowcomm);
		pdgemr2d_(&Ns, &Nt, HV, &ONE, &ONE, descHV, HV_BLCYC, &ONE, &ONE, 
			descHV_BLCYC, &ictxt_rowcomm);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("CS_Rayleigh_Ritz_process: VQ_BLCYC -> VQ: %.3f ms\n", (t2-t1)*1e3);
		#endif
		// find mass matrix
		// define BLCYC distribution of Ht // TODO: move to CS init!
		int descHt_BLCYC[9];
		mb = nb = max(1, Nt/max(nprow,npcol)); // block size mb must equal nb
		m_loc = numroc_(&Nt, &mb, &myrow, &ZERO, &nprow);
		n_loc = numroc_(&Nt, &nb, &mycol, &ZERO, &npcol);
		llda = max(1, m_loc);
		descinit_(descHt_BLCYC, &Nt, &Nt, &mb, &nb, &ZERO, &ZERO, &ictxt, 
			&llda, &info);
		assert(info == 0);

		t1 = MPI_Wtime();
		// Ht = V' * H * V
		// find projected partial subspace Hamiltonian
		double *Ht_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));
		assert(Ht_BLCYC != NULL);
// 		double alpha = 0.5, beta = 0.0;
// 		extern void pdsyr2k_();
// 		pdsyr2k_("U", "T", &Nt, &Ns, &alpha, V_BLCYC, &ONE, &ONE, descV_BLCYC,
// 			HV_BLCYC, &ONE, &ONE, descHV_BLCYC,
// 			&beta, Ht_BLCYC, &ONE, &ONE, descHt_BLCYC);

// #ifdef PRINT_MAT
// 		if (rank == 0) {
// 			printf("local Ht (pdsyr2k_) in rank %d\n", rank);
// 			for (int i = 0; i < min(m_loc,10); i++) {
// 				for (int j = 0; j < min(n_loc,10); j++) {
// 					printf("%15.8e ", Ht_BLCYC[j*m_loc+i]);
// 				}	
// 				printf("\n");
// 			}
// 		}
// #endif
		
		double alpha = 1.0, beta = 0.0;
		pdgemm_("T", "N", &Nt, &Nt, &Ns, &alpha, V_BLCYC, &ONE, &ONE, descV_BLCYC,
				HV_BLCYC, &ONE, &ONE, descHV_BLCYC, &beta, Ht_BLCYC, &ONE, &ONE, descHt_BLCYC);
		// printf("local Ht (pdgemm_) in rank %d\n", rank);
		// for (int i = 0; i < m_loc; i++) {
		// 	for (int j = 0; j < n_loc; j++) {
		// 		printf("%15.8e ", Ht_BLCYC[j*m_loc+i]);
		// 	}	
		// 	printf("\n");
		// }
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("CS_Rayleigh_Ritz_process: Ht = V' * H * V: %.3f ms\n", (t2-t1)*1e3);
		#endif

		// solve the complementary subspace eigenproblem
		// assume Qt has the same distribution as Mt
		int descQt_s_BLCYC[9];
		for (int i = 0; i < 9; i++) descQt_s_BLCYC[i] = descHt_BLCYC[i];
		// double *w = (double *)malloc(Nt*sizeof(double));
		double *Qt_s_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));
		assert(Qt_s_BLCYC != NULL);
		int il = 1, iu = 1, M, NZ;
		double vl = 0.0, vu = 0.0, abstol;

		int isSerial = (Nt < pSPARC->eig_serial_maxns || nproc_grid == 1) ? 1 : 0; // consider ictxt size too?
		
		isSerial = 0; // TODO: remove this after check
		t1 = MPI_Wtime();
		if (isSerial) {
			// TODO: implement sequential calculation of dense full eigen solve
			LAPACKE_dsyevd (LAPACK_COL_MAJOR, 'V', 'U', Nt, Ht_BLCYC, Nt, w);
			for (int i = 0; i < m_loc*n_loc; i++) {
				Qt_s_BLCYC[i] = Ht_BLCYC[i];
			}
		} else {
			automem_pdsyevd_ ( 
				"V", "U", &Nt, Ht_BLCYC, &ONE, &ONE, descHt_BLCYC, 
				w, Qt_s_BLCYC, &ONE, &ONE, descQt_s_BLCYC, &info);
			if (info) printf("info = %d\n", info);
		}
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("CS_Rayleigh_Ritz_process: eig(Ht): %.3f ms\n", (t2-t1)*1e3);
		#endif
#ifdef PRINT_MAT
		if (!rank) {
			printf("w = \n");
			for (int i = 0; i < Nt; i++) printf(RED "%.10f \n" RESET, w[i]);
			printf("local Z (CS eigenvectors) in rank %d\n", rank);
			for (int i = 0; i < min(m_loc,12); i++) {
				for (int j = 0; j < min(n_loc,12); j++) {
					printf("%15.8e ", Qt_s_BLCYC[j*m_loc+i]);
				}	
				printf("\n");
			}
		}
#endif

		t1 = MPI_Wtime();
		// complementary subspace rotation V = V * Qt_s
		alpha = 1.0; beta = 0.0;
		pdgemm_("N", "N", &Ns, &Nt, &Nt, &alpha, V_BLCYC, &ONE, &ONE, descV_BLCYC,
			Qt_s_BLCYC, &ONE, &ONE, descQt_s_BLCYC, &beta, VQ_BLCYC, &ONE, &ONE, descVQ_BLCYC);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("CS_Rayleigh_Ritz_process: V = V * Qt_s: %.3f ms\n", (t2-t1)*1e3);
		#endif

		t1 = MPI_Wtime();
		// convert VQ from BLCYC to BP
		pdgemr2d_(&Ns, &Nt, VQ_BLCYC, &ONE, &ONE, descVQ_BLCYC, VQ, &ONE, &ONE, descV, &ictxt_rowcomm);
		t2 = MPI_Wtime();
		#ifdef DEBUG
		if (!rank) printf("CS_Rayleigh_Ritz_process: VQ_BLCYC -> VQ: %.3f ms\n", (t2-t1)*1e3);
		#endif

#ifdef PRINT_MAT
		if (rank == 0) {
			printf("local VQ in rank %d\n", rank);
			for (int i = 0; i < Ns; i++) {
				for (int j = 0; j < min(ncol,12); j++) {
					printf("%14.10f ", VQ[j*Ns+i]);
				}	
				printf("\n");
			}
		}
#endif

		free(Qt_s_BLCYC); 
		// free(Mt_BLCYC);
		free(Ht_BLCYC);
		free(V_BLCYC);
		free(HV_BLCYC);
		free(VQ_BLCYC);
		Cblacs_gridexit(ictxt);
	} else {
		int ONE = 1, descV_BLCYC[9], descHV_BLCYC[9], descVQ_BLCYC[9];
		double *V_BLCYC, *HV_BLCYC, *VQ_BLCYC;
		for (int i = 0; i < 9; i++) 
			descV_BLCYC[i] = descHV_BLCYC[i] = descVQ_BLCYC[i] = -1;
		pdgemr2d_(&Ns, &Nt, V, &ONE, &ONE, descV, V_BLCYC, &ONE, &ONE, descV_BLCYC, &ictxt_rowcomm);
		pdgemr2d_(&Ns, &Nt, HV, &ONE, &ONE, descHV, HV_BLCYC, &ONE, &ONE, descHV_BLCYC, &ictxt_rowcomm);
		pdgemr2d_(&Ns, &Nt, VQ_BLCYC, &ONE, &ONE, descVQ_BLCYC, VQ, &ONE, &ONE, descV, &ictxt_rowcomm);
	}

	t1 = MPI_Wtime();
	MPI_Bcast(w, Nt, MPI_DOUBLE, 0, rowcomm);
	t2 = MPI_Wtime();
	#ifdef DEBUG
	if (!rank) printf("CS_Rayleigh_Ritz_process: Bcast w: %.3f ms\n", (t2-t1)*1e3);
	#endif

	free(HV);
#endif // (USE_MKL or USE_SCALAPACK)
}




/**
 * @brief   Chebyshev-filtered subspace iteration eigensolver.
 */
void CheFSI_dense_eig(
	SPARC_OBJ *pSPARC, int Ns, int Nt, double lambda_cutoff, int repeat_chefsi, 
	int count, int k, int spn_i)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
	int rank, rank_blacscomm, nproc_kptcomm, nproc_blacscomm;
	MPI_Comm_rank(MPI_COMM_WORLD,   &rank);
	MPI_Comm_rank(pSPARC->blacscomm,&rank_blacscomm);
	MPI_Comm_size(pSPARC->kptcomm,  &nproc_kptcomm);
	MPI_Comm_size(pSPARC->blacscomm,&nproc_blacscomm);
	
	double t1, t2, t3, t_temp;



	// find the distribution of the top Nt eigenvectors assigned to each 
	// bandcomm, this is a special case of block-cyclic distribution (*,block)
	// TODO: move to CS init
	int nv_local;
	if (pSPARC->bandcomm_index == -1) {
		nv_local = 0;
	} else {
		int NB =  ceil(Nt / (double)pSPARC->npband); 
		nv_local = pSPARC->bandcomm_index < (Nt / NB) ? NB :
			(pSPARC->bandcomm_index == (Nt / NB) ? (Nt % NB) : 0);
	}

	// define descriptor for V in BP distribution // TODO: move to CS init
	int mb_v, nb_v, m_loc, n_loc, lld_v, ZERO = 0, ONE = 1, info, descV[9];
	int ictxt = pSPARC->ictxt_blacs;
	int nprow, npcol, myrow, mycol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
	nprow = 1;
	npcol = pSPARC->npband;
	mb_v = Ns; // block size
	nb_v = ceil(Nt / (double)pSPARC->npband);
	m_loc = numroc_(&Ns, &mb_v, &myrow, &ZERO, &nprow);
	n_loc = numroc_(&Nt, &nb_v, &mycol, &ZERO, &npcol);
	lld_v = max(1, m_loc);
	descinit_(descV, &Ns, &Nt, &mb_v, &nb_v, &ZERO, &ZERO, &ictxt, 
		&lld_v, &info);

	// allocate memory for the assigned local eigvecs
	size_t Nt_bp_msize = Ns * nv_local * sizeof(double);
	// double *X = (double *)malloc(Nt_bp_msize);
	double *X = pSPARC->CS_Qt;
	double *Y = (double *)malloc(Nt_bp_msize);
	// assert(X != NULL);
	assert(Y != NULL);

	// // initialize X 
	// srand(rank+1);
	// for (int i = 0; i < Ns * nv_local; i++)
	// 	X[i] = (double) rand() / RAND_MAX - 0.5;

	// prepare the dense matrix Hs in each process
	t1 = MPI_Wtime();
	double *Hs;
	#ifdef USE_DP_SUBEIG
	
	DP_CheFSI_t DP_CheFSI = (DP_CheFSI_t) pSPARC->DP_CheFSI;
	if (DP_CheFSI != NULL)
		Hs = DP_CheFSI->Hp_local; // "allreduced" in standard DP projection routine
	
	#else // USE_DP_SUBEIG
	
	Hs = (double *)malloc(Ns * Ns * sizeof(double));;
	if (pSPARC->useLAPACK == 1) {
		// in this case only rank_blacscomm 0 has the matrix
		if (rank_blacscomm == 0) {
			const int Ns2 = Ns * Ns;
			for (int i = 0; i < Ns2; i++) {
				Hs[i] = pSPARC->Hp[i];
			}
		}
	} else {
		int desc_Hs[9];
		descinit_(desc_Hs, &Ns, &Ns, &Ns, &Ns, 
			&ZERO, &ZERO, &pSPARC->ictxt_blacs_topo, &Ns, &info);
		assert(info == 0);
		pdgemr2d_(&Ns, &Ns, pSPARC->Hp, &ONE, &ONE, pSPARC->desc_Hp_BLCYC, 
			Hs, &ONE, &ONE, desc_Hs, &pSPARC->ictxt_blacs_topo);
	}
	MPI_Bcast(Hs, Ns*Ns, MPI_DOUBLE, 0, pSPARC->blacscomm);
	#endif // USE_DP_SUBEIG

	t2 = MPI_Wtime();
	#ifdef DEBUG
	if (!rank) printf("CheFSI_dense_eig: Time for collecting and bcasting Hs: %.3f ms\n", (t2-t1)*1e3);
	#endif

	// find trace of Hs
	double tr_Hp_k = 0.0;
	t1 = MPI_Wtime();
	for (int i = 0; i < Ns; i++) tr_Hp_k += Hs[i*Ns+i];
	pSPARC->tr_Hp_k[spn_i*pSPARC->Nkpts_kptcomm] = tr_Hp_k;
	t2 = MPI_Wtime();
	#ifdef DEBUG
	if(!rank) printf(GRN "rank = %d, trace of Hs = %.15f, time for finding trace of Hs: %.3f ms\n" RESET, 
					 rank, tr_Hp_k, (t2 - t1)*1e3);
	#endif


	// use Lanczos to estimate minimum eigenvalue of Hs
	t1 = MPI_Wtime();
	double TOL_eigmin, TOL_eigmax, eigmin, eigmax, *x0;
	TOL_eigmin = pSPARC->TOL_LANCZOS; TOL_eigmax = 0.1;
	// x0 = pSPARC->CS_x0;
	if (rank_blacscomm == 0) {
		Lanczos_dense_seq(pSPARC, Hs, Ns, &eigmin, &eigmax, x0, TOL_eigmin, TOL_eigmax, 
			50, k, spn_i, MPI_COMM_SELF);
	}
	double buf[2] = {eigmin,eigmax};
	MPI_Bcast(buf, 2, MPI_DOUBLE, 0, pSPARC->blacscomm);
	eigmin = buf[0]; eigmax = buf[1];
	pSPARC->lambda[spn_i*Ns] = eigmin;
	t2 = MPI_Wtime();
	#ifdef DEBUG
	if (!rank) 
		printf("CheFSI_dense_eig: Estimated extreme eigvals of Hs: eigmin = %.15f, eigmax = %f, "
			"Time for Lanczos_dense_seq: %.3f ms", 
			eigmin, eigmax, (t2-t1)*1e3);
	#endif

	int isFirstIt = (int) (pSPARC->elecgs_Count == 0 && count == 0);
	double a, b, a0;
	// determine the constants for performing chebyshev filtering
	double lambda_N1 = pSPARC->lambda[spn_i*Ns+Ns-Nt];
	// Chebyshevfilter_dense_constants(pSPARC, lambda_cutoff, pSPARC->eigmin[spn_i], 
	// 	lambda_N1, isFirstIt, &a, &b, &a0);
	double eigmax_calc = isFirstIt ? eigmax : lambda_cutoff-0.1;
	Chebyshevfilter_dense_constants(pSPARC, eigmax_calc, pSPARC->eigmin[spn_i], 
		lambda_N1, isFirstIt, &a, &b, &a0);
	
#ifdef DEBUG
	if (rank == 0) 
		printf("\nrank = %d, In Chebyshev filtering dense routine, a = %f, b = %f, a0 = %f\n",
		 rank, a, b, a0);
#endif


#ifdef PRINT_MAT
	if (rank == 0) {
		printf("local X in rank %d\n", rank);
		for (int i = 0; i < min(Ns,12); i++) {
			for (int j = 0; j < min(nv_local,12); j++) {
				printf("%15.8e ", X[j*Ns+i]);
			}	
			printf("\n");
		}
	}
#endif

	// ** Chebyshev filtering ** //
	int m = pSPARC->CS_npl, n = repeat_chefsi;
	double t_orth_s, t_orth_e, t_orth = 0.0, t_matmat = 0.0;
	t1 = MPI_Wtime();
	for (int r = 0; r < n; r++) {
		// X = Y
		if (r > 0) for (int i = 0; i < Ns * nv_local; i++) X[i] = Y[i];
		
		// Y = pm(Hs) * X
		ChebyshevFiltering_dense(pSPARC, Hs, Ns, X, Y, nv_local, m, a, b, a0, k,
			spn_i, pSPARC->dmcomm, &t_temp);
		t_matmat += t_temp;

		// orth Y
		t_orth_s = MPI_Wtime();
		orthChol(pSPARC, Ns, Nt, Y, nv_local, descV, pSPARC->blacscomm);
		t_orth_e = MPI_Wtime(); t_orth += (t_orth_e - t_orth_s);
	}
	t2 = MPI_Wtime();
	#ifdef DEBUG
	if(!rank) 
		printf("CheFSI_dense_eig: Time for dense Chebyshev filtering (%d col, deg = %d, rep = %d): "
			   "total time: %.3f ms (Hs_mat_vec time: %.3f ms, orthChol time: %.3f ms).\n", 
				nv_local, m, n, (t2-t1)*1e3, t_matmat*1e3, t_orth*1e3);
	#endif

	// normalize Y components (in the order of 1e-13)
	// if (nv_local && Y[0] != 0.0)  {
	// 	double scale_fac = 1e-13;
	// 	if (!rank) printf(GRN "Scaling the Y components ...\n" RESET);
	// 	for (int i = 0; i < Ns * nv_local; i++) {
	// 		Y[i] /= scale_fac;
	// 	}
	// }

	// perform a Rayleigh-Ritz step to find the approx. eigenpairs
	// 1. Ht = Y'*Hs*Y, (Mt = Y'*Y = I)
	// 2. Ht * Qt_s = Qt_s * Lambda_t
	// 3. X = Y * Qt_s
	t1 = MPI_Wtime();
	int il = Ns-Nt+1;
	double *w = pSPARC->lambda + spn_i*Ns + il-1;
	CS_Rayleigh_Ritz_process(pSPARC, Ns, Nt, Hs, Y, nv_local, w, X, descV,
		pSPARC->dmcomm, pSPARC->blacscomm);
	t2 = MPI_Wtime();
	#ifdef DEBUG
	if (!rank) 
		printf(BLU "Time for CS_Rayleigh_Ritz_process: %.3f ms\n" RESET, 
			(t2-t1)*1e3);
	#endif

#ifdef PRINT_MAT
	if (rank == 0) {
		printf("local X in rank %d\n", rank);
		for (int i = 0; i < min(Ns,12); i++) {
			for (int j = 0; j < min(nv_local,12); j++) {
				printf("%15.8e ", X[j*Ns+i]);
			}	
			printf("\n");
		}
	}
#endif

	// copy X back to the Qt matrix
	t1 = MPI_Wtime();
	// distribute eigenvectors to block cyclic format 
	pdgemr2d_(&Ns, &Nt, X, &ONE, &ONE, descV, pSPARC->Q, &ONE, &il, 
			  pSPARC->desc_Q_BLCYC, &pSPARC->ictxt_blacs_topo);
	t2 = MPI_Wtime();
	#ifdef DEBUG
	if(!rank) {
		printf("==standard eigenproblem: "
			   "distribute subspace eigenvectors into block cyclic format: %.3f ms\n", 
			   (t2 - t1)*1e3);
	}
	#endif



#ifdef CHECK_CS_EIG
	if (rank == 0) {
		t1 = MPI_Wtime();
		// TODO: remove after check
		int iu, M, *ifail;
		double vl, vu, abstol, *Z_temp, *lambda_temp;
		lambda_temp = (double *)malloc(Ns * sizeof(double));
		Z_temp = (double *)malloc(Ns * Ns * sizeof(double));

		ifail = (int *)malloc(Ns * sizeof(int));
		info = LAPACKE_dsyevx(LAPACK_COL_MAJOR, 'V', 'A', 'U', Ns, Hs, Ns, 
				vl, vu, il, iu, abstol, &M, lambda_temp, 
				Z_temp, Ns, ifail);
		assert(info == 0);

		for (int i = 0; i < Ns; i++) {
			double lambda_calc = *(pSPARC->lambda+spn_i*Ns+i);
			double lambda_ref = lambda_temp[i];
			double err_lambda = fabs(lambda_calc - lambda_ref); 
			if (i < 1 || i >= Ns-Nt)
				printf(GRN "lambda[%2d] = %20.15f, lambda_temp = %20.15f, err = %.3e\n" RESET, i+1, lambda_calc, lambda_ref, err_lambda);
			else 
				printf(WHT "lambda[%2d] = %20.15f, lambda_temp = %20.15f, err = %.3e\n" RESET, i+1, lambda_calc, lambda_ref, err_lambda);
		}
		free(ifail);
		free(lambda_temp);
		free(Z_temp);
		t2 = MPI_Wtime();
		printf(RED "Time for CS eigen solution checking: %.3f ms\n" RESET, (t2-t1)*1e3);
	}

#endif

	#ifndef USE_DP_SUBEIG
	free(Hs);
	#endif
	free(Y);
#endif // (USE_MKL or USE_SCALAPACK)
}




/**
 * @brief   Solve subspace eigenproblem Hp * x = lambda * x for the top Nt 
 *          eigenvalues/eigenvectors using the CheFSI algorithm.
 *
 *          Note: Hp = Psi' * H * Psi, where Psi' * Psi = I. 
 *          
 */
void Solve_partial_standard_EigenProblem_CheFSI(
	SPARC_OBJ *pSPARC, double lambda_cutoff, int Nt, int repeat_chefsi, int k, 
	int count, int spn_i) 
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
	int rank, rank_spincomm, rank_kptcomm;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
	MPI_Comm_rank(pSPARC->kptcomm,  &rank_kptcomm);
	#ifdef DEBUG
	if (!rank) printf("Start solving standard eigenvalue problem ...\n");
	#endif
	if (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) return;

	double t1, t2;
	#ifdef DEBUG
	double st = MPI_Wtime();
	#endif

	int Ns = pSPARC->Nstates;

	// find trace of Hs
	// double tr_Hp_k = 0.0;
	// t1 = MPI_Wtime();
	// if (pSPARC->useLAPACK) {
	// 	if(pSPARC->bandcomm_index == 0) // TODO: check if needed to bcast to all proc
	// 		for (int i = 0; i < Ns; i++) tr_Hp_k += pSPARC->Hp[i*Ns+i];
	// } else {
	// 	int ONE = 1;
	// 	tr_Hp_k = pdlatra_(&Ns, pSPARC->Hp, &ONE, &ONE, pSPARC->desc_Hp_BLCYC);
	// }
	// pSPARC->tr_Hp_k[spn_i*pSPARC->Nkpts_kptcomm] = tr_Hp_k;
	// t2 = MPI_Wtime();
	// #ifdef DEBUG
	// if(!rank) printf(GRN "rank = %d, trace of Hs = %.15f, time for finding trace of Hs: %.3f ms\n" RESET, 
	// 				 rank, tr_Hp_k, (t2 - t1)*1e3);
	// #endif

	CheFSI_dense_eig(pSPARC, Ns, Nt, lambda_cutoff, repeat_chefsi, count, k, spn_i);
	
	#ifdef DEBUG
	double et = MPI_Wtime();
	if (rank == 0) printf("rank = %d, Solve_partial_standard_EigenProblem used %.3lf ms\n", rank, 1000.0 * (et - st));
	#endif
		
#else // #if defined(USE_MKL) || defined(USE_SCALAPACK)
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) printf("[FATAL] Subspace eigenproblem are using ScaLAPACK routines but ScaLAPACK is not compiled\n");
	if (rank == 0) printf("\nPlease turn on USE_MKL or USE_SCALAPACK!\n");
	exit(255);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
} 



/**
 * @brief   Lanczos algorithm for calculating min and max eigenvalues
 *          for a symmetric dense matrix.
 */
void Lanczos_dense_seq(
	SPARC_OBJ *pSPARC, const double *A, const int N, 
	double *eigmin, double *eigmax, double *x0, const double TOL_min, 
	const double TOL_max, const int MAXIT, int k, int spn_i, MPI_Comm comm
) 
{
	double t1, t2;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	#ifdef DEBUG
	if (rank == 0) printf("\nStart dense Lanczos algorithm ...\n");
	#endif

	int rank_comm; MPI_Comm_rank(comm, &rank_comm);

	double vscal, err_eigmin, err_eigmax, eigmin_pre, eigmax_pre;
	double *V_j, *V_jm1, *V_jp1, *a, *b, *d, *e;
	int i, j, DMnd;
	DMnd = N;
	
	V_j   = (double*)malloc( DMnd * sizeof(double));
	V_jm1 = (double*)malloc( DMnd * sizeof(double));
	V_jp1 = (double*)malloc( DMnd * sizeof(double));
	a     = (double*)malloc( (MAXIT+1) * sizeof(double));
	b     = (double*)malloc( (MAXIT+1) * sizeof(double));
	d     = (double*)malloc( (MAXIT+1) * sizeof(double));
	e     = (double*)malloc( (MAXIT+1) * sizeof(double));

	/* set random initial guess vector V_jm1 with unit 2-norm */
	srand(rank_comm+1); 
 
	double rand_min = -1.0, rand_max = 1.0;
	for (i = 0; i < DMnd; i++) {
		//V_jm1[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
		//TODO: [1,...,1] might be a better guess for it's closer to the eigvec for Lap 
		//      with zero (or ~= zero) eigval, and since min eig is harder to converge
		//      this will make the overall convergence faster.
		V_jm1[i] = 1.0 - 1e-3 + 2e-3 * (double) rand() / RAND_MAX; // TODO: Move this to CS init!
		// V_jm1[i] = x0[i];
	}

	Vector2Norm(V_jm1, DMnd, &vscal, comm); // find norm of V_jm1
	vscal = 1.0 / vscal;
	// scale the random guess vector s.t. V_jm1 has unit 2-norm
	for (i = 0; i < DMnd; i++) 
		V_jm1[i] *= vscal;

	// calculate V_j = H * V_jm1
	t1 = MPI_Wtime();
	//Lap_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, V_jm1, V_j, comm);
	mat_vectors_mult(pSPARC, DMnd, A, 1, 0.0, V_jm1, V_j, comm);
	t2 = MPI_Wtime();
#ifdef DEBUG
	if(!rank && spn_i == 0) printf("rank = %2d, One H*x took %.3f ms\n", rank, (t2-t1)*1e3);   
#endif
	// find dot product of V_jm1 and V_j, and store the value in a[0]
	VectorDotProduct(V_jm1, V_j, DMnd, &a[0], comm);

	// orthogonalize V_jm1 and V_j
	for (i = 0; i < DMnd; i++)
		V_j[i] -= a[0] * V_jm1[i];
	
	// find norm of V_j
	Vector2Norm(V_j, DMnd, &b[0], comm); 
	
	if (!b[0]) {
		// if ||V_j|| = 0, pick an arbitrary vector with unit norm that's orthogonal to V_jm1
		rand_min = -1.0, rand_max = 1.0;
		for (i = 0; i < DMnd; i++) {
			V_j[i] = rand_min + (rand_max - rand_min) * (double) rand() / RAND_MAX;
		}
		// orthogonalize V_j and V_jm1
		VectorDotProduct(V_j, V_jm1, DMnd, &a[0], comm);
		for (i = 0; i < DMnd; i++)
			V_j[i] -= a[0] * V_jm1[i];
		// find norm of V_j
		Vector2Norm(V_j, DMnd, &b[0], comm);
	}

	// scale V_j
	vscal = (b[0] == 0.0) ? 1.0 : (1.0 / b[0]);
	for (i = 0; i < DMnd; i++) 
		V_j[i] *= vscal;

	eigmin_pre = *eigmin = 0.0;
	eigmax_pre = *eigmax = 0.0;
	err_eigmin = TOL_min + 1.0;
	err_eigmax = TOL_max + 1.0;
	j = 0;
	while ((err_eigmin > TOL_min || err_eigmax > TOL_max) && j < MAXIT) {
		//t1 = MPI_Wtime();        
		// V_{j+1} = H * V_j
		// Lap_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, V_j, V_jp1, comm);
		mat_vectors_mult(pSPARC, DMnd, A, 1, 0.0, V_j, V_jp1, comm);

		// a[j+1] = <V_j, V_{j+1}>
		VectorDotProduct(V_j, V_jp1, DMnd, &a[j+1], comm);

		for (i = 0; i < DMnd; i++) {
			// V_{j+1} = V_{j+1} - a[j+1] * V_j - b[j] * V_{j-1}
			V_jp1[i] -= (a[j+1] * V_j[i] + b[j] * V_jm1[i]);    
			// update V_{j-1}, i.e., V_{j-1} := V_j
			V_jm1[i] = V_j[i];
		}

		Vector2Norm(V_jp1, DMnd, &b[j+1], comm);
		if (!b[j+1]) {
			break;
		}
		
		vscal = 1.0 / b[j+1];
		// update V_j := V_{j+1} / ||V_{j+1}||
		for (i = 0; i < DMnd; i++)
			V_j[i] = V_jp1[i] * vscal;

		// solve for eigenvalues of the (j+2) x (j+2) tridiagonal matrix T = tridiag(b,a,b)
		for (i = 0; i < j+2; i++) {
			d[i] = a[i];
			e[i] = b[i];
		}
		
		t1 = MPI_Wtime();
		
		if (!LAPACKE_dsterf(j+2, d, e)) {
			*eigmin = d[0];
			*eigmax = d[j+1];
		} else {
			if (rank == 0) { printf("WARNING: Tridiagonal matrix eigensolver (?sterf) failed!\n");}
			break;
		}
		
		t2 = MPI_Wtime();
		
		err_eigmin = fabs(*eigmin - eigmin_pre);
		err_eigmax = fabs(*eigmax - eigmax_pre);

		eigmin_pre = *eigmin;
		eigmax_pre = *eigmax;

		j++;
	}
#ifdef DEBUG
	if (rank == 0) {
		printf("    Lanczos_dense_seq iter %d, eigmin  = %.9f, eigmax = %.9f, err_eigmin = %.3e, err_eigmax = %.3e\n",j,*eigmin, *eigmax,err_eigmin,err_eigmax);
	}
#endif
	
	free(V_j); free(V_jm1); free(V_jp1);
	free(a); free(b); free(d); free(e);
}



























/**
 * @brief  Perform a Rayleigh-Ritz step to find the approx. eigen pair of the 
 *         subspace Hamiltonian Hs in the subspace V.
 *
 *         The CS Rayleight-Ritz process consists of the following steps:
 *           1. Ht = V'*Hs*V, Mt = V'*V.
 *           2. Ht * Qt_s = Mt * Qt_s * Lambda_t.
 *           3. VQ = Y * Qt_s.
 */
void CS_generalized_Rayleigh_Ritz_process(
	SPARC_OBJ *pSPARC, int Ns, int Nt, double *Hs, double *V, int ncol, 
	double *w, double *VQ, int *descV, MPI_Comm colcomm, MPI_Comm rowcomm)
{
#if defined(USE_MKL) || defined(USE_SCALAPACK)
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int descHV[9];
	for (int i = 0; i < 9; i++) descHV[i] = descV[i];
	double *HV = (double *)malloc(Ns * ncol * sizeof(double));
	assert(HV != NULL);
	mat_vectors_mult(pSPARC, Ns, Hs, ncol, 0.0, V, HV, colcomm);

	// generate a (subset) process grid within rowcomm // TODO: move to CS init!
	extern int Csys2blacs_handle(MPI_Comm comm); // TODO: move this to scalapack.h
	int bhandle = Csys2blacs_handle(rowcomm); // create a context out of rowcomm
	int ictxt = bhandle;
	int nproc, gridsizes[2] = {Nt,Nt}, dims[2], ierr;
	MPI_Comm_size(rowcomm, &nproc);
	// for square matrices of size < 20000, it doesn't scale well beyond 64 proc
	SPARC_Dims_create(min(nproc,64), 2, gridsizes, 2, dims, &ierr); // TODO: set min size to 256
	// TODO: swap dim[0] and dim[1] value, since SPARC_Dims_create tends to give larger dim for dim[1] on a tie situation
	#ifdef DEBUG
	if (rank == 0) 
		printf("CS_Rayleigh_Ritz_process: process grid = (%d, %d)\n", dims[0], dims[1]);
	#endif
	Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);

	int ictxt_rowcomm = descV[1];
	// perform Rayleigh step 
	if (ictxt >= 0) {
		int nproc_grid, nprow, npcol, myrow, mycol;
		Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
		nproc_grid = nprow * npcol;

		// define new BLCYC distribution of V // TODO: move to CS init!
		int mb, nb, m_loc, n_loc, llda, ZERO = 0, ONE = 1, info, 
			descV_BLCYC[9], descHV_BLCYC[9], descVQ_BLCYC[9];
		mb = max(1, Ns/nprow); // block size
		nb = max(1, Nt/npcol);
		m_loc = numroc_(&Ns, &mb, &myrow, &ZERO, &nprow);
		n_loc = numroc_(&Nt, &nb, &mycol, &ZERO, &npcol);
		llda = max(1, m_loc);
		descinit_(descV_BLCYC, &Ns, &Nt, &mb, &nb, &ZERO, &ZERO, &ictxt, &llda, &info);
		assert(info == 0);
		for (int i = 0; i < 9; i++) 
			descHV_BLCYC[i] = descVQ_BLCYC[i] = descV_BLCYC[i];
		double *V_BLCYC  = (double *)malloc(m_loc*n_loc*sizeof(double));
		double *HV_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));
		double *VQ_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double)); 

#ifdef PRINT_MAT
		if (rank == 0) {
			printf("local V in rank %d\n", rank);
			for (int i = 0; i < Ns; i++) {
				for (int j = 0; j < min(ncol,12); j++) {
					printf("%15.8e ", V[j*Ns+i]);
				}	
				printf("\n");
			}
		}
#endif
		// convert V from BP to BLCYC
		pdgemr2d_(&Ns, &Nt, V, &ONE, &ONE, descV, V_BLCYC, &ONE, &ONE, 
			descV_BLCYC, &ictxt_rowcomm);
		pdgemr2d_(&Ns, &Nt, HV, &ONE, &ONE, descHV, HV_BLCYC, &ONE, &ONE, 
			descHV_BLCYC, &ictxt_rowcomm);

		// find mass matrix
		// define BLCYC distribution of Mt // TODO: move to CS init!
		int descMt_BLCYC[9];
		mb = nb = max(1, Nt/max(nprow,npcol)); // block size mb must equal nb
		m_loc = numroc_(&Nt, &mb, &myrow, &ZERO, &nprow);
		n_loc = numroc_(&Nt, &nb, &mycol, &ZERO, &npcol);
		llda = max(1, m_loc);
		descinit_(descMt_BLCYC, &Nt, &Nt, &mb, &nb, &ZERO, &ZERO, &ictxt, 
			&llda, &info);
		assert(info == 0);

		double *Mt_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));
		assert(Mt_BLCYC != NULL);
		
		// Mt = V' * V
		double alpha = 1.0, beta = 0.0;
		pdsyrk_("U", "T", &Nt, &Ns, &alpha, V_BLCYC, &ONE, &ONE, descV_BLCYC,
			&beta, Mt_BLCYC, &ONE, &ONE, descMt_BLCYC);

#ifdef PRINT_MAT
		if (rank == 0) {
			printf("local Mt (pdsyrk_) in rank %d\n", rank);
			for (int i = 0; i < min(m_loc,10); i++) {
				for (int j = 0; j < min(n_loc,10); j++) {
					printf("%15.8e ", Mt_BLCYC[j*m_loc+i]);
				}	
				printf("\n");
			}
		}
#endif

// 		pdgemm_("T", "N", &Nt, &Nt, &Ns, &alpha, V_BLCYC, &ONE, &ONE, descV_BLCYC,
// 				V_BLCYC, &ONE, &ONE, descV_BLCYC, &beta, Mt_BLCYC, &ONE, &ONE, descMt_BLCYC);
// #ifdef PRINT_MAT
// 		printf("local Mt (pdgemm_) in rank %d\n", rank);
// 		for (int i = 0; i < m_loc; i++) {
// 			for (int j = 0; j < n_loc; j++) {
// 				printf("%15.8e ", Mt_BLCYC[j*m_loc+i]);
// 			}	
// 			printf("\n");
// 		}
// #endif

		// find projected partial subspace Hamiltonian
		// assume Ht has the same distribution as Mt
		int descHt_BLCYC[9];
		for (int i = 0; i < 9; i++) descHt_BLCYC[i] = descMt_BLCYC[i];
		double *Ht_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));

		alpha = 0.5; beta = 0.0;
		extern void pdsyr2k_();
		pdsyr2k_("U", "T", &Nt, &Ns, &alpha, V_BLCYC, &ONE, &ONE, descV_BLCYC,
			HV_BLCYC, &ONE, &ONE, descHV_BLCYC,
			&beta, Ht_BLCYC, &ONE, &ONE, descHt_BLCYC);
		
#ifdef PRINT_MAT
		if (rank == 0) {
			printf("local Ht (pdsyr2k_) in rank %d\n", rank);
			for (int i = 0; i < min(m_loc,10); i++) {
				for (int j = 0; j < min(n_loc,10); j++) {
					printf("%15.8e ", Ht_BLCYC[j*m_loc+i]);
				}	
				printf("\n");
			}
		}
#endif
		
		// alpha = 1.0; beta = 0.0;
		// pdgemm_("T", "N", &Nt, &Nt, &Ns, &alpha, V_BLCYC, &ONE, &ONE, descV_BLCYC,
		// 		HV_BLCYC, &ONE, &ONE, descHV_BLCYC, &beta, Ht_BLCYC, &ONE, &ONE, descHt_BLCYC);
		// printf("local Ht (pdgemm_) in rank %d\n", rank);
		// for (int i = 0; i < m_loc; i++) {
		// 	for (int j = 0; j < n_loc; j++) {
		// 		printf("%15.8e ", Ht_BLCYC[j*m_loc+i]);
		// 	}	
		// 	printf("\n");
		// }

		// solve the complementary subspace eigenproblem
		// assume Qt has the same distribution as Mt
		int descQt_s_BLCYC[9];
		for (int i = 0; i < 9; i++) descQt_s_BLCYC[i] = descMt_BLCYC[i];
		// double *w = (double *)malloc(Nt*sizeof(double));
		double *Qt_s_BLCYC = (double *)malloc(m_loc*n_loc*sizeof(double));
		int il = 1, iu = 1, M, NZ;
		double vl = 0.0, vu = 0.0, orfac = 0.001, abstol;
		orfac = pSPARC->eig_paral_orfac;
        #ifdef DEBUG
        if(!rank) printf("rank = %d, orfac = %.3e\n", rank, orfac);
        #endif
		
		int isSerial = Nt < pSPARC->eig_serial_maxns ? 1 : 0; // consider ictxt size too?
		
		isSerial = 0; // TODO: remove this after check
		if (isSerial) {
			// TODO: implement sequential calculation of dense full eigen solve
		} else {
			int *ifail = (int *)malloc(Nt * sizeof(int));
			// this setting yields the most orthogonal eigenvectors
			abstol = pdlamch_(&ictxt, "U");
			automem_pdsygvx_ ( 
				&ONE, "V", "A", "U", &Nt, Ht_BLCYC, &ONE, &ONE, descHt_BLCYC, 
				Mt_BLCYC, &ONE, &ONE, descMt_BLCYC, &vl, &vu, &il, &iu, &abstol, &M, &NZ,
				w, &orfac, Qt_s_BLCYC, &ONE, &ONE, descQt_s_BLCYC, ifail, &info);
			if (info) printf("info = %d, ifail[0] = %d\n", info, ifail[0]);
			free(ifail);
		}

		if (!rank) {

#ifdef PRINT_MAT
			printf("w = \n");
			for (int i = 0; i < Nt; i++) printf(RED "%.10f \n" RESET, w[i]);
			printf("local Z (CS eigenvectors) in rank %d\n", rank);
			for (int i = 0; i < min(m_loc,12); i++) {
				for (int j = 0; j < min(n_loc,12); j++) {
					printf("%15.8e ", Qt_s_BLCYC[j*m_loc+i]);
				}	
				printf("\n");
			}
#endif
		}

		// complementary subspace rotation V = V * Qt_s
		alpha = 1.0; beta = 0.0;
		pdgemm_("N", "N", &Ns, &Nt, &Nt, &alpha, V_BLCYC, &ONE, &ONE, descV_BLCYC,
			Qt_s_BLCYC, &ONE, &ONE, descQt_s_BLCYC, &beta, VQ_BLCYC, &ONE, &ONE, descVQ_BLCYC);

		// convert VQ from BLCYC to BP
		pdgemr2d_(&Ns, &Nt, VQ_BLCYC, &ONE, &ONE, descVQ_BLCYC, VQ, &ONE, &ONE, descV, &ictxt_rowcomm);

#ifdef PRINT_MAT
		if (rank == 0) {
			printf("local VQ in rank %d\n", rank);
			for (int i = 0; i < Ns; i++) {
				for (int j = 0; j < min(ncol,12); j++) {
					printf("%14.10f ", VQ[j*Ns+i]);
				}	
				printf("\n");
			}
		}
#endif
		free(Qt_s_BLCYC); 
		free(Mt_BLCYC);
		free(Ht_BLCYC);
		free(V_BLCYC);
		free(HV_BLCYC);
		free(VQ_BLCYC);
		Cblacs_gridexit(ictxt);
	} else {
		int ONE = 1, descV_BLCYC[9], descHV_BLCYC[9], descVQ_BLCYC[9];
		double *V_BLCYC, *HV_BLCYC, *VQ_BLCYC;
		for (int i = 0; i < 9; i++) 
			descV_BLCYC[i] = descHV_BLCYC[i] = -1;
		pdgemr2d_(&Ns, &Nt, V, &ONE, &ONE, descV, V_BLCYC, &ONE, &ONE, descV_BLCYC, &ictxt_rowcomm);
		pdgemr2d_(&Ns, &Nt, HV, &ONE, &ONE, descHV, HV_BLCYC, &ONE, &ONE, descHV_BLCYC, &ictxt_rowcomm);
		pdgemr2d_(&Ns, &Nt, VQ_BLCYC, &ONE, &ONE, descVQ_BLCYC, VQ, &ONE, &ONE, descV, &ictxt_rowcomm);
	}

	MPI_Bcast(w, Nt, MPI_DOUBLE, 0, rowcomm);

	free(HV);
#endif // (USE_MKL or USE_SCALAPACK)
}
