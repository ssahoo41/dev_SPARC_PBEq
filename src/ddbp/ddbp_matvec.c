/**
 * @file    ddbp_matvec.c
 * @brief   This file contains the functions for the Discrete Discontinuous
 *          Basis Projection (DDBP) Hamiltonian matrix-vector routines.
 *          routines.
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
#include "nlocVecRoutines.h"
#include "eigenSolver.h"
#include "finalization.h"
#include "isddft.h"
#include "tools.h"
#include "sq3.h"
#include "cs.h"
#include "ddbp.h"


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define TEMP_TOL 1e-14

double t_haloX = 0.0, t_densmat = 0.0, t_nloc = 0.0, t_malloc = 0.0;

void adaptive_dsymm(
    const CBLAS_ORDER order, const CBLAS_SIDE side,
    const CBLAS_UPLO uplo, const int m, const int n, const double alpha,
    const double*a, const int lda, const double *b, const int ldb,
    const double beta, double *c, const int ldc)
{
    if ((n == 1 || m*(n+m) < 4096) &&
        (side == CblasLeft))
    {
        for (int i = 0; i < n; i++)
            cblas_dsymv(order, uplo, m, alpha, a, lda,
                b+i*ldb, 1, beta, c+i*ldc, 1);
    } else {
        cblas_dsymm(order, side, uplo, m, n,
            alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

void adaptive_dgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const int m, const int n, const int k,
    const double alpha, const double *a, const int lda, const double *b,
    const int ldb, const double beta, double *c, const int ldc)
{
    if ((n == 1 || m*(n+k) < 4096) &&
        ((Layout == CblasColMajor && transb == CblasNoTrans) ||
         (Layout == CblasRowMajor && transb == CblasTrans  ))
    ) {
        for (int i = 0; i < n; i++)
            cblas_dgemv(Layout, transa, m, k,
                alpha, a, lda, b+i*ldb, 1, beta, c+i*ldc, 1);
    } else {
        cblas_dgemm(Layout, transa, transb, m, n, k,
            alpha, a, lda, b, ldb, beta, c, ldc);
    }
}


/**
 * @brief Perform the matrix-vector multiplication of the diagonal
 *        block of the DDBP Hamiltonian. This routine does only the
 *        local part of DDBP Hamiltonian.
 *
 *        Hxk = alpha * (H_DDBP(k,k) + c*I) * xk + beta * Hxk.
 *
 * @param alpha Scalar alpha.
 * @param H_DDBP_Ek Element row blocks of the DDBP Hamiltonian
 *                  corresponding to an element Ek.
 * @param c Scalar c.
 * @param xk Array that falls in element Ek to be multiplied.
 * @param beta Scalar beta.
 * @param Hxk Result array (output).
 * @param ncol Number of columns in the array xk.
 */
void H_DDBP_diag_block_matvec(
    const double alpha, const DDBP_HAMILT_ERBLKS *H_DDBP_Ek,
    const double c, const double *xk, const double beta, double *Hxk,
    int ncol)
{
    int m = H_DDBP_Ek->blksz;
    const double *h_kk = H_DDBP_Ek->h_kj[6];

    adaptive_dsymm(CblasColMajor, CblasLeft, CblasUpper, m, ncol, alpha,
            h_kk, m, xk, m, beta, Hxk, m);

#ifdef DEBUG_DOUBLE_CHECK
    double *Hxk_ref = (double *)malloc(m * ncol * sizeof(double));
    assert(Hxk_ref != NULL);
    // *Method 1:
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, ncol, m,
        alpha, h_kk, m, xk, m, beta, Hxk_ref, m);
    // *Method 2:
    // adaptive_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, ncol, m,
    //     alpha, h_kk, m, xk, m, beta, Hxk_ref, m);
    // *Method 3:
    // adaptive_dsymm(CblasColMajor, CblasLeft, CblasUpper, m, ncol, alpha,
    //         h_kk, m, xk, m, beta, Hxk_ref, m);
    const int nele = m * ncol;
    double err = 0.0;
    for (int i = 0; i < nele; i++) {
        err = max(err, fabs(Hxk_ref[i] - Hxk[i]));
    }
    free(Hxk_ref);
    assert(err < TEMP_TOL);
#endif

    // add shift alpha * c * X
    const double scal = alpha * c;
	if (fabs(scal) > TEMP_TOL) {
		const int nele = m * ncol;
		for (int i = 0; i < nele; i++) {
			Hxk[i] += scal * xk[i];
		}
	}
}


/**
 * @brief Perform the matrix-vector multiplication of the off-diagonal
 *        blocks of the DDBP Hamiltonian.
 *
 * @param Edims Global number of elements in each direction.
 * @param E_k_coords Element coordinates.
 * @param alpha Scalar alpha.
 * @param H_DDBP_Ek Element row blocks of the DDBP Hamiltonian
 *                  corresponding to an element Ek.
 * @param c Scalar c.
 * @param recvbuf Received buffer from neighbors of Ek.
 * @param beta Scalar beta.
 * @param Hxk Result array (output).
 * @param ncol Number of columns in the array xk.
 */
void H_DDBP_offdiag_block_matvec(
    const int Edims[3], const int E_k_coords[3], const double alpha,
    const DDBP_HAMILT_ERBLKS *H_DDBP_Ek, const double c,
    const double *recvbuf, const int *rdispls, const double beta,
    double *Hxk, int ncol)
{
    int m = H_DDBP_Ek->blksz;
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
        // which direction
        int dim = nbr_i / 2;
        // which face of neighbor, 0:left, 1:right
        int face = nbr_i % 2;
        // check special cases
        if ( (Edims[dim] == 1) || ((Edims[dim] == 2) && (E_k_coords[dim] == face)) )
            continue;
        const double *h_kj = H_DDBP_Ek->h_kj[nbr_i];
        const double *xj = recvbuf + rdispls[nbr_i];
        adaptive_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, ncol, m,
            alpha, h_kj, m, xj, m, beta, Hxk, m);
    }
}


void scale_IP_by_Gamma_Jl(
    const int Ntypes, const int *nAtomv, const int *localPsd, PSD_OBJ *psd,
    double *IP, const int ncol)
{
    // go over all atoms and multiply gamma_Jl to the inner product
    int count = 0;
    for (int ityp = 0; ityp < Ntypes; ityp++) {
        int lloc = localPsd[ityp];
        int lmax = psd[ityp].lmax;
        for (int iat = 0; iat < nAtomv[ityp]; iat++) {
            for (int n = 0; n < ncol; n++) {
                int ldispl = 0;
                for (int l = 0; l <= lmax; l++) {
                    // skip the local l
                    if (l == lloc) {
                        ldispl += psd[ityp].ppl[l];
                        continue;
                    }
                    for (int np = 0; np < psd[ityp].ppl[l]; np++) {
                        for (int m = -l; m <= l; m++) {
                            IP[count++] *= psd[ityp].Gamma[ldispl+np];
                        }
                    }
                    ldispl += psd[ityp].ppl[l];
                }
            }
        }
    }
}


void Vnl_DDBP_vectors_mult_kernel(
    const double alpha, DDBP_HAMILTONIAN *H_DDBP, const DDBP_ARRAY *X,
    const double beta, DDBP_ARRAY *HX, MPI_Comm comm)
{
    int ncol = X->ncol;
    int nelem = X->nelem;
    if (nelem < 1) return;

    // get params for Vnl_DDBP from the first element
    int Ntypes = H_DDBP->Ntypes;
    int n_atom = H_DDBP->n_atom;
    int *IP_displ = H_DDBP->IP_displ;
    int nJlm = IP_displ[n_atom];
    double *IP = (double *)calloc(nJlm*ncol, sizeof(double));
    assert(IP != NULL);

    // compute inner products (IP): alpha * <Chi_Jlm, x_n> for all J,l,m,n
    for (int k = 0; k < nelem; k++) {
        // DDBP_ELEM *E_k = &X->elem_list[k];
        DDBP_HAMILT_ERBLKS *H_DDBP_Ek = H_DDBP->H_DDBP_Ek_list[k];
        DDBP_VNL *Vnl_DDBP = &H_DDBP_Ek->Vnl_DDBP;
        double *xk = X->array[k];
        int ldx = X->nrows[k];
        ATOM_NLOC_INFLUENCE_OBJ *AtmNloc = Vnl_DDBP->AtmNloc;
        NLOC_PROJ_OBJ *nlocProj = Vnl_DDBP->nlocProj;
        double scale = alpha * Vnl_DDBP->dV;
        // compute inner products and add to IP, IP_J += scale * <Chi_Jlm, xk_n>
        for (int ityp = 0; ityp < Ntypes; ityp++) {
            int nproj = nlocProj[ityp].nproj;
            if (!nproj) continue; // this is typical for hydrogen
            int n_atom_type = AtmNloc[ityp].n_atom;
            for (int iat = 0; iat < n_atom_type; iat++) {
                int atom_index = AtmNloc[ityp].atom_index[iat];
                int ndc = AtmNloc[ityp].ndc[iat];
                int *grid_pos_J = AtmNloc[ityp].grid_pos[iat];
                double *Chixk = IP+IP_displ[atom_index]*ncol;
                Chi_J_X_mult(ndc, nproj, grid_pos_J, scale, nlocProj[ityp].Chi[iat],
                    xk, ldx, 1.0, Chixk, nproj, ncol);
            }
        }
    }

    // if there are element parallelization, we need to sum over elements
    int commsize;
    MPI_Comm_size(comm, &commsize);
    if (commsize > 1) {
        MPI_Allreduce(MPI_IN_PLACE, IP, nJlm*ncol, MPI_DOUBLE, MPI_SUM, comm);
    }

    // HX = beta * HX, if beta != 1.0
    if (fabs(beta - 1.0) > 1e-12) {
        for (int k = 0; k < nelem; k++) {
            int nrow = HX->nrows[k];
            int ncol = HX->ncol;
            int len = nrow * ncol;
            double *Hxk = HX->array[k];
            for (int i = 0; i < len; i++) Hxk[i] *= beta;
        }
    }

    // go over all atoms and multiply gamma_Jl to the inner product
    int *nAtomv = H_DDBP->nAtomv;
    int *localPsd = H_DDBP->localPsd;
    PSD_OBJ *psd = H_DDBP->psd;
    scale_IP_by_Gamma_Jl(Ntypes, nAtomv, localPsd, psd, IP, ncol);

    // beta * HX += sum_Jlm gamma_Jl |Chi_Jlm> IP_Jlmn
    for (int k = 0; k < nelem; k++) {
        DDBP_HAMILT_ERBLKS *H_DDBP_Ek = H_DDBP->H_DDBP_Ek_list[k];
        DDBP_VNL *Vnl_DDBP = &H_DDBP_Ek->Vnl_DDBP;
        double *Hxk = HX->array[k];
        ATOM_NLOC_INFLUENCE_OBJ *AtmNloc = Vnl_DDBP->AtmNloc;
        NLOC_PROJ_OBJ *nlocProj = Vnl_DDBP->nlocProj;
        for (int ityp = 0; ityp < Ntypes; ityp++) {
            int nproj = nlocProj[ityp].nproj;
            if (!nproj) continue; // this is typical for hydrogen
            int n_atom_type = AtmNloc[ityp].n_atom;
            for (int iat = 0; iat < n_atom_type; iat++) {
                int ndc = AtmNloc[ityp].ndc[iat];
                int atom_index = AtmNloc[ityp].atom_index[iat];
                int displ = IP_displ[atom_index];
                adaptive_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    ndc, ncol, nproj, 1.0, nlocProj[ityp].Chi[iat], ndc,
                    IP+displ*ncol, nproj, 1.0, Hxk, ndc);
            }
        }
    }
    free(IP);
}



/**
 * @brief DDBP Hamiltonian matrix-vector multiplication kernel.
 *
 * @details This routine finds
 *              HX = alpha * (H_DDBP + c*I) X + beta * HX,
 *          where H_DDBP is the DDBP Hamiltonian, alpha, beta, c
 *          are constants.
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
void DDBP_Hamiltonian_vectors_mult_kernel(
    const double alpha, DDBP_HAMILTONIAN *H_DDBP,
    const double c, const DDBP_ARRAY *X, const double beta, DDBP_ARRAY *HX,
    MPI_Comm comm)
{
    double t1, t2;

    int ncol = X->ncol;
    int nelem = X->nelem;
    int Edims[3];
    Edims[0] = X->Edims[0];
    Edims[1] = X->Edims[1];
    Edims[2] = X->Edims[2];

    // initialize the halo exchange
    t1 = MPI_Wtime();
    double **recvbufs = malloc(nelem * sizeof(*recvbufs));
    assert(recvbufs != NULL);
    t2 = MPI_Wtime();
    t_malloc += t2 - t1;
    for (int k = 0; k < nelem; k++) {
        haloX_t *haloX = &X->haloX_info[k];
        // *send buf is just the whole array, same for all neighbors
        // int sendcount_tot = haloX->sendcounts[0]; // pick anyone
        // *recv buf is different from neighbors
        int recvcount_tot = 0;
        for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
            recvcount_tot += haloX->recvcounts[nbr_i];
        }
        t1 = MPI_Wtime();
        double *recvbuf = calloc(recvcount_tot, sizeof(double));
        assert(recvbuf != NULL);
        t2 = MPI_Wtime();
        t_malloc += t2 - t1;
        recvbufs[k] = recvbuf; // save the address for later to deallocate
        const double *sendbuf = X->array[k];
        t1 = MPI_Wtime();
        DDBP_element_Ineighbor_alltoallv(haloX, sendbuf, recvbuf, comm);
        t2 = MPI_Wtime();
        t_haloX += t2 - t1;
    }

    // while the halo exchange is ongoing, perform the mat-vec between
    // the diagonal blocks (loc and nloc) and the vector first
    // TODO: implement the code here
    for (int k = 0; k < nelem; k++) {
        // DDBP_ELEM *E_k = &X->elem_list[k];
        // haloX_t *haloX = &X->haloX_info[k];
        DDBP_HAMILT_ERBLKS *H_DDBP_Ek = H_DDBP->H_DDBP_Ek_list[k];
        // ? If we combine the nonlocal projectors to a single symmetric
        // ? matrix (of the same size as the local diagonal block), this
        // ? way the nonlocal part doesn't have to be separated from the
        // ? local part! H_kk = h_loc_kk + Vnl (some element doesn't have
        // ? Vnl for an atom)
        // ? The question is whether the result is correct since projectors
        // ? are distributed among different elements, and when do we reduce
        // ? the results
        t1 = MPI_Wtime();
        H_DDBP_diag_block_matvec(alpha, H_DDBP_Ek, c, X->array[k], beta,
            HX->array[k], ncol);
        t2 = MPI_Wtime();
        t_densmat += t2 - t1;
    }

    // nonlocal part alpha * Vnl * X
    t1 = MPI_Wtime();
    Vnl_DDBP_vectors_mult_kernel(alpha, H_DDBP, X, 1.0, HX, comm);
    t2 = MPI_Wtime();
    t_nloc += t2 - t1;

    for (int k = 0; k < nelem; k++) {
        DDBP_ELEM *E_k = &X->elem_list[k];
        haloX_t *haloX = &X->haloX_info[k];
        // wait for the halo exchange to be completed here
        t1 = MPI_Wtime();
        // wait for the element halo exchange of Hv data to be completed here
        MPI_Waitall(2*haloX->n_neighbors, haloX->requests, MPI_STATUS_IGNORE);
        t2 = MPI_Wtime();
        t_haloX += t2 - t1;

        // do the off-diagonal part (from local)
        // TODO: implement the code here
        DDBP_HAMILT_ERBLKS *H_DDBP_Ek = H_DDBP->H_DDBP_Ek_list[k];
        double *recvbuf = recvbufs[k];
        t1 = MPI_Wtime();
        H_DDBP_offdiag_block_matvec(Edims, E_k->coords, alpha, H_DDBP_Ek, c,
            recvbuf, haloX->rdispls, 1.0, HX->array[k], ncol);
        t2 = MPI_Wtime();
        t_densmat += t2 - t1;
    }

    // deallocate the buffers
    for (int k = 0; k < nelem; k++) {
        free(recvbufs[k]);
    }
    free(recvbufs);
}



/**
 * @brief Set up an aux DDBP Array that points to selected columns of a
 *        DDBP Array.
 *
 * @param X Original DDBP Array.
 * @param X_aux Auxiliary DDBP Array that points to the selected columns.
 * @param is Start index of the array.
 * @param ncol Number of columns the auxiliary array refers to.
 */
void setup_aux_DDBP_Array(
    const DDBP_ARRAY *X, DDBP_ARRAY *X_aux, const int is, const int ncol)
{
    int nelem = X->nelem;
    *X_aux = *X; // shallow copy the values of X into X_aux

    X_aux->haloX_info = malloc(nelem * sizeof(*X_aux->haloX_info));
    X_aux->array = malloc(nelem * sizeof(*X_aux->array));
    assert(X_aux->haloX_info != NULL);
    assert(X_aux->array != NULL);

    X_aux->ncol = ncol;
    for (int k = 0; k < nelem; k++) {
        X_aux->array[k] = X->array[k] + is * X->nrows[k];
        // modify haloX info for X_aux
        const haloX_t *haloX = &X->haloX_info[k];
        haloX_t *haloX_aux = &X_aux->haloX_info[k];
        haloX_aux->n_neighbors = haloX->n_neighbors;
        haloX_aux->sendtype = haloX->sendtype;
        haloX_aux->recvtype = haloX->recvtype;
        for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
            haloX_aux->neighbor_indices[nbr_i] = haloX->neighbor_indices[nbr_i];
            haloX_aux->neighbor_ranks[nbr_i] = haloX->neighbor_ranks[nbr_i];
            haloX_aux->stags[nbr_i] = haloX->stags[nbr_i];
            haloX_aux->rtags[nbr_i] = haloX->rtags[nbr_i];
            haloX_aux->sendcounts[nbr_i] = X_aux->ncol * X_aux->nrows[k];
            haloX_aux->recvcounts[nbr_i] = X_aux->ncol * X_aux->nrows[k];
        }
        haloX_aux->sdispls[0] = 0;
        haloX_aux->rdispls[0] = 0;
        for (int nbr_i = 0; nbr_i < 5; nbr_i++) {
            haloX_aux->sdispls[nbr_i+1] = 0;
            haloX_aux->rdispls[nbr_i+1] = haloX_aux->rdispls[nbr_i] + haloX_aux->recvcounts[nbr_i];
        }
    }
}


/**
 * @brief Free the memory allocated for the auxiliary DDBP Array.
 *
 * @param X_aux Auxiliary DDBP Array to be freed.
 */
void free_aux_DDBP_Array(DDBP_ARRAY *X_aux)
{
    free(X_aux->haloX_info);
    free(X_aux->array);
}



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
void DDBP_Hamiltonian_vectors_mult_kernel_selectcol(
    const double alpha, DDBP_HAMILTONIAN *H_DDBP,
    const double c, const DDBP_ARRAY *X, const double beta, DDBP_ARRAY *HX,
    MPI_Comm comm, const int is, const int ncol)
{
    if (ncol == 0) return;

    assert((is >= 0) && (is + ncol <= X->ncol) && (ncol >= 0));
    DDBP_ARRAY X_aux;
    DDBP_ARRAY HX_aux;
    setup_aux_DDBP_Array(X, &X_aux, is, ncol);
    setup_aux_DDBP_Array(HX, &HX_aux, is, ncol);
    DDBP_Hamiltonian_vectors_mult_kernel(
        alpha, H_DDBP, c, &X_aux, beta, &HX_aux, comm);
    free_aux_DDBP_Array(&X_aux);
    free_aux_DDBP_Array(&HX_aux);
}



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
    MPI_Comm comm)
{
    int ncol = X->ncol;
    if (ncol == 0) return;
    // * do blksz columns at a time, instead of all columns altogether
    // hard-coded, this seems a good choice for nALB = 80 - 500
    int blksz = 30;
    int nblks = max(ncol / blksz, 1);
    for (int nb = 0; nb < nblks; nb++) {
        int ncol_nb = block_decompose(ncol, nblks, nb);
        int ns_nb = block_decompose_nstart(ncol, nblks, nb);
        DDBP_Hamiltonian_vectors_mult_kernel_selectcol(
            alpha, H_DDBP, c, X, beta, HX, comm, ns_nb, ncol_nb);
    }

// #define DEBUG_DOUBLE_CHECK
#ifdef DEBUG_DOUBLE_CHECK
    DDBP_ARRAY HX_ref;
    create_DDBP_Array(
        X->BCs, X->Edims, X->nelem, X->elem_list, ncol, &HX_ref);
    DDBP_Hamiltonian_vectors_mult_kernel(
        alpha, H_DDBP, c, X, beta, &HX_ref, comm);
    int info = double_check_DDBP_arrays(HX, &HX_ref, 0, ncol);
    delete_DDBP_Array(&HX_ref);
    assert(info == 0);
#endif
}


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
    MPI_Comm comm, const int is, const int ncol)
{
    if (ncol == 0) return;
    assert((is >= 0) && (is + ncol <= X->ncol) && (ncol >= 0));
    DDBP_ARRAY X_aux;
    DDBP_ARRAY HX_aux;
    setup_aux_DDBP_Array(X, &X_aux, is, ncol);
    setup_aux_DDBP_Array(HX, &HX_aux, is, ncol);
    DDBP_Hamiltonian_vectors_mult(
        alpha, H_DDBP, c, &X_aux, beta, &HX_aux, comm);
    free_aux_DDBP_Array(&X_aux);
    free_aux_DDBP_Array(&HX_aux);

#define DEBUG_DOUBLE_CHECK
#ifdef DEBUG_DOUBLE_CHECK
    DDBP_ARRAY HX_ref;
    create_DDBP_Array(X->BCs, X->Edims, X->nelem, X->elem_list, X->ncol, &HX_ref);
    DDBP_Hamiltonian_vectors_mult_kernel_selectcol(
        alpha, H_DDBP, c, X, beta, &HX_ref, comm, is, ncol);
    int info = double_check_DDBP_arrays(HX, &HX_ref, is, ncol);
    delete_DDBP_Array(&HX_ref);
    assert(info == 0);
#endif
}

