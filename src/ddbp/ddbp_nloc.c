/**
 * @file    ddbp_nloc.c
 * @brief   This file contains the functions for the nonlocal routines
 *          for the Discrete Discontinuous Basis Projection (DDBP)
 *          method.
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

#include "nlocVecRoutines.h"
#include "isddft.h"
#include "tools.h"
#include "ddbp.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define TEMP_TOL 1e-14

static double t_allgatherv, t_trans, t_ChiX;

/**
 * @brief Allocate memory for nloc projectors object for DDBP Hamiltonian.
 * 
 * @param nALB Total number of adaptive local basis functions in this element.
 * @param Ntypes Number of atom types.
 * @param nlocProj The global nonlocal projector restricted in the element.
 * @param AtmNloc The influencing atoms for this element.
 * @param nlocProj_DDBP The nonlocal projector object for DDBP (output).
 * @param AtmNloc_DDBP The influencing atoms (for DDBP Hamiltonian) for this element.
 * @param proc_active Flag to indicate if the process is active (0 - inactive).
 */
void init_nlocProj_DDBP(int nALB, int Ntypes,
    const NLOC_PROJ_OBJ *nlocProj, const ATOM_NLOC_INFLUENCE_OBJ *AtmNloc,
    NLOC_PROJ_OBJ **nlocProj_DDBP, ATOM_NLOC_INFLUENCE_OBJ **AtmNloc_DDBP,
    int proc_active)
{
// TODO: store DDBP projectors Chi in a contiguous array for all atoms!
// * An advantage of DDBP nloc projectors is that they have the same size.
// * So it's possible to concatenate them together and utilize BLAS 3 to
// * a deeper extend.
    if (proc_active == 0) return;

    // set up AtmNloc_DDBP
    (*AtmNloc_DDBP) = malloc(Ntypes * sizeof(ATOM_NLOC_INFLUENCE_OBJ));
    assert((*AtmNloc_DDBP) != NULL);
    for (int ityp = 0; ityp < Ntypes; ityp++) {
        int n_atom_type = AtmNloc[ityp].n_atom;
        (*AtmNloc_DDBP)[ityp].n_atom = n_atom_type;
        (*AtmNloc_DDBP)[ityp].coords = NULL;
        (*AtmNloc_DDBP)[ityp].atom_index = (int *)malloc(n_atom_type * sizeof(int));
        assert((*AtmNloc_DDBP)[ityp].atom_index != NULL);
        (*AtmNloc_DDBP)[ityp].xs  = NULL;
        (*AtmNloc_DDBP)[ityp].ys  = NULL;
        (*AtmNloc_DDBP)[ityp].zs  = NULL;
        (*AtmNloc_DDBP)[ityp].xe  = NULL;
        (*AtmNloc_DDBP)[ityp].ye  = NULL;
        (*AtmNloc_DDBP)[ityp].ze  = NULL;
        (*AtmNloc_DDBP)[ityp].ndc = (int *)malloc(n_atom_type * sizeof(int));
        assert((*AtmNloc_DDBP)[ityp].ndc != NULL);
        (*AtmNloc_DDBP)[ityp].grid_pos = (int **)malloc(n_atom_type * sizeof(int*));
        assert((*AtmNloc_DDBP)[ityp].grid_pos != NULL);
        for (int iat = 0; iat < n_atom_type; iat++) {
            (*AtmNloc_DDBP)[ityp].ndc[iat] = nALB;
            (*AtmNloc_DDBP)[ityp].grid_pos[iat] = NULL;
            (*AtmNloc_DDBP)[ityp].atom_index[iat] = AtmNloc[ityp].atom_index[iat];
        }
    }

    // set up nlocProj_DDBP
    (*nlocProj_DDBP) = malloc(Ntypes * sizeof(NLOC_PROJ_OBJ));
    assert((*nlocProj_DDBP) != NULL);
    for (int ityp = 0; ityp < Ntypes; ityp++) {
        int n_atom_type = AtmNloc[ityp].n_atom;
        (*nlocProj_DDBP)[ityp].Chi = (double **)malloc(n_atom_type * sizeof(double *));
        assert((*nlocProj_DDBP)[ityp].Chi != NULL);
        int nproj = nlocProj[ityp].nproj;
        (*nlocProj_DDBP)[ityp].nproj = nproj;
        if (!nproj) {
            // easier to keep track by forcing it to NULL
            for (int iat = 0; iat < n_atom_type; iat++) {
                (*nlocProj_DDBP)[ityp].Chi[iat] = NULL;
            }
            continue;
        }
        for (int iat = 0; iat < n_atom_type; iat++) {
            int ndc = nALB; // ! same for all atom types
            (*nlocProj_DDBP)[ityp].Chi[iat] = (double *)malloc(ndc * nproj * sizeof(double));
            assert((*nlocProj_DDBP)[ityp].Chi[iat] != NULL);
        }
    }
}


/**
 * @brief Init Vnl_DDBP object. Point the pointers regarding atom information
 *        to the corresponding arrays. Note that no new memory is allocated.
 * 
 * @param Vnl_DDBP Vnl_DDBP object to be initialized.
 * @param Ntypes Global number of atom types.
 * @param n_atom Global total number of atoms.
 * @param nAtomv Number of atoms of each type.
 * @param localPsd Local pseudopotential l (lloc) values.
 * @param IP_displ Inner Product displacements for all atoms.
 * @param psd Pseudopotential object.
 * @param dV Integration weights.
 */
void init_Vnl_DDBP(
    DDBP_VNL *Vnl_DDBP, int Ntypes, int n_atom, int *nAtomv,
    int *localPsd, int *IP_displ, PSD_OBJ *psd, double dV)
{
    Vnl_DDBP->Ntypes = Ntypes;
    Vnl_DDBP->dV = dV;
    // ! The following variables are no longer available
    // Vnl_DDBP->n_atom = n_atom;
    // Vnl_DDBP->nAtomv = nAtomv;
    // Vnl_DDBP->localPsd = localPsd;
    // Vnl_DDBP->IP_displ = IP_displ;
    // Vnl_DDBP->psd = psd;
}


/**
 * @brief  Calculate product between the nolocal projectors for an
 *         influencing atom J and a bunch of vectors, i.e.,
 *              ChiX = alpha * Chi_J^T * X + beta * ChiX,
 *         where alpha, beta are scaling factors.
 *
 *         Note this only does the multiplication locally, if Chi_J
 *         is distributed, the results needs to be summed later.
 *
 * @param ndc        Number of non-zeros in the rc domain.
 * @param nproj      Number of projectors for the given atom.
 * @param grid_pos_J Nonzero grid position indices. If Chi_J is dense
 *                   in the process domain, set this to NULL.
 * @param alpha      Scaling factor.
 * @param Chi_J      Nonlocal projectors for Jth atom.
 * @param X          Vector to be multiplied.
 * @param ldX        Leading dimension of X.
 * @param beta       Scaling factor.
 * @param ChiX       Result vectors.
 * @param ldChiX     Leading dimension of ChiX.
 * @param ncol       Number of columns of X/ChiX.
 */
void Chi_J_X_mult(
    int ndc, int nproj, const int *grid_pos_J,
    double alpha, double *Chi_J, double *X,
    int ldX, double beta, double *ChiX, int ldChiX, int ncol)
{
    if (!nproj) return; // this is typical for hydrogen

    double *x_rc;
    // check if Chi_J is sparse in the distributed domain
    int isSparse = (int) (grid_pos_J != NULL);
    if (isSparse) {
        // if Chi is sparse, we extract the corresponding non-zeros of X
        x_rc = (double *)malloc(ndc * ncol * sizeof(double));
        assert(x_rc != NULL);
        for (int n = 0; n < ncol; n++) {
            for (int i = 0; i < ndc; i++) {
                // x_rc[n*ndc+i] = X[n*ldX + AtmNloc[ityp].grid_pos[iat][i]];
                x_rc[n*ndc+i] = X[n*ldX + grid_pos_J[i]];
            }
        }
    } else {
        x_rc = X;
    }

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nproj, ncol,
        ndc, alpha, Chi_J, ndc, x_rc, ndc, beta, ChiX, ldChiX);

    if (isSparse) free(x_rc);
}



/**
 * @brief  Calculate product between the nolocal projectors for an
 *         influencing atom J and a bunch of vectors, i.e.,
 *                       alpha * vk^T * Chi_Jlm,
 *         where alpha is a scaling factor.
 *
 *         Note this only does the multiplication locally, if Chi_Jlm is
 *         distributed, the results needs to be summed later.
 *
 * @param AtmNloc       Nonlocal influencing atom object for E_k.
 * @param nlocProj      Nonlocal projector object for global Vnl restricted to E_k.
 * @param Ntypes        Number of atom types.
 * @param alpha         Scaling factor.
 * @param X             Vector to be multiplied.
 * @param ldX           Leading dimension of X.
 * @param beta          Scaling factor.
 * @param nlocProj_DDBP Nonlocal projector object for H_DDBP_Ek.
 * @param ncol          Local number of basis functions in this process.
 * @param nALB          Total number of basis functions in this element.
 * @param comm          Communicator over which X is distributed column-wisely.
 */
void vk_Chi_mult(
    ATOM_NLOC_INFLUENCE_OBJ *AtmNloc, NLOC_PROJ_OBJ *nlocProj, int Ntypes,
    double alpha, double *X, int ldX, double beta,
    NLOC_PROJ_OBJ *nlocProj_DDBP, int ncol, int nALB, MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    // if (rank_t == 0) printf("Start finding Chi^T * vk ... \n");
    #endif

    double t1,t2;
    // double t_allgatherv = 0.0, t_trans = 0.0, t_ChiX = 0.0;

    int nproc;
    MPI_Comm_size(comm, &nproc);
    int *recvcounts = (int *)malloc(nproc * sizeof(int));
    int *displs = (int *)malloc(nproc * sizeof(int));
    assert(recvcounts != NULL && displs != NULL);

    for (int ityp = 0; ityp < Ntypes; ityp++) {
        int nproj = nlocProj[ityp].nproj;
        if (!nproj) continue; // this is typical for hydrogen

        // set up recvcounts for gathering ChiX
        for (int p = 0; p < nproc; p++) {
            int ncol_p = block_decompose_BLCYC_fashion(nALB, nproc, p);
            recvcounts[p] = nproj * ncol_p;
        }
        // set up displs for gathering ChiX
        displs[0] = 0;
        for (int p = 1; p < nproc; p++)
            displs[p] = displs[p-1] + recvcounts[p-1];

        int n_atom_type = AtmNloc[ityp].n_atom;
        MPI_Request *req = malloc(n_atom_type * sizeof(*req));
        assert(req != NULL);
        double **ChiX_type = malloc(n_atom_type * sizeof(*ChiX_type));
        assert(ChiX_type != NULL);

        // find Chi^T * vk and initialize Iallgatherv
        for (int iat = 0; iat < n_atom_type; iat++) {
            double *ChiX = (double *)malloc(nproj * ncol * sizeof(double));
            assert(ChiX != NULL);
            ChiX_type[iat] = ChiX; // save for later to deallocate

            int ndc = AtmNloc[ityp].ndc[iat];
            int *grid_pos_J = AtmNloc[ityp].grid_pos[iat];
            t1 = MPI_Wtime();
            Chi_J_X_mult(ndc, nproj, grid_pos_J, alpha, nlocProj[ityp].Chi[iat],
                X, ldX, beta, ChiX, nproj, ncol);
            t2 = MPI_Wtime();
            t_ChiX += t2 - t1;

            t1 = MPI_Wtime();
            // perform allgather to collect ChiX (over bands)
            // MPI_Request request;
            MPI_Iallgatherv(ChiX, nproj*ncol, MPI_DOUBLE, nlocProj_DDBP[ityp].Chi[iat],
                recvcounts, displs, MPI_DOUBLE, comm, &req[iat]);
            // MPI_Wait(&request, MPI_STATUS_IGNORE);
            t2 = MPI_Wtime();
            t_allgatherv += t2 - t1;
            // free(ChiX);
        }

        // t1 = MPI_Wtime();
        // MPI_Waitall(n_atom_type, req, MPI_STATUS_IGNORE);
        // t2 = MPI_Wtime();
        // t_allgatherv += t2 - t1;

        // while non-blocking allgather is going on, find the transpose
        for (int iat = 0; iat < n_atom_type; iat++) {
            t1 = MPI_Wtime();
            MPI_Wait(&req[iat], MPI_STATUS_IGNORE);
            t2 = MPI_Wtime();
            t_allgatherv += t2 - t1;

            t1 = MPI_Wtime();
            // find transpose of the gathered result
            double *Chi = nlocProj_DDBP[ityp].Chi[iat];
            inplace_matrix_traspose('C', Chi, nproj, nALB);
            t2 = MPI_Wtime();
            t_trans += t2 - t1;

            free(ChiX_type[iat]);
        }
        free(ChiX_type);
        free(req);
    }

    // #ifdef DEBUG
    // if (rank_t == 0) printf("== Vnl_DDBP ==: allgather took %.3f ms\n", t_allgatherv*1e3);
    // if (rank_t == 0) printf("== Vnl_DDBP ==: transpose(Chi)*X took %.3f ms\n", t_ChiX*1e3);
    // if (rank_t == 0) printf("== Vnl_DDBP ==: inplace transpose took %.3f ms\n", t_trans*1e3);
    // #endif

    free(recvcounts);
    free(displs);
}



/**
 * @brief   Calculate the nonlocal projectors expressed in the DDBP basis
 *          for the DDBP Hamiltonian.
 *
 * @details The nonlocal part of DDBP Hailtonian is defined by
 *                        Vnl_ddbp := V^T * Vnl * V,
 *          where Vnl = sum_Jlm gamma_Jl |Chi_Jlm> <Chi_Jlm| is the
 *          nonlocal pseudopotential operator.
 *
 *          Substituting the Vnl expression into Vnl_ddbp, we obtain
 *            Vnl_ddbp = sum_Jlm gamma_Jl V^T|Chi_Jlm> <Chi_Jlm|V,
 *                     = sum_Jlm gamma_Jl |Chi'_Jlm> <Chi'_Jlm|,
 *          where |Chi'_Jlm> = V^T |Chi_Jlm> is the nonlocal projectors
 *          expressed in the DDBP basis.
 */
void Calculate_nloc_projectors_DDBP_Hamiltonian(
    SPARC_OBJ *pSPARC, int count, int kpt, int spn_i
)
{
    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start calculating nonlocal projectors in DDBP basis ... \n");
    #endif

    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;

    // re-init the timings to 0
    t_allgatherv = t_trans = t_ChiX = 0.0;

    // find V^T * Chi_Jlm, since we restrict Chi_Jlm to within E_k strictly,
    // the result is still block diagonal
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;
        DDBP_HAMILT_ERBLKS *H_DDBP_Ek = &E_k->H_DDBP_Ek;
        DDBP_VNL *Vnl_DDBP = &H_DDBP_Ek->Vnl_DDBP;
        // find vk^T * Chi_Jlm for all atoms J and quantum numbers l,m
        int ncol = DDBP_info->n_basis_basiscomm;
        int nALB = E_k->nALB;
        vk_Chi_mult(E_k->AtmNloc, E_k->nlocProj, ESPRC_k->Ntypes, 1.0, E_k->v,
            E_k->nd_d, 0.0, Vnl_DDBP->nlocProj, ncol, nALB, DDBP_info->blacscomm);
    }

    #ifdef DEBUG
    if (rank_t == 0) printf("== Vnl_DDBP ==: allgather took %.3f ms\n", t_allgatherv*1e3);
    if (rank_t == 0) printf("== Vnl_DDBP ==: transpose(Chi)*X took %.3f ms\n", t_ChiX*1e3);
    if (rank_t == 0) printf("== Vnl_DDBP ==: inplace transpose took %.3f ms\n", t_trans*1e3);
    #endif
}
