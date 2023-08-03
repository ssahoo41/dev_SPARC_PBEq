/**
 * @file    ddbp_paral.c
 * @brief   This file contains the parallelization functions for the Discrete
 *          Discontinuous Basis Projection (DDBP) method.
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
#include "isddft.h"
#include "tools.h"
#include "ddbp_tools.h"
#include "ddbp_paral.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define TEMP_TOL 1e-14



/**
 * @brief   Set up sub-communicators elemcomm for distributing DDBP elements.
 *
 * @param DDBP_info  Pointer to a DDBP_INFO object.
 * @param comm       Communicator to be splitted.
 */
void create_DDBP_elemcomm(DDBP_INFO *DDBP_info, MPI_Comm comm) {
    // split comm into multiple DDBP elemcomm's
    // DDBP_info->npelem = -1; // let the program choose. TODO: make this an input choice
    DDBP_info->elemcomm_index = create_subcomm(
        comm, &DDBP_info->elemcomm, &DDBP_info->npelem, DDBP_info->Ne_tot
    );

    // assign (find start and end indices of) the elements to each elemcomm
    assign_task_subcomm(
        DDBP_info->Ne_tot, DDBP_info->npelem, DDBP_info->elemcomm_index,
        &DDBP_info->elem_start_index, &DDBP_info->elem_end_index, &DDBP_info->n_elem_elemcomm
    );

    // allocate memory for the assigned element list
    DDBP_info->elem_list = malloc(DDBP_info->n_elem_elemcomm * sizeof(*DDBP_info->elem_list));
}


/**
 * @brief   Set up sub-communicators elemcomm for distributing DDBP elements.
 *
 * @param DDBP_info  Pointer to a DDBP_INFO object.
 * @param comm       Communicator to be splitted.
 */
void create_DDBP_bandcomm(DDBP_INFO *DDBP_info, MPI_Comm comm) {
    // split comm into multiple DDBP bandcomm's (bridge comm between elemcomm's)
    MPI_Comm elemcomm = DDBP_info->elemcomm;
    if (DDBP_info->elemcomm_index < 0) elemcomm = MPI_COMM_NULL;
    DDBP_info->bandcomm_index = create_bridge_comm(
        comm, elemcomm, &DDBP_info->bandcomm);

    // assign (find start and end indices of) the bands to each bandcomm
    assign_task_BLCYC_fashion(
        DDBP_info->Nstates, DDBP_info->npband, DDBP_info->bandcomm_index,
        &DDBP_info->band_start_index, &DDBP_info->band_end_index,
        &DDBP_info->n_band_bandcomm
    );
}


/**
 * @brief Create a bridge comm between subcomm's within a comm.
 *
 * @details Let comm be a communicator, and subcomm's are subcommunicators
 *          within comm. We create bridge communicators between all processes
 *          with the same rank in each subcomm.
 *
 * @param comm The global communicator that includes all the subcomm's.
 * @param subcomm The subcomm this process belongs to.
 * @param bridge_comm The brdige communicator between subcomm's (output).
 */
int create_bridge_comm(MPI_Comm comm, MPI_Comm subcomm, MPI_Comm *bridge_comm)
{
    int rank_subcomm;
    if (subcomm != MPI_COMM_NULL)
        MPI_Comm_rank(subcomm, &rank_subcomm);
    else
        rank_subcomm = -1;

    int color = rank_subcomm;
    if (color < 0) color = INT_MAX; // color must not be negtive
    MPI_Comm_split(comm, color, 0, bridge_comm);
    return rank_subcomm; // can be used as index for bridge_comm
}


/**
 * @brief   Given the index of an element E_k, find which elemcomm it's assigned to.
 *
 *          This is accompanied by the routine above to assign elements. If the way
 *          to assign elements is changed, this has to be modified so that they're
 *          consistent.
 *
 * @param nelem Total number of elements.
 * @param npelem Number of element comms.
 * @param k Global index of an element.
 * @return int Elemcomm_index.
 */
int element_index_to_elemcomm_index(int nelem, int npelem, int k) {
    return block_decompose_rank(nelem, npelem, k);
}


/**
 * @brief   Set up sub-communicators basiscomm for distributing DDBP basis functions.
 *
 * @param DDBP_info  Pointer to a DDBP_INFO object.
 * @param comm       Communicator to be splitted.
 */
void create_DDBP_basiscomm(DDBP_INFO *DDBP_info, MPI_Comm comm) {
    // TODO: here we're assuming that all elements assigned to this comm have the
    //       same number of basis functions!
    int Nb_k = DDBP_info->nALB_tot / DDBP_info->Ne_tot;

    // split comm into multiple DDBP elemcomm's
    // DDBP_info->npbasis = -1; // let the program choose. TODO: make this an input choice
    DDBP_info->basiscomm_index = create_subcomm(
        comm, &DDBP_info->basiscomm, &DDBP_info->npbasis, Nb_k
    );

    // assign (find start and end indices of) the basis to each basiscomm
    assign_task_BLCYC_fashion(
        Nb_k, DDBP_info->npbasis, DDBP_info->basiscomm_index,
        &DDBP_info->basis_start_index, &DDBP_info->basis_end_index, &DDBP_info->n_basis_basiscomm
    );
}


/**
 * @brief   Given the index of a basis, find which basiscomm it's assigned to.
 *
 *          This is accompanied by the routine above to assign basis. If the way
 *          to assign basis is changed, this has to be modified so that they're
 *          consistent. Note that we assume all elements have the same number of
 *          basis functions, therefore communication is not required.
 *
 * @param DDBP_info  Pointer to a DDBP_INFO object.
 * @param k          Global index of an element. (Currently unused.)
 * @param n          Index of a basis within element E_k.
 */
int basis_index_to_basiscomm_index(DDBP_INFO *DDBP_info, int k, int n) {
    int Nb_k = DDBP_info->nALB_tot / DDBP_info->Ne_tot;
    return block_decompose_rank_BLCYC_fashion(Nb_k, DDBP_info->npbasis, n);
}


/**
 * @brief   Given the indices of elecomm, basiscomm, dmcomm, determine the
 *          rank of the corresponding process within kptcomm.
 *
 *          This is accompanied by the routines to create elemcomm, basiscomm,
 *          and dmcomm. If the way these sub-communicators are changed, this has
 *          to be modified so that they're consistent. Note that we assume no
 *          domain paral, so dmcomm only contains a single process.
 *
 * @param DDBP_info       Pointer to a DDBP_INFO object.
 * @param elemcomm_index  Index of elemcomm.
 * @param basiscomm_index Index of basiscomm.
 * @param dmcomm_rank     Rank of process in the dmcomm.
 */
int indices_to_rank_kptcomm(
    DDBP_INFO *DDBP_info, int elemcomm_index, int basiscomm_index,
    int dmcomm_rank, MPI_Comm kptcomm
)
{
    if (dmcomm_rank != 0) {
        printf("\nDomain parallelization for elements is currently not supported!\n");
        exit(EXIT_FAILURE);
    }

    int nproc;
    MPI_Comm_size(kptcomm, &nproc);

    int npelem = DDBP_info->npelem;
    int npbasis = DDBP_info->npbasis;

    // rank of the root of elemcomm with index elemcomm_index in kptcomm
    int size_elemcomm = nproc / npelem;
    int root_elemcomm = elemcomm_index * size_elemcomm;

    // rank of the root of basiscomm with index basiscomm_index in elecomm
    int size_basiscomm = size_elemcomm / npbasis;
    int root_basiscomm = basiscomm_index * size_basiscomm;

    // Warning: here we assume dmcomm_rank is always 0, but in the case it's
    //   not 0, we need to be careful that when creating the Cartesian
    //   topology, if reorder-rank is on, adding the dmcomm_rank might give
    //   wrong result!
    return root_elemcomm + root_basiscomm + dmcomm_rank;
}


/**
 * @brief   Set up an inter-communicator between a sub-communicator
 *          within a communicator and the communicator including the
 *          rest processes.
 *
 * @param comm    A communicator.
 * @param subcomm A sub-communicator in the communicator containing ranks 0,
 *                1,...,nproc_sumbcomm-1, if a process in comm doesn't belong
 *                to subcomm, it should give MPI_COMM_NULL.
 * @param inter   The inter-communicator to be created.
 */
void create_Cart_inter_comm(MPI_Comm comm, MPI_Comm subcomm, MPI_Comm *inter)
{
    int nproc_comm, nproc_subcomm;
    MPI_Comm_size(comm, &nproc_comm);
    if (subcomm != MPI_COMM_NULL)
        MPI_Comm_size(subcomm, &nproc_subcomm);

    MPI_Bcast(&nproc_subcomm, 1, MPI_INT, 0, comm);

    if (nproc_subcomm < nproc_comm) {
#ifdef DEBUG
        double t1 = MPI_Wtime();
#endif
        // first create a comm that includes all the processes that are excluded from the subcomm
        MPI_Group comm_group, comm_group_excl;
        MPI_Comm_group(comm, &comm_group);
        int *incl_ranks, count;
        incl_ranks = (int *)malloc((nproc_comm - nproc_subcomm) * sizeof(int));
        count = 0;
        for (int i = nproc_subcomm; i < nproc_comm; i++) {
            incl_ranks[count] = i; count++;
        }
        MPI_Group_incl(comm_group, count, incl_ranks, &comm_group_excl);
        MPI_Comm excl_comm;
        MPI_Comm_create_group(comm, comm_group_excl, 110, &excl_comm);

        // now create an inter-comm between subcomm and excl_comm
        if (subcomm != MPI_COMM_NULL) {
            MPI_Intercomm_create(subcomm, 0, comm, nproc_subcomm, 111, inter);
        } else {
            MPI_Intercomm_create(excl_comm, 0, comm, 0, 111, inter);
        }

        free(incl_ranks);
        MPI_Group_free(&comm_group);
        MPI_Group_free(&comm_group_excl);
        if (excl_comm != MPI_COMM_NULL)
            MPI_Comm_free(&excl_comm);

#ifdef DEBUG
        double t2 = MPI_Wtime();
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) printf("\n--set up inter-comm took %.3f ms\n" ,(t2-t1)*1000);
#endif
    } else {
        *inter = MPI_COMM_NULL;
    }
}


/**
 * @brief   Set up Cartesian topology for domain parallelization of DDBP basis.
 *
 *          Need to do this after init_DDBP, i.e., after the grids are defined for the
 *          extended elements.
 */
void Embed_Cart_DDBP(SPARC_OBJ *pSPARC) {
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;

    int periods[3];
    periods[0] = 1 - DDBP_info->EBCx;
    periods[1] = 1 - DDBP_info->EBCy;
    periods[2] = 1 - DDBP_info->EBCz;
    int minsize = DDBP_info->fd_order / 2;
    int gridsizes[3] = {INT_MAX, INT_MAX, INT_MAX};
    // go over each element and find the minimum #grids
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        gridsizes[0] = min(gridsizes[0], E_k->nx_ex);
        gridsizes[1] = min(gridsizes[1], E_k->ny_ex);
        gridsizes[2] = min(gridsizes[2], E_k->nz_ex);
    }

    if (DDBP_info->n_elem_elemcomm < 1) {
        gridsizes[0] = gridsizes[1] = gridsizes[2] = 0;
        DDBP_info->dmcomm_dims[0] = 1;
        DDBP_info->dmcomm_dims[1] = 1;
        DDBP_info->dmcomm_dims[2] = 1;
    }

    //------------------------------------------------//
    //                  elemcomm_topo                 //
    //------------------------------------------------//
    // Embed a Cartesian topology in each elemcomm
    // this is analogous to the poisson domain (for Lanczos)
    int nproc_elemcomm;
    MPI_Comm_size(DDBP_info->elemcomm, &nproc_elemcomm);
    create_dmcomm(
        DDBP_info->elemcomm, DDBP_info->elemcomm_index, gridsizes, periods,
        minsize, DDBP_info->elemcomm_topo_dims, &DDBP_info->elemcomm_topo, nproc_elemcomm
    );

    //------------------------------------------------//
    //     inter-communicator elemcomm_topo_inter     //
    //------------------------------------------------//
    // set up an inter-communicator between elemcomm_topo and the rest processes in elemcomm
    if (DDBP_info->elemcomm_index >= 0)
        create_Cart_inter_comm(DDBP_info->elemcomm,
            DDBP_info->elemcomm_topo, &DDBP_info->elemcomm_topo_inter);
    else
        DDBP_info->elemcomm_topo_inter = MPI_COMM_NULL;

    // assign grid points to the processes in the topology
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;
        int gridsizes[3];
        gridsizes[0] = E_k->nx_ex;
        gridsizes[1] = E_k->ny_ex;
        gridsizes[2] = E_k->nz_ex;
        assign_task_Cart(
            DDBP_info->elemcomm_topo, 3, DDBP_info->elemcomm_topo_dims, gridsizes, E_k->DMVert_ex_topo
        );
        E_k->nx_ex_d_topo = E_k->DMVert_ex_topo[1] - E_k->DMVert_ex_topo[0] + 1;
        E_k->ny_ex_d_topo = E_k->DMVert_ex_topo[3] - E_k->DMVert_ex_topo[2] + 1;
        E_k->nz_ex_d_topo = E_k->DMVert_ex_topo[5] - E_k->DMVert_ex_topo[4] + 1;
        E_k->nd_ex_d_topo = E_k->nx_ex_d_topo * E_k->ny_ex_d_topo * E_k->nz_ex_d_topo;
        ESPRC_k->Veff_loc_kptcomm_topo = (double *)calloc(E_k->nd_ex_d_topo, sizeof(double));
        assert(ESPRC_k->Veff_loc_kptcomm_topo != NULL);
        // set Veff in phi-domain to be the same as in kptcomm_topo (for Lanczos)
        ESPRC_k->Veff_loc_dmcomm_phi = ESPRC_k->Veff_loc_kptcomm_topo;

        if (ESPRC_k->isGammaPoint) {
            ESPRC_k->Lanczos_x0 = (double *)malloc(E_k->nd_ex_d_topo * sizeof(double));
            assert(ESPRC_k->Lanczos_x0 != NULL);
        } else {
            ESPRC_k->Lanczos_x0_complex = (double _Complex *)malloc(E_k->nd_ex_d_topo * sizeof(double _Complex));
            assert(ESPRC_k->Lanczos_x0_complex != NULL);
        }
    }

    //------------------------------------------------//
    //                     dmcomm                     //
    //------------------------------------------------//
    // Embed a Cartesian topology in each basiscomm (analogous to domain comm in psi-domain)
    // for now we're restricting ourselves to no domain paral., i.e., only one process
    create_dmcomm(
        DDBP_info->basiscomm, DDBP_info->basiscomm_index, gridsizes, periods,
        minsize, DDBP_info->dmcomm_dims, &DDBP_info->dmcomm, DDBP_info->npdm
    );

    //------------------------------------------------//
    //          blacscomm & ictxt_blacscomm           //
    //------------------------------------------------//
    // create blacscomm in each elemcomm for projection
    int rank_dmcomm = INT_MAX;
    if (DDBP_info->dmcomm != MPI_COMM_NULL)
        MPI_Comm_rank(DDBP_info->dmcomm, &rank_dmcomm);
    int color = rank_dmcomm;
    if (DDBP_info->elemcomm_index == -1 || DDBP_info->basiscomm_index == -1 ||
        DDBP_info->dmcomm == MPI_COMM_NULL)
    {
        color = INT_MAX;
    }
    // split the kptcomm into several cblacscomms using color = rank_dmcomm
    color = (color >= 0) ? color : INT_MAX;
    MPI_Comm_split(DDBP_info->elemcomm, color, DDBP_info->basiscomm_index, &DDBP_info->blacscomm);

    // create a ScaLAPACK context within blacscomm
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int nproc_blacscomm;
    MPI_Comm_size(DDBP_info->blacscomm, &nproc_blacscomm);
    int dims[2], bhandle, ictxt;

    bhandle = Csys2blacs_handle(DDBP_info->blacscomm); // create a context out of blacscomm
    ictxt = bhandle;
    // for original blacscomm context
    dims[0] = 1; dims[1] = nproc_blacscomm;
    Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);
    DDBP_info->ictxt_blacs = ictxt;

    // for block-cyclic grid
    bhandle = Csys2blacs_handle(DDBP_info->blacscomm); // create a context out of blacscomm
    ictxt = bhandle;
    dims[0] = nproc_blacscomm; dims[1] = 1;
    // int gridsizes_2d[2] = {gridsizes[0]*gridsizes[1]*gridsizes[2], DDBP_info->nALB_tot};
    // int dims[2], ierr;
    // // for square matrices of size < 20000, it doesn't scale well beyond 64 proc
    // SPARC_Dims_create(nproc_blacscomm, 2, gridsizes_2d, 256, dims, &ierr);
    // if (ierr) {
    //     dims[0] = nproc_blacscomm;
    //     dims[1] = 1;
    // }
    // TODO: swap dim[0] and dim[1] value, since SPARC_Dims_create tends to give larger dim for dim[1] on a tie situation
    Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);
    DDBP_info->ictxt_blacs_topo = ictxt;
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    // assign the basis (find start and end indices of) to each dmcomm
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;
        int gridsizes[3];
        gridsizes[0] = E_k->nx_ex;
        gridsizes[1] = E_k->ny_ex;
        gridsizes[2] = E_k->nz_ex;
        assign_task_Cart(
            DDBP_info->dmcomm, 3, DDBP_info->dmcomm_dims, gridsizes, E_k->DMVert_ex
        );
        E_k->nx_ex_d = E_k->DMVert_ex[1] - E_k->DMVert_ex[0] + 1;
        E_k->ny_ex_d = E_k->DMVert_ex[3] - E_k->DMVert_ex[2] + 1;
        E_k->nz_ex_d = E_k->DMVert_ex[5] - E_k->DMVert_ex[4] + 1;
        E_k->nd_ex_d = E_k->nx_ex_d * E_k->ny_ex_d * E_k->nz_ex_d;

        // Note: this won't work correctly if there is more than 1 proc in dmcomm! since in
        //   mat-vec operations we assume it's distributed using block_decompose_ routines
        E_k->DMVert[0] = max(E_k->DMVert_ex[0], E_k->is - E_k->is_ex);
        E_k->DMVert[1] = min(E_k->DMVert_ex[1], E_k->ie - E_k->is_ex);
        E_k->DMVert[2] = max(E_k->DMVert_ex[2], E_k->js - E_k->js_ex);
        E_k->DMVert[3] = min(E_k->DMVert_ex[3], E_k->je - E_k->js_ex);
        E_k->DMVert[4] = max(E_k->DMVert_ex[4], E_k->ks - E_k->ks_ex);
        E_k->DMVert[5] = min(E_k->DMVert_ex[5], E_k->ke - E_k->ks_ex);
        E_k->nx_d = E_k->DMVert[1] - E_k->DMVert[0] + 1;
        E_k->ny_d = E_k->DMVert[3] - E_k->DMVert[2] + 1;
        E_k->nz_d = E_k->DMVert[5] - E_k->DMVert[4] + 1;
        E_k->nd_d = E_k->nx_d * E_k->ny_d * E_k->nz_d;
        E_k->nd_d = max(E_k->nd_d, 0); // in case there's no overlap, set #grid to 0

        // allocate memory for the assigned extended basis, and init v_tilde
        int nspn = pSPARC->Nspin_spincomm;
        int nkpt = pSPARC->Nkpts_kptcomm;
        int nrow = E_k->nd_ex_d;
        int ncol = DDBP_info->n_basis_basiscomm;
        // size of each matrix
        int size_k = nrow * ncol;
        int size_s = size_k * nkpt;
        if (ESPRC_k->isGammaPoint) {
            E_k->v_tilde = malloc(E_k->nd_ex_d * ncol * nspn * sizeof(*(E_k->v_tilde)));
            E_k->v = calloc(E_k->nd_d * ncol * nspn, sizeof(*(E_k->v)));
            E_k->v_prev = calloc(E_k->nd_d * ncol * nspn, sizeof(*(E_k->v_prev)));
            assert(E_k->v_tilde != NULL);
            assert(E_k->v != NULL);
            assert(E_k->v_prev != NULL);
            // init v_tilde randomly
            for(int spn_i = 0; spn_i < nspn; spn_i++) {
                SetRandMat(
                    E_k->v_tilde + spn_i*size_s,
                    nrow, ncol, -0.5, 0.5, pSPARC->spincomm
                );
            }
        } else {
            E_k->v_tilde_cmplx = malloc(E_k->nd_ex_d * ncol * nkpt * nspn * sizeof(*(E_k->v_tilde_cmplx)));
            E_k->v_cmplx = calloc(E_k->nd_d * ncol * nkpt * nspn, sizeof(*(E_k->v_cmplx)));
            E_k->v_prev_cmplx = calloc(E_k->nd_d * ncol * nkpt * nspn, sizeof(*(E_k->v_prev_cmplx)));
            assert(E_k->v_tilde_cmplx != NULL);
            assert(E_k->v_cmplx != NULL);
            assert(E_k->v_prev_cmplx != NULL);
            // init v_tilde_cmplx randomly
            for(int spn_i = 0; spn_i < nspn; spn_i++) {
                for (int kpt_i = 0; kpt_i < nkpt; kpt_i++) {
                    SetRandMat_complex(
                        E_k->v_tilde_cmplx + spn_i*size_s + kpt_i*size_k,
                        nrow, ncol, -0.5, 0.5, pSPARC->spincomm
                    );
                }
            }
        }

        // allocate memory for basis overlap matrix (Mvvp = v^T * vp)
        int nALB = E_k->nALB;
        E_k->Mvvp = malloc(nspn * sizeof(*E_k->Mvvp));
        assert(E_k->Mvvp != NULL);
        for(int spn_i = 0; spn_i < nspn; spn_i++) {
            E_k->Mvvp[spn_i] = malloc(nkpt * sizeof(*E_k->Mvvp[spn_i]));
            assert(E_k->Mvvp[spn_i] != NULL);
            for (int kpt_i = 0; kpt_i < nkpt; kpt_i++) {
                E_k->Mvvp[spn_i][kpt_i] = calloc(nALB * nALB, sizeof(double));
                assert(E_k->Mvvp[spn_i][kpt_i] != NULL);
                // init Mvvp to identity
                double *Mvvp_i = E_k->Mvvp[spn_i][kpt_i];
                for (int ind = 0; ind < nALB*nALB; ind+=nALB+1) {
                    // Mvvp_i(j,j) = Mvvp_i[j + j*nALB] = 1.0
                    Mvvp_i[ind] = 1.0;
                }
            }
        }

        // copy the ddbp communicators/indices into the ESPRC_k obj
        ESPRC_k->dmcomm = DDBP_info->dmcomm;
        ESPRC_k->dmcomm_phi = DDBP_info->elemcomm_topo;
        ESPRC_k->kptcomm_topo = DDBP_info->elemcomm_topo;
        ESPRC_k->kptcomm_inter = DDBP_info->elemcomm_topo_inter;
        ESPRC_k->kptcomm = DDBP_info->elemcomm;
        ESPRC_k->bandcomm = DDBP_info->basiscomm;
        ESPRC_k->blacscomm = DDBP_info->blacscomm;
        ESPRC_k->ictxt_blacs = DDBP_info->ictxt_blacs;
        ESPRC_k->ictxt_blacs_topo = DDBP_info->ictxt_blacs_topo;
        // comm index
        ESPRC_k->kptcomm_index = DDBP_info->elemcomm_index;
        ESPRC_k->bandcomm_index = DDBP_info->basiscomm_index;
        // paral params
        ESPRC_k->npband = DDBP_info->npbasis;
        ESPRC_k->npNd   = DDBP_info->npdm;
        ESPRC_k->npNdx  = DDBP_info->dmcomm_dims[0];
        ESPRC_k->npNdy  = DDBP_info->dmcomm_dims[1];
        ESPRC_k->npNdz  = DDBP_info->dmcomm_dims[2];
        // dmcomm_phi is equal to kptcomm_topo in this case
        ESPRC_k->is_phi_eq_kpt_topo = 1;

        // load distribution
        // ESPRC_k->band_start_indx = E_k->ALB_ns;
        // ESPRC_k->band_end_indx = E_k->ALB_ne;
        // ESPRC_k->Nband_bandcomm = E_k->nALB;
        ESPRC_k->band_start_indx = DDBP_info->basis_start_index;
        ESPRC_k->band_end_indx = DDBP_info->basis_end_index;
        ESPRC_k->Nband_bandcomm = DDBP_info->n_basis_basiscomm;


        // Domain paral
        for (int i = 0; i < 6; ++i) {
            ESPRC_k->DMVertices_dmcomm[i] = E_k->DMVert_ex[i];
            ESPRC_k->DMVertices_kptcomm[i] = E_k->DMVert_ex_topo[i];
        }
        ESPRC_k->Nd_d_dmcomm = E_k->nd_ex_d;

        // orbitals/basis funcs, we'll try to use E_k->v_tilde directry
        ESPRC_k->Xorb = E_k->v_tilde;
        ESPRC_k->Xorb_kpt = E_k->v_tilde_cmplx;

#if defined(USE_MKL) || defined(USE_SCALAPACK)
        // set up ScaLAPACK descriptors for the extended element basis calculation
        int ZERO = 0, mb, nb, llda, info;
        mb = max(1, E_k->nd_ex_d);
        nb = (E_k->nALB - 1) / DDBP_info->npbasis + 1; // equal to ceil(nALB/npbasis)
        // set up descriptor for storage of extended basis funcs in ictxt_blacscomm (original)
        llda = max(1, E_k->nd_ex_d);
        if (DDBP_info->basiscomm_index != -1 && DDBP_info->dmcomm != MPI_COMM_NULL) {
            descinit_(ESPRC_k->desc_orbitals, &E_k->nd_ex_d, &ESPRC_k->Nstates,
                  &mb, &nb, &ZERO, &ZERO, &ESPRC_k->ictxt_blacs, &llda, &info);
        } else {
            for (int i = 0; i < 9; i++) ESPRC_k->desc_orbitals[i] = -1;
        }

        // set up descriptor for storage of extended basis funcs in ictxt_blacscomm
        mb = max(1, E_k->nd_ex_d / dims[0]); // this is only block, no cyclic! Tune this to improve efficiency!
        nb = max(1, ESPRC_k->Nstates / dims[1]); // this is only block, no cyclic!
        int nprow, npcol, myrow, mycol;
        Cblacs_gridinfo(ESPRC_k->ictxt_blacs_topo, &nprow, &npcol, &myrow, &mycol);
        // find number of rows/cols of the local distributed orbitals
        if (ESPRC_k->bandcomm_index != -1 && ESPRC_k->dmcomm != MPI_COMM_NULL) {
            ESPRC_k->nr_orb_BLCYC = numroc_( &E_k->nd_ex_d, &mb, &myrow, &ZERO, &nprow);
            ESPRC_k->nc_orb_BLCYC = numroc_( &ESPRC_k->Nstates, &nb, &mycol, &ZERO, &npcol);
        } else {
            ESPRC_k->nr_orb_BLCYC = 1;
            ESPRC_k->nc_orb_BLCYC = 1;
        }
        llda = max(1, ESPRC_k->nr_orb_BLCYC);
        if (ESPRC_k->bandcomm_index != -1 && ESPRC_k->dmcomm != MPI_COMM_NULL) {
            descinit_(ESPRC_k->desc_orb_BLCYC, &E_k->nd_ex_d, &ESPRC_k->Nstates,
                      &mb, &nb, &ZERO, &ZERO, &ESPRC_k->ictxt_blacs_topo, &llda, &info);
        } else {
            for (int i = 0; i < 9; i++)
                ESPRC_k->desc_orb_BLCYC[i] = -1;
        }

        // set up ScaLAPACK descriptors for the element basis calculation (restriced)
        mb = max(1, E_k->nd_d);
        nb = (E_k->nALB - 1) / DDBP_info->npbasis + 1; // equal to ceil(nALB/npbasis)
        // set up descriptor for storage of extended basis funcs in ictxt_blacscomm (original)
        llda = max(1, E_k->nd_d);
        if (DDBP_info->basiscomm_index != -1 && DDBP_info->dmcomm != MPI_COMM_NULL) {
            descinit_(E_k->desc_v, &E_k->nd_d, &E_k->nALB,
                  &mb, &nb, &ZERO, &ZERO, &ESPRC_k->ictxt_blacs, &llda, &info);
        } else {
            for (int i = 0; i < 9; i++) E_k->desc_v[i] = -1;
        }

        // set up descriptor for storage of subspace eigenproblem
        // the maximum Nstates up to which we will use LAPACK to solve
        // the subspace eigenproblem in serial
        // int MAX_NS = 2000;
        int MAX_NS = ESPRC_k->eig_serial_maxns;
        ESPRC_k->useLAPACK = (ESPRC_k->Nstates <= MAX_NS) ? 1 : 0;

        // calculate maximum number of processors for eigenvalue solver
        if (ESPRC_k->useLAPACK == 0) {
            // ESPRC_k->eig_paral_maxnp = pSPARC->eig_paral_maxnp;
            int gridsizes[2] = {ESPRC_k->Nstates,ESPRC_k->Nstates}, ierr = 1, size_blacscomm;
            MPI_Comm_size(ESPRC_k->blacscomm, &size_blacscomm);
            SPARC_Dims_create(min(size_blacscomm,ESPRC_k->eig_paral_maxnp),
                2, gridsizes, 1, ESPRC_k->eig_paral_subdims, &ierr);
            if (ierr) ESPRC_k->eig_paral_subdims[0] = ESPRC_k->eig_paral_subdims[1] = 1;
            // #ifdef DEBUG
            //     if (rank == 0) printf("\nMaximun number of processors for eigenvalue solver is %d\n", ESPRC_k->eig_paral_maxnp);
            //     if (rank == 0) printf("The dimension of subgrid for eigen sovler is (%d x %d).\n", 
            //                             ESPRC_k->eig_paral_subdims[0], ESPRC_k->eig_paral_subdims[1]);
            // #endif
        }

        int mbQ, nbQ, lldaQ;

        // block size for storing Hp and Mp
        if (ESPRC_k->useLAPACK == 1) {
            // in this case we will call LAPACK instead to solve the subspace eigenproblem
            mb = nb = ESPRC_k->Nstates;
            mbQ = nbQ = 64; // block size for storing subspace eigenvectors
        } else {
            // in this case we will use ScaLAPACK to solve the subspace eigenproblem
            mb = nb = ESPRC_k->eig_paral_blksz;
            mbQ = nbQ = ESPRC_k->eig_paral_blksz; // block size for storing subspace eigenvectors
        }

        if (ESPRC_k->bandcomm_index != -1 && ESPRC_k->dmcomm != MPI_COMM_NULL) {
            ESPRC_k->nr_Hp_BLCYC = ESPRC_k->nr_Mp_BLCYC = numroc_( &ESPRC_k->Nstates, &mb, &myrow, &ZERO, &nprow);
            ESPRC_k->nr_Hp_BLCYC = ESPRC_k->nr_Mp_BLCYC = max(1, ESPRC_k->nr_Mp_BLCYC);
            ESPRC_k->nc_Hp_BLCYC = ESPRC_k->nc_Mp_BLCYC = numroc_( &ESPRC_k->Nstates, &nb, &mycol, &ZERO, &npcol);
            ESPRC_k->nc_Hp_BLCYC = ESPRC_k->nc_Mp_BLCYC = max(1, ESPRC_k->nc_Mp_BLCYC);
            ESPRC_k->nr_Q_BLCYC = numroc_( &ESPRC_k->Nstates, &mbQ, &myrow, &ZERO, &nprow);
            ESPRC_k->nc_Q_BLCYC = numroc_( &ESPRC_k->Nstates, &nbQ, &mycol, &ZERO, &npcol);
        } else {
            ESPRC_k->nr_Hp_BLCYC = ESPRC_k->nc_Hp_BLCYC = 1;
            ESPRC_k->nr_Mp_BLCYC = ESPRC_k->nc_Mp_BLCYC = 1;
            ESPRC_k->nr_Q_BLCYC  = ESPRC_k->nc_Q_BLCYC  = 1;
        }

        llda = max(1, ESPRC_k->nr_Hp_BLCYC);
        lldaQ= max(1, ESPRC_k->nr_Q_BLCYC);
        if (ESPRC_k->bandcomm_index != -1 && ESPRC_k->dmcomm != MPI_COMM_NULL) {
            descinit_(&ESPRC_k->desc_Hp_BLCYC[0], &ESPRC_k->Nstates, &ESPRC_k->Nstates,
                      &mb, &nb, &ZERO, &ZERO, &ESPRC_k->ictxt_blacs_topo, &llda, &info);
            for (int i = 0; i < 9; i++) {
                //ESPRC_k->desc_Q_BLCYC[i] = ESPRC_k->desc_Mp_BLCYC[i] = ESPRC_k->desc_Hp_BLCYC[i];
                ESPRC_k->desc_Mp_BLCYC[i] = ESPRC_k->desc_Hp_BLCYC[i];
            }
            descinit_(&ESPRC_k->desc_Q_BLCYC[0], &ESPRC_k->Nstates, &ESPRC_k->Nstates,
                      &mbQ, &nbQ, &ZERO, &ZERO, &ESPRC_k->ictxt_blacs_topo, &lldaQ, &info);
        } else {
            for (int i = 0; i < 9; i++) {
                ESPRC_k->desc_Q_BLCYC[i] = ESPRC_k->desc_Mp_BLCYC[i] = ESPRC_k->desc_Hp_BLCYC[i] = -1;
            }
        }
#else
        ESPRC_k->useLAPACK = 1;
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)


        // if (ESPRC_k->dmcomm != MPI_COMM_NULL) {
            ESPRC_k->Veff_loc_dmcomm = (double *)calloc(E_k->nd_ex_d * nspn, sizeof(double));
            assert(ESPRC_k->Veff_loc_dmcomm != NULL);
            E_k->Veff_loc_dmcomm_prev = (double *)calloc(E_k->nd_ex_d * nspn, sizeof(double));
            assert(E_k->Veff_loc_dmcomm_prev != NULL);
        // }

        // allocate memory for storing eigenvalues
        ESPRC_k->lambda = (double *)calloc(ESPRC_k->Nstates * ESPRC_k->Nkpts_kptcomm * ESPRC_k->Nspin_spincomm, sizeof(double));
        assert(ESPRC_k->lambda != NULL);
        ESPRC_k->lambda_sorted = ESPRC_k->lambda;

        ESPRC_k->eigmin = (double *) malloc(ESPRC_k->Nkpts_kptcomm * ESPRC_k->Nspin_spincomm * sizeof (double));
        ESPRC_k->eigmax = (double *) malloc(ESPRC_k->Nkpts_kptcomm * ESPRC_k->Nspin_spincomm * sizeof (double));
    }
}


/**
 * @brief   Set up a Cartesian topology in a communicator for domain decomposition
 *          according to the given grid and dimensions (if given).
 *
 * @param comm        Given communicator to be embeded.
 * @param comm_index  Index corresponding to comm, if set to -1, this comm is not embeded
 * @param gridsizes   Number of grids in all three dimentions.
 * @param periods     Periodic flags, 1 - periodic, 0 - not periodic.
 * @param minsize     Minimum number of grids each process must contain in each dimention.
 * @param dims        The dimension of the Cartesian topology, if set to {-1,-1,-1}, will
 *                    be found automatically. (Input/Output)
 * @param subcomm     Subcommunicators. (Output)
 * @param nsubcomm    Maximum number of subcomms to be created, might not be used in full.
 */
void create_dmcomm(
    const MPI_Comm comm, const int comm_index, const int gridsizes[3], const int periods[3],
    const int minsize, int dims[3], MPI_Comm *subcomm, int nsubcomm)
{
    if (comm == MPI_COMM_NULL) {
        dims[0] = dims[1] = dims[2] = 0;
        *subcomm = MPI_COMM_NULL;
        return;
    }
    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    // if dims are not provided, find an optimized set of dims
    if (dims[0] <=0 || dims[1] <= 0 || dims[2] <= 0) {
        if (nsubcomm < 0) nsubcomm = nproc;
        int ierr;
        SPARC_Dims_create(nsubcomm, 3, gridsizes, minsize, dims, &ierr);
        if (ierr < 0) exit(EXIT_FAILURE);
    }

    if (comm_index < 0)
        *subcomm = MPI_COMM_NULL;
    else
        MPI_Cart_create(comm, 3, dims, periods, 1, subcomm); // 1 is to reorder rank
}


/**
 * @brief Split comm into nsubcomm subcomm's for the parallelization of a given number of objects.
 * @param comm        Given communicator to be splitted.
 * @param subcomm     Sub-communicators.
 * @param nsubcomm    Maximum number of subcomms to be created, might not be used in full,
 *                    if set to <= 0, will be changed to min(nproc,n_obj).
 * @param n_obj       Number of objects to be distributed across the subcomms.
 **/
int create_subcomm(const MPI_Comm comm, MPI_Comm *subcomm, int *nsubcomm, const int n_obj)
{
    if (comm == MPI_COMM_NULL || n_obj == 0) {
        *nsubcomm = 0;
        *subcomm = MPI_COMM_NULL;
        return -1;
    }

    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    // if nsubcomm is not set, use as many nsubcomm as possible
    if (*nsubcomm <= 0) {
        *nsubcomm = min(nproc, n_obj);
    }

    int size_subcomm = nproc / *nsubcomm;
    int subcomm_index = -1;

    if (rank < (nproc - nproc % *nsubcomm)) {
        subcomm_index = rank / size_subcomm;
    }

    int color = subcomm_index;

    // if (color < 0) color = MPI_UNDEFINED;
    if (color < 0) color = INT_MAX; // color must not be negtive
    // color += 1;
    MPI_Comm_split(comm, color, 0, subcomm);

    // TODO: confirm if this is safe
    // if (subcomm_index < 0) {
    //     MPI_Comm_free(subcomm);
    //     *subcomm = MPI_COMM_NULL;
    // }

    return subcomm_index;
}



/**
 * @brief Assign objects to nsubcomm subcomm's.
 * @param nsubcomm         Number of subcomms.
 * @param n_obj            Number of objects to be distributed across the subcomms.
 * @param subcomm_index    Index of the current subcomm (LOCAL).
 * @param obj_start_index  (OUTPUT) Start index of the assigned objects (LOCAL).
 * @param obj_end_index    (OUTPUT) End index of the assigned objects (LOCAL).
 * @param n_obj_subcomm    (OUTPUT) Number of the assigned objects (LOCAL).
 **/
void assign_task_subcomm(
    const int n_obj, const int nsubcomm, const int subcomm_index,
    int *obj_start_index, int *obj_end_index, int *n_obj_subcomm)
{
    if (subcomm_index == -1) {
        *n_obj_subcomm = 0;
        *obj_start_index = 0;
    } else {
        *n_obj_subcomm = block_decompose(n_obj, nsubcomm, subcomm_index);
        *obj_start_index = block_decompose_nstart(n_obj, nsubcomm, subcomm_index);
    }
    *obj_end_index = *obj_start_index + *n_obj_subcomm - 1;
}


/**
 * @brief   Calculate number of rows/cols of a distributed array owned by
 *          the process (in one direction) using the block-cyclic way,
 *          except we force no cyclic.
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int block_decompose_BLCYC_fashion(const int n, const int p, const int rank)
{
    if (rank < 0) return 0;
    // set block size to ceil(n/p) to force no cyclic
    int nb = (n - 1) / p + 1;
    int nblocks = n / nb; // find number of whole blocks
    int numroc = (nblocks / p) * nb; // min #row/col a proc can have
    // int numroc = 0; // in our case, min #row/col a proc can have is 0
    int extrablks = nblocks % p; // this is always nblocks since we use ceil
    int mydist = rank; // if start index of matrix /= 0, => (rank + p - is) % p
    if (mydist < extrablks) {
        numroc += nb;
    } else if (mydist == extrablks) {
        numroc += n % nb;
    }
    return numroc;
}


/**
 * @brief   Calculate the start index of a distributed array owned by
 *          the process (in one direction) using the block-cyclic way,
 *          except we force no cyclic.
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int block_decompose_nstart_BLCYC_fashion(const int n, const int p, const int rank)
{
    // set block size to ceil(n/p) to force no cyclic
    int nb = (n - 1) / p + 1;
    int nblocks = n / nb; // find number of whole blocks
    int nstart = 0;
    if (rank <= nblocks) {
        nstart = rank * nb;
    } else {
        nstart = n; // point to the end of the array
    }
    return nstart;
}


/**
 * @brief   Calculate which process owns the provided node of a distributed
 *          array (in one direction) using the block-cyclic way, except we
 *          force no cyclic.
 *
 * @param n           Number of nodes in the given direction of the global domain.
 * @param p           Total number of processes in the given direction of the
 *                    process topology.
 * @param node_index  Node index.
 */
int block_decompose_rank_BLCYC_fashion(
    const int n, const int p, const int node_indx
)
{
    assert(node_indx < n);
    // set block size to ceil(n/p) to force no cyclic
    int nb = (n - 1) / p + 1;
    int rank = node_indx / nb;
    return rank;
}


/**
 * @brief Assign objects to nsubcomm subcomm's using the block-cyclic fashion, except
 *        force no cyclic, but only block. This is slightly different from the block
 *        decomposition we do for domain parallelization.
 *
 * @param nsubcomm         Number of subcomms.
 * @param n_obj            Number of objects to be distributed across the subcomms.
 * @param subcomm_index    Index of the current subcomm (LOCAL).
 * @param obj_start_index  (OUTPUT) Start index of the assigned objects (LOCAL).
 * @param obj_end_index    (OUTPUT) End index of the assigned objects (LOCAL).
 * @param n_obj_subcomm    (OUTPUT) Number of the assigned objects (LOCAL).
 **/
void assign_task_BLCYC_fashion(
    const int n_obj, const int nsubcomm, const int subcomm_index,
    int *obj_start_index, int *obj_end_index, int *n_obj_subcomm)
{
    if (subcomm_index == -1) {
        *n_obj_subcomm = 0;
        *obj_start_index = 0;
    } else {
        *n_obj_subcomm = block_decompose_BLCYC_fashion(n_obj, nsubcomm, subcomm_index);
        *obj_start_index = block_decompose_nstart_BLCYC_fashion(n_obj, nsubcomm, subcomm_index);
    }
    *obj_end_index = *obj_start_index + *n_obj_subcomm - 1;
}


/**
 * @brief Assign objects to a communicator with Cartesian topology.
 *
 * @param comm             Communicator with Cartesian topology.
 * @param ndims            Number of dimensions of Cartesian grid.
 * @param dims             Number of processes in each dimension.
 * @param gridsizes        Number of grids/objects in each dimension.
 * @param DMvert (OUTPUT)  The vertices of the domain assigned to the current process.
 *
 **/
void assign_task_Cart(
    MPI_Comm comm, int ndims, const int dims[], const int gridsizes[], int DMvert[]
)
{
    if (comm == MPI_COMM_NULL) {
        for (int n = 0; n < ndims; n++) {
            DMvert[2*n] = 0;
            DMvert[2*n+1] = gridsizes[n] - 1; // or set it to 0
        }
        return;
    }

    int rank_cart;
    int *coord_cart = malloc(ndims * sizeof(*coord_cart));
    MPI_Comm_rank(comm, &rank_cart);
    MPI_Cart_coords(comm, rank_cart, ndims, coord_cart);

    for (int n = 0; n < ndims; n++) {
        int ng_d = block_decompose(gridsizes[n], dims[n], coord_cart[n]);
        DMvert[2*n] = block_decompose_nstart(gridsizes[n], dims[n], coord_cart[n]);
        DMvert[2*n+1] = DMvert[2*n] + ng_d - 1;
    }

    free(coord_cart);
}



/**
 * @brief   Set up default paral. parameters for DDBP method if not set by user.
 *
 */
void set_default_DDBP_paral_params(DDBP_INFO *DDBP_info, MPI_Comm comm) {
    int np;
    MPI_Comm_size(comm, &np);

    int Ne_tot = DDBP_info->Ne_tot; // Number of elements
    int Nb_k = DDBP_info->nALB_tot / DDBP_info->Ne_tot; // number of basis per element

    // if user provides one of them
    if (DDBP_info->npelem > 0 || DDBP_info->npbasis > 0) {
        if (DDBP_info->npelem < 0) {
            DDBP_info->npelem = np / DDBP_info->npbasis;
        } else {
            DDBP_info->npbasis = np / DDBP_info->npelem;
        }
    } else {
        // set up npelem, npbasis
        // dims_divide_2d(Ne_tot, Nb_k, np, &DDBP_info->npelem, &DDBP_info->npbasis);
        dims_divide_2d(Nb_k, Ne_tot, np, &DDBP_info->npbasis, &DDBP_info->npelem);
        // printf("Ne_tot = %d, Nb_k = %d, np = %d, npelem = %d, npbasis = %d\n",
        //         Ne_tot, Nb_k, np, DDBP_info->npelem, DDBP_info->npbasis);
    }

    // set npband
    DDBP_info->npband = DDBP_info->npbasis;

    // set up domain parallelization params in each elemcomm
    DDBP_info->elemcomm_topo_dims[0] = -1;
    DDBP_info->elemcomm_topo_dims[1] = -1;
    DDBP_info->elemcomm_topo_dims[2] = -1;

    // set up npdm for domain parallelization in each basiscomm
    // TODO: implement domain paral. for DDBP, remove the following once implemented
    DDBP_info->dmcomm_dims[0] = 1;
    DDBP_info->dmcomm_dims[1] = 1;
    DDBP_info->dmcomm_dims[2] = 1;
    DDBP_info->npdm = 1;
}


/**
 * @brief Allocate memory and initialize variables for DDBP Hamiltonian.
 */
void init_DDBP_Hamiltonian(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, DDBP_HAMILT_ERBLKS *H_DDBP_Ek)
{
    H_DDBP_Ek->nblks = E_k->n_element_nbhd;
    H_DDBP_Ek->blksz = E_k->nALB;
    // TODO: for large nALB, consider distributing h_kj and perform
    // TODO: h_kj * psi in parallel using pdgemm!
    H_DDBP_Ek->isserial = 'T';

    size_t blk_msize = H_DDBP_Ek->blksz * H_DDBP_Ek->blksz * sizeof(double);
    H_DDBP_Ek->h_kj[6] = (double *)malloc(blk_msize);
    assert(H_DDBP_Ek->h_kj[6]);

    // int BCs[3] = {DDBP_info->BCx,DDBP_info->BCy,DDBP_info->BCz};
    int Nes[3] = {DDBP_info->Nex,DDBP_info->Ney,DDBP_info->Nez};
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
        // which direction
        int dim = nbr_i / 2;
        // which face of neighbor, 0:left, 1:right
        int face = nbr_i % 2;

        // *if there's only one element in this direction, skip this direction
        if (Nes[dim] == 1) {
            H_DDBP_Ek->h_kj[nbr_i] = NULL;
            continue;
        }

        // *if there're two elements in this direction, there's only one neighbor
        // *element. For PBC, we can merge the non-zeros and skip any of the
        // *neighbors. For DBC, we'll have to skip the corresponding neighbor,
        // *whether left or right depends on the element coordinates.
        if (Nes[dim] == 2) { // coord can only be 0 or 1 in this case
            if (E_k->coords[dim] == face) {
                H_DDBP_Ek->h_kj[nbr_i] = NULL;
                continue;
            }
        }

        H_DDBP_Ek->h_kj[nbr_i] = (double *)malloc(blk_msize);
        assert(H_DDBP_Ek->h_kj[nbr_i]);
    }

    // allocate memory for DDBP nonlocal projectors
    // * since it needs to be done for every MD step, we move this to another place
    // init_nlocProj_DDBP(DDBP_info, E_k);
}




/**
 * @brief   Generate a send/recv tag.
 *
 *          Since each process might contain multiple elements, and the
 *          non-blocking Isend/Irecv takes only ranks and tags to identify
 *          the exact data transfer, the only way to differentiate the
 *          transfer between different elements (even if they are from
 *          the same sender and receiver) is to use a unique tag to
 *          identify each unique transfer.
 *          Since one can use a tuple (Ei->Ej, side), meaning from E_i to
 *          E_j, plus a side (0-5) of the domain to define a transfer,
 *          we want to generate a unique tag based these 3 numbers. This
 *          kind of problem is related to the hash functions.
 */
int generate_tag(int Ne_tot, int sindex, int rindex, int side) {
    // int tag = (sindex * Ne_tot + rindex) * Ne_tot + side;
    int tag = (side * Ne_tot + rindex) * Ne_tot + sindex;

    // upper bound of MPI tag value, for some implementation of MPI, MPI_TAG_UB is set to 0
    const int tag_ub = (MPI_TAG_UB ? MPI_TAG_UB : 1073741823); // 1073741823 = 2^30-1
    tag = tag % tag_ub;
    // tag must be non-negative!
    if (tag < 0) tag += tag_ub;
    return tag;
}



/**
 * @brief   Set up send/recv parameters for the non-blocking data transfer
 *          (halo exchange) between neighbor elements.
 */
void set_haloX_Hvk_params(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, int ncol, MPI_Comm kptcomm)
{
    SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    int FDn = ESPRC_k->order / 2;

    haloX_t *haloX = &E_k->haloX_Hv;

    haloX->n_neighbors = 6;
    haloX->sendtype = MPI_DOUBLE;
    haloX->recvtype = MPI_DOUBLE;

    // element partition info
    int dims[3];
    dims[0] = DDBP_info->Nex;
    dims[1] = DDBP_info->Ney;
    dims[2] = DDBP_info->Nez;
    int periods[3];
    periods[0] = 1 - DDBP_info->BCx;
    periods[1] = 1 - DDBP_info->BCy;
    periods[2] = 1 - DDBP_info->BCz;

    // domain size info
    int DMnx = E_k->nx_d;
    int DMny = E_k->ny_d;
    int DMnz = E_k->nz_d;
    // int DMnd = E_k->nd_d;
    int DMnx_ex = DMnx + FDn * 2;
    int DMny_ex = DMny + FDn * 2;
    int DMnz_ex = DMnz + FDn * 2;
    // int DMnd_ex = DMnx_ex * DMny_ex * DMnz_ex;
    // set up send/recv buffer params
    int nxex_in = DMnx_ex - FDn;
    int nyex_in = DMny_ex - FDn;
    int nzex_in = DMnz_ex - FDn;
    int istart[6] = {0,         nxex_in,   FDn,       FDn,        FDn,       FDn},
          iend[6] = {FDn-1,     DMnx_ex-1, nxex_in-1, nxex_in-1,  nxex_in-1, nxex_in-1},
        jstart[6] = {FDn,       FDn,       0,         nyex_in,    FDn,       FDn},
          jend[6] = {nyex_in-1, nyex_in-1, FDn-1,     DMny_ex-1,  nyex_in-1, nyex_in-1},
        kstart[6] = {FDn,       FDn,       FDn,       FDn,        0,         nzex_in},
          kend[6] = {nzex_in-1, nzex_in-1, nzex_in-1, nzex_in-1,  FDn-1,     DMnz_ex-1};
    int nx_in = DMnx - FDn;
    int ny_in = DMny - FDn;
    int nz_in = DMnz - FDn;
    int isrecv[6] = {0,      nx_in,  0,      0,      0,      0     };
    int ierecv[6] = {FDn-1,  DMnx-1, DMnx-1, DMnx-1, DMnx-1, DMnx-1};
    int jsrecv[6] = {0,      0,      0,      ny_in,  0,      0     };
    int jerecv[6] = {DMny-1, DMny-1, FDn-1,  DMny-1, DMny-1, DMny-1};
    int ksrecv[6] = {0,      0,      0,      0,      0,      nz_in};
    int kerecv[6] = {DMnz-1, DMnz-1, DMnz-1, DMnz-1, FDn-1,  DMnz-1};
    int my_index = E_k->index;
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
        // which direction
        int dir = nbr_i / 2;
        // which side of neighbor, -1:left, +1:right
        int side = nbr_i % 2 ? 1 : -1;

        // find neighbor element index
        int index_ngbr;
        DDBP_Cart_shift(dims, periods, dir, side, my_index, &index_ngbr);

        if (index_ngbr < 0) {
            printf("dims = [%d,%d,%d], periods = [%d,%d,%d], dir = %d, side = %d, my_index = %d ==> index_ngbr = %d\n",
                dims[0],dims[1],dims[2], periods[0], periods[1], periods[2],
                dir, side, my_index, index_ngbr);
        }
        haloX->neighbor_indices[nbr_i] = index_ngbr; // TODO: for debugging only

        // find which rank owns this neighbor
        int owner_rank = -1;
        if (DDBP_info->basiscomm_index >= 0) {
            owner_rank = DDBP_basis_owner(
                DDBP_info, index_ngbr, -1, DDBP_info->basiscomm_index, 0, kptcomm
            );
        }

        haloX->neighbor_ranks[nbr_i] = owner_rank; // -1 if no neighbor in this dir
        haloX->stags[nbr_i] = generate_tag(DDBP_info->Ne_tot, my_index, index_ngbr, nbr_i);
        haloX->rtags[nbr_i] = generate_tag(DDBP_info->Ne_tot, index_ngbr, my_index, 2*dir+(-side+1)/2);

        int i_s = istart[nbr_i];
        int i_e = iend  [nbr_i];
        int j_s = jstart[nbr_i];
        int j_e = jend  [nbr_i];
        int k_s = kstart[nbr_i];
        int k_e = kend  [nbr_i];
        haloX->issend[nbr_i] = i_s;
        haloX->iesend[nbr_i] = i_e;
        haloX->jssend[nbr_i] = j_s;
        haloX->jesend[nbr_i] = j_e;
        haloX->kssend[nbr_i] = k_s;
        haloX->kesend[nbr_i] = k_e;

        // TODO: set recv indices -> isrecv[6], ...
        haloX->isrecv[nbr_i] = isrecv[nbr_i];
        haloX->ierecv[nbr_i] = ierecv[nbr_i];
        haloX->jsrecv[nbr_i] = jsrecv[nbr_i];
        haloX->jerecv[nbr_i] = jerecv[nbr_i];
        haloX->ksrecv[nbr_i] = ksrecv[nbr_i];
        haloX->kerecv[nbr_i] = kerecv[nbr_i];

        // the number of elements to send to neighbor nbr_i
        haloX->sendcounts[nbr_i] = ncol * (i_e-i_s+1) * (j_e-j_s+1) * (k_e-k_s+1);

        // the number of elements to receive from neighbor nbr_i
        haloX->recvcounts[nbr_i] = haloX->sendcounts[nbr_i];
    }

    // set up displacements
    haloX->sdispls[0] = 0;
    haloX->rdispls[0] = 0;
    for (int nbr_i = 0; nbr_i < 5; nbr_i++) {
        // the displacement (offset from sendbuf, in units of sendtype)
        // from which to send data to neighbor
        haloX->sdispls[nbr_i+1] = haloX->sdispls[nbr_i] + haloX->sendcounts[nbr_i];
        // the displacement (offset from sendbuf, in units of sendtype)
        // to which data from neighbor nbr_i should be written
        haloX->rdispls[nbr_i+1] = haloX->rdispls[nbr_i] + haloX->recvcounts[nbr_i];
    }
}



/**
 * @brief   Set up send/recv parameters for the non-blocking data transfer
 *          (halo exchange) between neighbor elements.
 */
void set_haloX_DDBP_Array_Ek(
    DDBP_INFO *DDBP_info, int k, DDBP_ARRAY *X, haloX_t *haloX, MPI_Comm bandcomm)
{
    DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
    // SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    // int FDn = ESPRC_k->order / 2;

    haloX->n_neighbors = 6;
    haloX->sendtype = MPI_DOUBLE;
    haloX->recvtype = MPI_DOUBLE;

    // element partition info
    int dims[3];
    dims[0] = DDBP_info->Nex;
    dims[1] = DDBP_info->Ney;
    dims[2] = DDBP_info->Nez;
    int periods[3];
    periods[0] = 1 - DDBP_info->BCx;
    periods[1] = 1 - DDBP_info->BCy;
    periods[2] = 1 - DDBP_info->BCz;

    int ncol = X->ncol;
    int nALB = E_k->nALB;
    int my_index = E_k->index;
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
        // which direction
        int dir = nbr_i / 2;
        int face = nbr_i % 2; // which face of neighbor, 0:left, 1:right
        // shift of neighbor, -1:left, +1:right
        int side = nbr_i % 2 ? 1 : -1;

        // find neighbor element index
        int index_ngbr;
        DDBP_Cart_shift(dims, periods, dir, side, my_index, &index_ngbr);

        if (index_ngbr < 0) {
            printf("dims = [%d,%d,%d], periods = [%d,%d,%d], dir = %d, side = %d, my_index = %d ==> index_ngbr = %d\n",
                dims[0],dims[1],dims[2], periods[0], periods[1], periods[2],
                dir, side, my_index, index_ngbr);
        }
        haloX->neighbor_indices[nbr_i] = index_ngbr; // TODO: for debugging only

        // find which rank owns this neighbor in bandcomm
        int owner_rank = element_index_to_elemcomm_index(DDBP_info->Ne_tot, DDBP_info->npelem, index_ngbr);

        haloX->neighbor_ranks[nbr_i] = owner_rank; // -1 if no neighbor in this dir
        // check special cases
        if (dims[dir] == 1) {
            // no neighbor in this case
            haloX->neighbor_ranks[nbr_i] = -1;
        }
        if (dims[dir] == 2) { // coord can only be 0 or 1 in this case
            if (E_k->coords[dir] == face) {
                // in this case, left/right neighbors are the same process
                haloX->neighbor_ranks[nbr_i] = -1;
            }
        }

        haloX->stags[nbr_i] = generate_tag(DDBP_info->Ne_tot, my_index, index_ngbr, nbr_i);
        haloX->rtags[nbr_i] = generate_tag(DDBP_info->Ne_tot, index_ngbr, my_index, 2*dir+(1-face));
        haloX->issend[nbr_i] = 0;
        haloX->iesend[nbr_i] = 0;
        haloX->jssend[nbr_i] = 0;
        haloX->jesend[nbr_i] = 0;
        haloX->kssend[nbr_i] = 0;
        haloX->kesend[nbr_i] = 0;
        haloX->isrecv[nbr_i] = 0;
        haloX->ierecv[nbr_i] = 0;
        haloX->jsrecv[nbr_i] = 0;
        haloX->jerecv[nbr_i] = 0;
        haloX->ksrecv[nbr_i] = 0;
        haloX->kerecv[nbr_i] = 0;
        // the number of elements to send to neighbor nbr_i
        haloX->sendcounts[nbr_i] = ncol * nALB;
        // the number of elements to receive from neighbor nbr_i
        haloX->recvcounts[nbr_i] = haloX->sendcounts[nbr_i];
    }

    // set up displacements
    // *since the send data is the whole vector, the sendbuf will be the same for
    // *all neighbors
    haloX->sdispls[0] = 0;
    haloX->rdispls[0] = 0;
    for (int nbr_i = 0; nbr_i < 5; nbr_i++) {
        // the displacement (offset from sendbuf, in units of sendtype)
        // from which to send data to neighbor
        // *since the send data is the whole vector, the sendbuf will be the same for
        // *all neighbors
        haloX->sdispls[nbr_i+1] = 0;
        // the displacement (offset from sendbuf, in units of sendtype)
        // to which data from neighbor nbr_i should be written
        haloX->rdispls[nbr_i+1] = haloX->rdispls[nbr_i] + haloX->recvcounts[nbr_i];
    }
}


/**
 * @brief Set the up haloX info for DDBP array.
 *
 * @param DDBP_info DDBP_info object.
 * @param X DDBP array.
 * @param bandcomm Communicator where X is distributed. // TODO: not used anymore
 */
void setup_haloX_DDBP_Array(DDBP_INFO *DDBP_info, DDBP_ARRAY *X, MPI_Comm bandcomm)
{
    int nelem = X->nelem;
    for (int k = 0; k < nelem; k++)
        set_haloX_DDBP_Array_Ek(DDBP_info, k, X, &X->haloX_info[k], bandcomm);
}


/**
 * @brief Allocate memory or initialize variables for DDBP.
 */
void allocate_init_DDBP_variables(SPARC_OBJ *pSPARC) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
    DDBP_HAMILTONIAN *H_DDBP = &DDBP_info->H_DDBP;

    H_DDBP->nelem = DDBP_info->n_elem_elemcomm;
    H_DDBP->elem_list = DDBP_info->elem_list;
    H_DDBP->H_DDBP_Ek_list = malloc(H_DDBP->nelem * sizeof(*(H_DDBP->H_DDBP_Ek_list)));
    assert(H_DDBP->H_DDBP_Ek_list != NULL);

    H_DDBP->Ntypes = pSPARC->Ntypes;
    H_DDBP->n_atom = pSPARC->n_atom;
    // these pointers points to the global arrays
    H_DDBP->nAtomv = pSPARC->nAtomv;
    H_DDBP->localPsd = pSPARC->localPsd;
    H_DDBP->IP_displ = pSPARC->IP_displ;
    H_DDBP->psd = pSPARC->psd;

    int Nkpts = pSPARC->Nkpts_kptcomm;
    int Nspin = pSPARC->Nspin_spincomm;
    int nelem = DDBP_info->n_elem_elemcomm;
    int Nband = DDBP_info->n_band_bandcomm; // #bands (KS orbitals)
    int nbasis = DDBP_info->n_basis_basiscomm; // #basis (ALB)
    int nALB = 0; // number of basis in one element, set up later
    int Nstates = DDBP_info->Nstates; // (global) number of states

    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        nALB = E_k->nALB; // we assume it's the same for all elements!

        // init the DDBP Hamiltonian object for element E_k
        init_DDBP_Hamiltonian(DDBP_info, E_k, &E_k->H_DDBP_Ek);
        H_DDBP->H_DDBP_Ek_list[k] = &E_k->H_DDBP_Ek;

        // set up halo exchange info for performing halo exchange of Hvk
        set_haloX_Hvk_params(DDBP_info, E_k, nbasis, pSPARC->kptcomm);
        // print_haloX(E_k, &E_k->haloX_Hv, kptcomm);
    }

    // set up init guess for Lanczos on H_DDBP
    DDBP_ARRAY *X0 = &DDBP_info->Lanczos_x0;
    init_DDBP_Array(DDBP_info, 1, X0, DDBP_info->bandcomm);
    randomize_DDBP_Array(X0, DDBP_info->bandcomm);

    // allocate memory for eigenvalues
    // DDBP_info->lambda = calloc(Nstates * Nkpts * Nspin, sizeof(double));
    // assert(DDBP_info->lambda != NULL);
    DDBP_info->lambda = pSPARC->lambda;

    // set up DDBP Kohn-Sham orbitals
    // set up xorb
    DDBP_info->xorb = malloc(Nspin * sizeof(*DDBP_info->xorb));
    assert(DDBP_info->xorb != NULL);
    for(int spn_i = 0; spn_i < Nspin; spn_i++) {
        DDBP_info->xorb[spn_i] = malloc(Nkpts * sizeof(DDBP_ARRAY));
        assert(DDBP_info->xorb[spn_i] != NULL);
        for (int kpt = 0; kpt < Nkpts; kpt++) {
            DDBP_ARRAY *X_ks = &DDBP_info->xorb[spn_i][kpt];
            init_DDBP_Array(DDBP_info, Nband, X_ks, DDBP_info->bandcomm);
            randomize_DDBP_Array(X_ks, pSPARC->kptcomm);
        }
    }
    // set up yorb
    DDBP_ARRAY *Y = &DDBP_info->yorb;
    init_DDBP_Array(DDBP_info, Nband, Y, DDBP_info->bandcomm);

    // set up KS orbitals on original FD grid
    DDBP_info->psi = malloc(Nspin * sizeof(*DDBP_info->psi));
    assert(DDBP_info->psi != NULL);
    for(int spn_i = 0; spn_i < Nspin; spn_i++) {
        DDBP_info->psi[spn_i] = malloc(Nkpts * sizeof(*DDBP_info->psi[spn_i]));
        assert(DDBP_info->psi[spn_i] != NULL);
        for (int kpt = 0; kpt < Nkpts; kpt++) {
            DDBP_info->psi[spn_i][kpt] = malloc(nelem * sizeof(double*));
            assert(DDBP_info->psi[spn_i][kpt] != NULL);
            for (int k = 0; k < nelem; k++) {
                DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
                DDBP_info->psi[spn_i][kpt][k] = malloc(E_k->nd_d * Nband * sizeof(double));
                assert(DDBP_info->psi[spn_i][kpt][k] != NULL);
            }
        }
    }

    // set up electron density on original FD grid (element distribution)
    DDBP_info->rho = malloc(nelem * sizeof(*DDBP_info->rho));
    assert(DDBP_info->rho != NULL);
    for (int k = 0; k < nelem; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        DDBP_info->rho[k] = malloc(E_k->nd_d * (2*Nspin-1) * sizeof(double));
        assert(DDBP_info->rho[k] != NULL);
    }

    // for(int spn_i = 0; spn_i < Nspin; spn_i++) {
    //     DDBP_info->rho[spn_i] = malloc(Nkpts * sizeof(*DDBP_info->rho[spn_i]));
    //     assert(DDBP_info->rho[spn_i] != NULL);
    //     for (int kpt = 0; kpt < Nkpts; kpt++) {
    //         DDBP_info->rho[spn_i][kpt] = malloc(nelem * sizeof(double*));
    //         assert(DDBP_info->rho[spn_i][kpt] != NULL);
    //         for (int k = 0; k < nelem; k++) {
    //             DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
    //             DDBP_info->rho[spn_i][kpt][k] = malloc(E_k->nd_d * Nband * sizeof(double));
    //             assert(DDBP_info->rho[spn_i][kpt][k] != NULL);
    //         }
    //     }
    // }

#if defined(USE_MKL) || defined(USE_SCALAPACK)
    // create a context corresponding to bandcomm (original grid)
    int nproc_elemcomm;
    MPI_Comm_size(DDBP_info->elemcomm, &nproc_elemcomm);
    int bhandle = Csys2blacs_handle(DDBP_info->elemcomm); // create a context out of bandcomm
    int ictxt = bhandle;
    // for original blacscomm context
    int dims[2];
    dims[0] = 1; dims[1] = nproc_elemcomm;
    Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);
    DDBP_info->ictxt_elemcomm = ictxt;

    // set up descriptor for DDBP KS orbitals (original band paral in each elemcomm)
    // ! Note we assume all elements have the same number of basis (nALB)
    int ZERO = 0, mb, nb, llda, info;
    int Nrows = nALB; // global #rows
    int Ncols = Nstates; // global #cols
    mb = max(1, Nrows);
    nb = (Ncols - 1) / DDBP_info->npband + 1; // equal to ceil(Ncols/npband)
    // set up descriptor for storage of KS orbitals in ictxt_bandcomm (original)
    llda = max(1, Nrows);
    if (DDBP_info->bandcomm_index != -1 && DDBP_info->bandcomm != MPI_COMM_NULL) {
        descinit_(DDBP_info->desc_xorb, &Nrows, &Ncols, &mb, &nb,
            &ZERO, &ZERO, &DDBP_info->ictxt_elemcomm, &llda, &info);
        assert(info == 0);
    } else {
        for (int i = 0; i < 9; i++) DDBP_info->desc_xorb[i] = -1;
    }

    // set up descriptor for KS orbitals on original FD grid
    DDBP_info->desc_psi = malloc(nelem * sizeof(int*));
    assert(DDBP_info->desc_psi != NULL);
    for (int k = 0; k < nelem; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        DDBP_info->desc_psi[k] = malloc(9 * sizeof(int));
        assert(DDBP_info->desc_psi[k] != NULL);
        int ZERO = 0, mb, nb, llda, info;
        int Nrows = E_k->nd_d; // global #rows
        int Ncols = Nstates; // global #cols
        mb = max(1, Nrows);
        nb = (Ncols - 1) / DDBP_info->npband + 1; // equal to ceil(Ncols/npband)
        // set up descriptor for storage of KS orbitals in ictxt_bandcomm (original)
        llda = max(1, Nrows);
        if (DDBP_info->bandcomm_index != -1 && DDBP_info->bandcomm != MPI_COMM_NULL) {
            descinit_(DDBP_info->desc_psi[k], &Nrows, &Ncols, &mb, &nb,
                &ZERO, &ZERO, &DDBP_info->ictxt_elemcomm, &llda, &info);
            assert(info == 0);
        } else {
            for (int i = 0; i < 9; i++) DDBP_info->desc_psi[k][i] = -1;
        }
    }

    // create a context for projection and rotation

    // create a context for subspace eigensolver
    int np_best = best_max_nproc(Nstates, Nstates, "pdgemm");
    int ierr, gridsizes[2] = {Nstates,Nstates};
    SPARC_Dims_create(min(nproc_elemcomm,np_best), 2, gridsizes, 256, dims, &ierr);
    ierr = 1;
    int ishift = 8;
    while (ierr && ishift) {
        SPARC_Dims_create(min(nproc_elemcomm,np_best), 2, gridsizes, 1<<ishift, dims, &ierr);
        ishift--;
    }
    if (ierr) dims[0] = dims[1] = 1;

#ifdef DEBUG
    if (rank == 0) printf("eigentopo size = [%d,%d]\n",dims[0],dims[1]);
#endif

    bhandle = Csys2blacs_handle(DDBP_info->elemcomm);
    ictxt = bhandle; // a process grid within elemcomm

    Cblacs_gridinit(&ictxt, "Row", dims[0], dims[1]);
    DDBP_info->ictxt_elemcomm_eigentopo = ictxt;

    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(DDBP_info->ictxt_elemcomm_eigentopo, &nprow, &npcol, &myrow, &mycol );

    // set up mass matrix Ms and subspace Hamiltonian Hs
    int MAX_NS = pSPARC->eig_serial_maxns;
    pSPARC->useLAPACK = (Nstates <= MAX_NS) ? 1 : 0;
    int mbQ, nbQ, lldaQ;
    // block size for storing Hp and Mp
    if (pSPARC->useLAPACK == 1) {
        // in this case we will call LAPACK instead to solve the subspace eigenproblem
        mb = nb = Nstates;
        mbQ = nbQ = 64; // block size for storing subspace eigenvectors
    } else {
        // in this case we will use ScaLAPACK to solve the subspace eigenproblem
        mb = nb = pSPARC->eig_paral_blksz;
        mbQ = nbQ = pSPARC->eig_paral_blksz; // block size for storing subspace eigenvectors
    }
#ifdef DEBUG
    if (!rank) printf("rank = %d, mb = nb = %d, mbQ = nbQ = %d\n", rank, mb, mbQ);
#endif

    if (DDBP_info->bandcomm_index != -1 && DDBP_info->bandcomm != MPI_COMM_NULL) {
        DDBP_info->nr_Hp_BLCYC = numroc_(&Nstates, &mb, &myrow, &ZERO, &nprow);
        DDBP_info->nc_Hp_BLCYC = numroc_(&Nstates, &nb, &mycol, &ZERO, &npcol);
        DDBP_info->nr_Mp_BLCYC = numroc_(&Nstates, &mb, &myrow, &ZERO, &nprow);
        DDBP_info->nc_Mp_BLCYC = numroc_(&Nstates, &nb, &mycol, &ZERO, &npcol);
        DDBP_info->nr_Q_BLCYC  = numroc_(&Nstates, &mbQ, &myrow, &ZERO, &nprow);
        DDBP_info->nc_Q_BLCYC  = numroc_(&Nstates, &nbQ, &mycol, &ZERO, &npcol);
        // make sure all the numbers are greater than 0
        DDBP_info->nr_Hp_BLCYC = max(1, DDBP_info->nr_Hp_BLCYC);
        DDBP_info->nc_Hp_BLCYC = max(1, DDBP_info->nc_Hp_BLCYC);
        DDBP_info->nr_Mp_BLCYC = max(1, DDBP_info->nr_Mp_BLCYC);
        DDBP_info->nc_Mp_BLCYC = max(1, DDBP_info->nc_Mp_BLCYC);
        DDBP_info->nr_Q_BLCYC  = max(1, DDBP_info->nr_Q_BLCYC);
        DDBP_info->nc_Q_BLCYC  = max(1, DDBP_info->nc_Q_BLCYC);
    } else {
        DDBP_info->nr_Hp_BLCYC = DDBP_info->nc_Hp_BLCYC = 1;
        DDBP_info->nr_Mp_BLCYC = DDBP_info->nc_Mp_BLCYC = 1;
        DDBP_info->nr_Q_BLCYC  = DDBP_info->nc_Q_BLCYC  = 1;
    }

    llda = max(1, DDBP_info->nr_Hp_BLCYC);
    lldaQ= max(1, DDBP_info->nr_Q_BLCYC);
    if (DDBP_info->bandcomm_index != -1 && DDBP_info->bandcomm != MPI_COMM_NULL
        && DDBP_info->ictxt_elemcomm_eigentopo >= 0)
    {
        descinit_(DDBP_info->desc_Hp_BLCYC, &Nstates, &Nstates, &mb, &nb, &ZERO,
            &ZERO, &DDBP_info->ictxt_elemcomm_eigentopo, &llda, &info);
        for (int i = 0; i < 9; i++) {
            DDBP_info->desc_Mp_BLCYC[i] = DDBP_info->desc_Hp_BLCYC[i];
        }
        descinit_(DDBP_info->desc_Q_BLCYC, &Nstates, &Nstates, &mbQ, &nbQ, &ZERO,
            &ZERO, &DDBP_info->ictxt_elemcomm_eigentopo, &lldaQ, &info);
    } else {
        for (int i = 0; i < 9; i++) {
            DDBP_info->desc_Q_BLCYC[i] = DDBP_info->desc_Mp_BLCYC[i]
            = DDBP_info->desc_Hp_BLCYC[i] = -1;
        }
    }
#ifdef DEBUG
    if (!rank)
        printf("rank = %d, nr_Hp = %d, nc_Hp = %d\n",
            rank, DDBP_info->nr_Hp_BLCYC, DDBP_info->nc_Hp_BLCYC);
#endif

    // allocate memory for block cyclic distribution of projected Hamiltonian and mass matrix
    if (pSPARC->isGammaPoint){
        DDBP_info->Hp = (double *)calloc(DDBP_info->nr_Hp_BLCYC * DDBP_info->nc_Hp_BLCYC, sizeof(double));
        DDBP_info->Mp = (double *)calloc(DDBP_info->nr_Mp_BLCYC * DDBP_info->nc_Mp_BLCYC, sizeof(double));
        DDBP_info->Q  = (double *)calloc(DDBP_info->nr_Q_BLCYC  * DDBP_info->nc_Q_BLCYC , sizeof(double));
        assert(DDBP_info->Hp != NULL);
        assert(DDBP_info->Mp != NULL);
        assert(DDBP_info->Q  != NULL);
    } else {
        DDBP_info->Hp_kpt = (double _Complex *)malloc(DDBP_info->nr_Hp_BLCYC * DDBP_info->nc_Hp_BLCYC * sizeof(double _Complex));
        DDBP_info->Mp_kpt = (double _Complex *)malloc(DDBP_info->nr_Mp_BLCYC * DDBP_info->nc_Mp_BLCYC * sizeof(double _Complex));
        DDBP_info->Q_kpt  = (double _Complex *)malloc(DDBP_info->nr_Q_BLCYC  * DDBP_info->nc_Q_BLCYC  * sizeof(double _Complex));
        assert(DDBP_info->Hp_kpt != NULL);
        assert(DDBP_info->Mp_kpt != NULL);
        assert(DDBP_info->Q_kpt  != NULL);
    }
#else
    // TODO: implement corresponding routines without MKL/ScaLAPACK
    assert(0);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
}



/**
 * @brief   Start the non-blocking data transfer (halo exchange) between neighbor
 *          elements.
 *
 * @param sendbuf  The buffer array with data that will be sent out.
 * @param recvbuf  The buffer array which will be used to receive data.
 * @param kptcomm  The global kptcomm that includes all the elemcomm's.
 */
void DDBP_element_Ineighbor_alltoallv(
    haloX_t *haloX, const void *sendbuf, void *recvbuf, MPI_Comm kptcomm)
{
    int my_rank;
    MPI_Comm_rank(kptcomm, &my_rank);

    MPI_Aint lower_bound;
    MPI_Aint extent_sendtype, extent_recvtype;
    MPI_Type_get_extent(haloX->sendtype, &lower_bound, &extent_sendtype);
    MPI_Type_get_extent(haloX->recvtype, &lower_bound, &extent_recvtype);
    int n_neighbors = haloX->n_neighbors;
    // *Warning: here we perform the send and recv even if the target is itself!
    for (int dim = 0, i = 0; dim < 3; ++dim) {
        int r0 = haloX->neighbor_ranks[i];
        if (r0 >= 0) {
            MPI_Isend(sendbuf + haloX->sdispls[i] * extent_sendtype,
                haloX->sendcounts[i], haloX->sendtype, r0, haloX->stags[i], kptcomm, &haloX->requests[i]);
            MPI_Irecv(recvbuf + haloX->rdispls[i] * extent_recvtype,
                haloX->recvcounts[i], haloX->recvtype, r0, haloX->rtags[i], kptcomm, &haloX->requests[n_neighbors+i]);
        } else {
            haloX->requests[i] = MPI_REQUEST_NULL;
            haloX->requests[n_neighbors+i] = MPI_REQUEST_NULL;
        }
        ++i;
        int r1 = haloX->neighbor_ranks[i];
        if (r1 >= 0) {
            MPI_Isend(sendbuf + haloX->sdispls[i] * extent_sendtype,
                haloX->sendcounts[i], haloX->sendtype, r1, haloX->stags[i], kptcomm, &haloX->requests[i]);
            MPI_Irecv(recvbuf + haloX->rdispls[i] * extent_recvtype,
                haloX->recvcounts[i], haloX->recvtype, r1, haloX->rtags[i], kptcomm, &haloX->requests[n_neighbors+i]);
        } else {
            haloX->requests[i] = MPI_REQUEST_NULL;
            haloX->requests[n_neighbors+i] = MPI_REQUEST_NULL;
        }
        ++i;
    }
}



/**
 * @brief   Start the REMOTE non-blocking data transfer (halo exchange) between
 *          neighbor elements.
 *
 *          This routine only starts the MPI_Isend and MPI_Irecv if the targe
 *          is not the process itself. It SKIPS all the communication to/from
 *          itself and leaves that part of data untouched.
 *
 * @param sendbuf  The buffer array with data that will be sent out.
 * @param recvbuf  The buffer array which will be used to receive data.
 * @param kptcomm  The global kptcomm that includes all the elemcomm's.
 */
void DDBP_remote_element_Ineighbor_alltoallv(
    haloX_t *haloX, const void *sendbuf, void *recvbuf, MPI_Comm kptcomm)
{
    int my_rank;
    MPI_Comm_rank(kptcomm, &my_rank);

    MPI_Aint lower_bound;
    MPI_Aint extent_sendtype, extent_recvtype;
    MPI_Type_get_extent(haloX->sendtype, &lower_bound, &extent_sendtype);
    MPI_Type_get_extent(haloX->recvtype, &lower_bound, &extent_recvtype);
    int n_neighbors = haloX->n_neighbors;
    // skip (replace with local direct copying) the send and recv if the target is itself!
    for (int dim = 0, i = 0; dim < 3; ++dim) {
        int r0 = haloX->neighbor_ranks[i];
        if (r0 != my_rank && r0 >= 0) {
            MPI_Isend(sendbuf + haloX->sdispls[i] * extent_sendtype,
                haloX->sendcounts[i], haloX->sendtype, r0, haloX->stags[i], kptcomm, &haloX->requests[i]);
            MPI_Irecv(recvbuf + haloX->rdispls[i] * extent_recvtype,
                haloX->recvcounts[i], haloX->recvtype, r0, haloX->rtags[i], kptcomm, &haloX->requests[n_neighbors+i]);
        } else {
            haloX->requests[i] = MPI_REQUEST_NULL;
            haloX->requests[n_neighbors+i] = MPI_REQUEST_NULL;
        }
        ++i;
        int r1 = haloX->neighbor_ranks[i];
        if (r1 != my_rank && r1 >= 0) {
            MPI_Isend(sendbuf + haloX->sdispls[i] * extent_sendtype,
                haloX->sendcounts[i], haloX->sendtype, r1, haloX->stags[i], kptcomm, &haloX->requests[i]);
            MPI_Irecv(recvbuf + haloX->rdispls[i] * extent_recvtype,
                haloX->recvcounts[i], haloX->recvtype, r1, haloX->rtags[i], kptcomm, &haloX->requests[n_neighbors+i]);
        } else {
            haloX->requests[i] = MPI_REQUEST_NULL;
            haloX->requests[n_neighbors+i] = MPI_REQUEST_NULL;
        }
        ++i;
    }
}



/**
 * @brief   Start the LOCAL data transfer (halo exchange) between neighbor
 *          elements.
 *
 *          This routine only performs the data transfer if the target process
 *          is the process itself. This routine is to be used together with
 *          its REMOTE counterpart
 *              DDBP_remote_element_Ineighbor_alltoallv().
 *
 *
 * @param sendbuf  The buffer array with data that will be sent out.
 * @param recvbuf  The buffer array which will be used to receive data.
 * @param kptcomm  The global kptcomm that includes all the elemcomm's.
 */
void DDBP_local_element_Ineighbor_alltoallv(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, haloX_t *haloX,
    const void *sendbuf, void *recvbuf, MPI_Comm kptcomm)
{
    int my_rank;
    MPI_Comm_rank(kptcomm, &my_rank);

    MPI_Aint lower_bound;
    MPI_Aint extent_sendtype, extent_recvtype;
    MPI_Type_get_extent(haloX->sendtype, &lower_bound, &extent_sendtype);
    MPI_Type_get_extent(haloX->recvtype, &lower_bound, &extent_recvtype);
    // int n_neighbors = haloX->n_neighbors;
    // skip (replace with local direct copying) the send and recv if the target is itself!
    for (int dim = 0, i = 0; dim < 3; ++dim) {
        int r0 = haloX->neighbor_ranks[i];
        if (r0 == my_rank) {
            int k0 = haloX->neighbor_indices[i];
            int k0_loc = k0 - DDBP_info->elem_start_index; // local index
            DDBP_ELEM *E_k0 = &DDBP_info->elem_list[k0_loc];
            haloX_t *haloX0 = &E_k0->haloX_Hv;
            int i0 = 2 * dim + 1;
            memcpy((void *)E_k0->recvbuf + haloX0->rdispls[i0] * extent_recvtype,
                sendbuf + haloX->sdispls[i] * extent_sendtype, haloX->sendcounts[i] * extent_sendtype);
        }
        ++i;
        int r1 = haloX->neighbor_ranks[i];
        if (r1 == my_rank) {
            int k1 = haloX->neighbor_indices[i];
            int k1_loc = k1 - DDBP_info->elem_start_index; // local index
            DDBP_ELEM *E_k1 = &DDBP_info->elem_list[k1_loc];
            haloX_t *haloX1 = &E_k1->haloX_Hv;
            int i1 = 2 * dim;
            memcpy((void *)E_k1->recvbuf + haloX1->rdispls[i1] * extent_recvtype,
                sendbuf + haloX->sdispls[i] * extent_sendtype, haloX->sendcounts[i] * extent_sendtype);
        }
        ++i;
    }
}



/**
 * @brief Given coordinates and dims of the cartesian topology, return a unique index
 * to a unique coordinates. This can be viewd as a simple type of hashing.
 *
 * @param dims Dimensions of the Cartesian topology the coordinates are relative to.
 * @param i Coordinate in the 1st dimension.
 * @param j Coordinate in the 2nd dimension.
 * @param k Coordinate in the 3rd dimension.
 * @return int A unique index corresponding to the coordinates.
 */
int coord_to_index(const int dims[3], int i, int j, int k)
{
    int index = i + dims[0] * (j + k * dims[1]);
    return index;
}

/**
 * @brief Given index and dims of the cartesian topology, return the coordinates
 * This can be viewd as a simple type of hashing.
 *
 * @param dims Dimensions of the Cartesian topology the coordinates are relative to.
 * @param index A unique index corresponding to the coordinates.
 * @param i Coordinate in the 1st dimension (output).
 * @param j Coordinate in the 2nd dimension (output).
 * @param k Coordinate in the 3rd dimension (output).
 */
void index_to_coord(const int dims[3], int index, int *i, int *j, int *k)
{
    assert(index >= 0);
    int base = dims[0] * dims[1];
    int kk = index / base;
    index -= kk * base;
    base = dims[0];
    int jj = index / base;
    index -= jj * base;
    int ii = index;
    // copy values to ouput vars
    *i = ii; *j = jj; *k = kk;
}



void colcommind_rowcommind_to_union_rank(
    int colcomm_sind, int colcomm_eind, int n_targets, int rowstride,
    int colstride, const int *target_rowinds,  int *target_ranks
)
{
    int colcomm_nind = colcomm_eind - colcomm_sind + 1;
    for (int n = colcomm_nind-1; n >= 0; n--) {
        int *target_ranks_n = target_ranks + n * n_targets;
        int colcomm_ind = n + colcomm_sind;
        for (int i = 0; i < n_targets; i++) {
            target_ranks_n[i] = target_rowinds[i] * rowstride + colcomm_ind * colstride;
        }
    }
}



/**
 * @brief Determines process rank in communicator given Cartesian
 *        location. This routine is to replace the MPI_Cart_rank
 *        routine, which requires the calling process to be within
 *        the catesian topology.
 *
 *        WARNING: if for some reason, the MPI in-built routine
 *        MPI_Cart_rank uses a different algorithm, don't use this!
 *
 * @param ndims Number of dimensions of the cartesian topology.
 * @param dims Integer array of size 'ndims'.
 * @param coords Integer array (of size  'ndims') specifying the cartesian
 *               coordinates of a process.
 * @param rank rank of specified process (integer).
 * 
 * @ref This routine is copied from https://github.com/open-mpi/ompi.git.
 */
int topo_base_cart_rank(int ndims, const int *dims, const int *coords, int *rank)
{
   int prank, dim, ord, factor, i;
   const int *d;

   /*
    * Loop over coordinates computing the rank.
    */
    factor = 1;
    prank = 0;

    i = ndims - 1;
    d = dims + i;
    for (; i >= 0; --i, --d) {
        dim = *d;
        ord = coords[i];
        /* Per MPI-2.1 7.5.4 (description of MPI_CART_RANK), if the
        dimension is periodic and the coordinate is outside of 0 <=
        coord(i) < dim, then normalize it.  If the dimension is not
        periodic, it's an error. */
        if ((ord < 0) || (ord >= dim)) {
            ord %= dim;
            if (ord < 0) {
                ord += dim;
            }
        }
        prank += factor * ord;
        factor *= dim;
    }
    *rank = prank;

    return 0;
}


// find ranks to send in union_comm
// ! This routine makes the following assumptions
// * assumption 1: every sender is also active as a receiver (colcomm != MPI_COMM_NULL)
// * assumption 2: every colcomm has the same order of arranging ranks in the Cart topology
void colcommind_coords_to_ranks(
    int colcomm_sind, int colcomm_eind, int n_targets,
    const int *target_coords,  int *target_ranks,
    MPI_Comm rowcomm, int rowsize, MPI_Comm colcomm, const int *coldims,
    int colcomm_index, MPI_Comm union_comm
)
{
    int rank, rank_union_comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_rank(union_comm, &rank_union_comm);

    // the root of union comm must be in colcomm!
    // int root  = 0;
    // assert(!(rank_union_comm == root && colcomm == MPI_COMM_NULL));

    // TODO: take care of inactive processes！
    if (colcomm == MPI_COMM_NULL) {
        // todo: use npband, coldims and size of union_comm to find out
        // todo: which processes are inactive, send those targets to rank 0
        // todo: and calculate the ranks, then send it back.
        // * For rank 0, one way is to use ANY_SOURCE and status.MPI_SOURCE
        // int buf[32];
        // MPI_Status status;
        // receive message from any source
        // MPI_recv(buf, 32, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        // int replybuf[];
        // send reply back to sender of the message received above
        // MPI_send(buf, 32, MPI_INT, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
        // * Another way is to precalculate the inactive processes in an array, and loop over it
        
        printf("rank = %d, colcomm is MPI_COMM_NULL, coldims = [%d,%d,%d]!\n",
            rank,coldims[0],coldims[1],coldims[2]);
        // assert(colcomm != MPI_COMM_NULL);
    }


    int *target_ranks_colcomm = malloc(n_targets * sizeof(int));
    assert(target_ranks_colcomm != NULL);
    // first find the ranks of the coords in each column Cart topology
    for (int i = 0; i < n_targets; i++) {
        topo_base_cart_rank(3, coldims, &target_coords[3*i], &target_ranks_colcomm[i]);
    }

    // this is for debugging purpose, can be skipped
    if (colcomm != MPI_COMM_NULL) {
        int *target_ranks_colcomm_ref = malloc(n_targets * sizeof(int));
        assert(target_ranks_colcomm_ref != NULL);
        // ! topo_base_cart_rank should give the same result as MPI_Cart_rank
        // but MPI_Cart_rank requires the calling process to be within colcomm
        for (int i = 0; i < n_targets; i++) {
            MPI_Cart_rank(colcomm, &target_coords[3*i], &target_ranks_colcomm_ref[i]);
        }
        int info = double_check_int_arrays(target_ranks_colcomm_ref, target_ranks_colcomm, n_targets);
        if (info) {
            printf("[FATAL rank_%d]: the topo_base_cart_rank routine is in conflict with the MPI\n"
                   "   in-built MPI_Cart_rank routine! Please contact the developer.\n",rank);
        }

        free(target_ranks_colcomm_ref);
        assert(info == 0);

        int nproc_colcomm;
        MPI_Comm_size(colcomm, &nproc_colcomm);
        assert(nproc_colcomm == coldims[0]*coldims[1]*coldims[2]);
    }

    colcommind_rowcommind_to_union_rank(
        colcomm_sind, colcomm_eind, n_targets,
        rowsize, 1, target_ranks_colcomm, target_ranks
    );

    free(target_ranks_colcomm);
}


void find_ranks_to_send(
    E2D_INFO *E2D_info, int nelem, DDBP_ELEM *elem_list, int *gridsizes, int Ncol,
    int send_ns, int send_ncol, MPI_Comm r_rowcomm, int r_rowsize, MPI_Comm r_colcomm, int *r_coldims,
    int r_colcomm_index, MPI_Comm union_comm)
{
    // first find which columns to send to which colcomm
    // int r_rowsize;
    // assert(r_rowcomm != MPI_COMM_NULL);
    // MPI_Comm_size(r_rowcomm, &r_rowsize);
    int send_ne = send_ns + send_ncol - 1;
    int send_colcomm_sind = block_decompose_rank_BLCYC_fashion(Ncol, r_rowsize, send_ns);
    int send_colcomm_eind = block_decompose_rank_BLCYC_fashion(Ncol, r_rowsize, send_ne);

    int n_send_colcomm = send_colcomm_eind - send_colcomm_sind + 1;

    int *send_nstarts = malloc(n_send_colcomm * sizeof(int));
    int *send_nends = malloc(n_send_colcomm * sizeof(int));
    assert(send_nstarts != NULL && send_nends != NULL);
    for (int n = 0; n < n_send_colcomm; n++) {
        int colcomm_ind = n + send_colcomm_sind;
        int ncol = block_decompose_BLCYC_fashion(Ncol, r_rowsize, colcomm_ind);
        int ns = block_decompose_nstart_BLCYC_fashion(Ncol, r_rowsize, colcomm_ind);
        int ne = ns + ncol - 1;
        // find overlapping columns
        ns = max(send_ns, ns);
        ne = min(send_ne, ne);
        send_nstarts[n] = ns;
        send_nends[n] = ne;
    }

    dyArray target_coords[4], elem_inds;
    init_dyarray(&target_coords[0]); // x coords
    init_dyarray(&target_coords[1]); // y coords
    init_dyarray(&target_coords[2]); // z coords
    init_dyarray(&target_coords[3]); // index correpsonding to (x,y,z), a hashing value
    init_dyarray(&elem_inds);

    int *send_elem_nsend_tot = malloc(nelem * sizeof(int));
    assert(send_elem_nsend_tot != NULL);
    int ***send_elem_verts = malloc(nelem * sizeof(*send_elem_verts));
    assert(send_elem_verts != NULL);
    int ***send_elem_send_coords = malloc(nelem * sizeof(*send_elem_send_coords));
    assert(send_elem_send_coords != NULL);

    // go over all elements and find out the domain it overlaps with
    for (int k = 0; k < nelem; k++) {
        DDBP_ELEM *E_k = &elem_list[k];
        int *elem_verts = E2D_info->elem_verts[k];
        int send_coord_s[3], send_coord_e[3];
        int nsend_tot = 1; // total number of target ranks element k crosses over
        for (int d = 0; d < 3; d++) {
            send_coord_s[d] = block_decompose_rank(gridsizes[d], r_coldims[d], elem_verts[d*2]);
            send_coord_e[d] = block_decompose_rank(gridsizes[d], r_coldims[d], elem_verts[d*2+1]);
            int nsend_d = (send_coord_e[d] - send_coord_s[d] + 1);
            nsend_tot *= nsend_d;
        }
        int *send_coords = (int *)malloc(nsend_tot * 3 * sizeof(int));
        assert(send_coords != NULL);
        // find out all the coordinates of the receiver processes in the recv_comm Cart Topology
        c_ndgrid(3, send_coord_s, send_coord_e, send_coords);
        int *send_coords_dirs[3];
        send_coords_dirs[0] = send_coords;
        send_coords_dirs[1] = send_coords + nsend_tot;
        send_coords_dirs[2] = send_coords + nsend_tot * 2;

        // store the nsend_tot for each element in an array
        send_elem_nsend_tot[k] = nsend_tot;
        send_elem_verts[k] = malloc(nsend_tot * sizeof(*send_elem_verts[k]));
        assert(send_elem_verts[k] != NULL);
        send_elem_send_coords[k] = malloc(nsend_tot * sizeof(*send_elem_send_coords[k]));
        assert(send_elem_send_coords[k] != NULL);
        // find overlapping of the current element and the the receiver domains
        for (int n = 0; n < nsend_tot; n++) {
            int DMnd = 1;
            int DMVert_temp[6], DMnxi[3], coord_temp[3];
            for (int i = 0; i < 3; i++) {
                coord_temp[i] = send_coords_dirs[i][n];
                // local domain in the receiver process
                DMVert_temp[2*i] = block_decompose_nstart(gridsizes[i], r_coldims[i], coord_temp[i]);
                DMnxi[i] = block_decompose(gridsizes[i], r_coldims[i], coord_temp[i]);
                DMVert_temp[2*i+1] = DMVert_temp[2*i] + DMnxi[i] - 1;
                // find intersect of local domains in send process and receiver process
                DMVert_temp[2*i] = max(DMVert_temp[2*i], elem_verts[2*i]);
                DMVert_temp[2*i+1] = min(DMVert_temp[2*i+1], elem_verts[2*i+1]);
                DMnxi[i] = DMVert_temp[2*i+1] - DMVert_temp[2*i] + 1;
                DMnd *= DMnxi[i];
            }

            // store the overlapping DMverts into the array
            send_elem_verts[k][n] = malloc(6 * sizeof(int));
            assert(send_elem_verts[k][n] != NULL);
            for (int i = 0; i < 6; i++) {
                send_elem_verts[k][n][i] = DMVert_temp[i];
            }
            // store the send_coords
            send_elem_send_coords[k][n] = malloc(4 * sizeof(int));
            assert(send_elem_send_coords[k][n] != NULL);
            for (int i = 0; i < 3; i++) {
                send_elem_send_coords[k][n][i] = coord_temp[i];
            }
        }

        // copy the coords into the dynamic array
        for (int i = 0; i < nsend_tot; i++) {
            int coord_i = send_coords_dirs[0][i];
            int coord_j = send_coords_dirs[1][i];
            int coord_k = send_coords_dirs[2][i];
            int index = coord_to_index(r_coldims, coord_i, coord_j, coord_k);
            // store the coordinate index (hashing value)
            send_elem_send_coords[k][i][3] = index;
            append_dyarray(&target_coords[0], coord_i); // x coord
            append_dyarray(&target_coords[1], coord_j); // y coord
            append_dyarray(&target_coords[2], coord_k); // z coord
            append_dyarray(&target_coords[3], index); // index
            append_dyarray(&elem_inds, k); // note that here we store the local element index
        }
        free(send_coords);
    }

    // find unique target coordinates to send to
    int len = target_coords[3].len;
    int *target_unique_index = malloc(len * sizeof(int));
    assert(target_unique_index != NULL);
    memcpy(target_unique_index, target_coords[3].array, len * sizeof(int));

    // find unique indices (hashing values)
    int n_unique_index = unique(target_unique_index, len);

    // for each unique index, find which how many/which elements are mapped to it
    dyArray *elem_mapped_to_coord_index = malloc(n_unique_index * sizeof(dyArray));
    assert(elem_mapped_to_coord_index != NULL);
    for (int i = 0; i < n_unique_index; i++) {
        dyArray *elem_inds_i = &elem_mapped_to_coord_index[i];
        init_dyarray(elem_inds_i);
        int unique_ind = target_unique_index[i];
        for (int j = 0; j < len; j++) {
            int coord_index_temp = target_coords[3].array[j];
            if (coord_index_temp == unique_ind) {
                int k_temp = elem_inds.array[j];
                append_dyarray(elem_inds_i, k_temp);
            }
        }
    }

    // recover the unique coordinates from the indices
    int *target_unique_coords = malloc(3 * n_unique_index * sizeof(int));
    assert(target_unique_coords != NULL);
    for (int i = 0; i < n_unique_index; i++) {
        index_to_coord(r_coldims, target_unique_index[i],
            &target_unique_coords[3*i], &target_unique_coords[3*i+1],
            &target_unique_coords[3*i+2]
        );
    }

    // find unique target ranks in the union_comm
    int send_colcomm_nind = send_colcomm_eind - send_colcomm_sind + 1;
    int nproc_to_send = send_colcomm_nind * n_unique_index;
    int *target_ranks = malloc(nproc_to_send * sizeof(int));
    assert(target_ranks != NULL);
    colcommind_coords_to_ranks(send_colcomm_sind, send_colcomm_eind,
        n_unique_index, target_unique_coords, target_ranks,
        r_rowcomm, r_rowsize, r_colcomm, r_coldims, r_colcomm_index,
        union_comm
    );

    E2D_info->nproc_to_send = nproc_to_send;
    E2D_info->ranks_to_send = target_ranks;

    // Find:
    // E2D_info->send_nstarts      :x
    // E2D_info->send_nends        :x
    // E2D_info->send_nelems       :x
    // E2D_info->send_elem_inds    :x
    // E2D_info->send_elem_verts   :x
    // E2D_info->send_elem_displs  :x
    // E2D_info->sendcounts        :x
    // E2D_info->sdispls           :x
    E2D_info->send_nstarts = malloc(nproc_to_send * sizeof(int));
    assert(E2D_info->send_nstarts != NULL);
    E2D_info->send_nends = malloc(nproc_to_send * sizeof(int));
    assert(E2D_info->send_nends != NULL);
    E2D_info->send_nelems = malloc(nproc_to_send * sizeof(int));
    assert(E2D_info->send_nelems != NULL);
    E2D_info->send_elem_inds = malloc(nproc_to_send * sizeof(int*));
    assert(E2D_info->send_elem_inds != NULL);
    E2D_info->send_elem_verts = malloc(nproc_to_send * sizeof(int**));
    assert(E2D_info->send_elem_verts != NULL);
    E2D_info->send_elem_displs = malloc(nproc_to_send * sizeof(int*));
    assert(E2D_info->send_elem_displs != NULL);
    E2D_info->sendcounts = malloc(nproc_to_send * sizeof(int));
    assert(E2D_info->sendcounts != NULL);
    E2D_info->sdispls = malloc((nproc_to_send+1) * sizeof(int));
    assert(E2D_info->sdispls != NULL);
    E2D_info->sdispls[0] = 0;
    int p_index = 0;
    for (int n = 0; n < n_send_colcomm; n++) {
        for (int i = 0; i < n_unique_index; i++) {
            int p = E2D_info->ranks_to_send[p_index]; // rank to send (global)
            int colcomm_ind = n + send_colcomm_sind; // colcomm index (global)
            int coord_ind = target_unique_index[i]; // coord index (hashing val for coord)
            E2D_info->send_nstarts[p_index] = send_nstarts[n];
            E2D_info->send_nends[p_index] = send_nends[n];
            int send_ncol = send_nends[n] - send_nstarts[n] + 1;
            // nelem that overlap with local domain for send rank p
            int n_overlap_elem = elem_mapped_to_coord_index[i].len;
            E2D_info->send_nelems[p_index] = n_overlap_elem;
            E2D_info->send_elem_inds[p_index] = malloc(n_overlap_elem * sizeof(int));
            assert(E2D_info->send_elem_inds[p_index] != NULL);
            E2D_info->send_elem_verts[p_index] = malloc(n_overlap_elem * sizeof(int*));
            assert(E2D_info->send_elem_verts[p_index] != NULL);

            // int rank;
            // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            // printf("rank = %d, send to rank %d, n_overlap_elem = %d\n",rank, p, n_overlap_elem);

            for (int k_i = 0; k_i < n_overlap_elem; k_i++) {
                int overlap_elem_ind = elem_mapped_to_coord_index[i].array[k_i];
                E2D_info->send_elem_inds[p_index][k_i] = overlap_elem_ind;
                E2D_info->send_elem_verts[p_index][k_i] = malloc(6 * sizeof(int));
                assert(E2D_info->send_elem_verts[p_index][k_i] != NULL);
                // we have stored the coord_inds each element has to send to in send_elem_send_coords[3]
                // now find which coord index corresponds to the current send rank p
                int n_coord_send_k = -1;
                for (int nn = 0; nn < send_elem_nsend_tot[overlap_elem_ind]; nn++) {
                    if (send_elem_send_coords[overlap_elem_ind][nn][3] == coord_ind) {
                        n_coord_send_k = nn;
                        break;
                    }
                }
                assert(n_coord_send_k >= 0);
                for (int ii = 0; ii < 6; ii++) {
                    E2D_info->send_elem_verts[p_index][k_i][ii] =
                        send_elem_verts[overlap_elem_ind][n_coord_send_k][ii];
                }
            }

            E2D_info->send_elem_displs[p_index] = malloc(n_overlap_elem * sizeof(int));
            assert(E2D_info->send_elem_displs[p_index] != NULL);
            E2D_info->send_elem_displs[p_index][0] = 0;
            int sendcount = 0; // total #value to send to rank p
            for (int k_i = 0; k_i < n_overlap_elem; k_i++) {
                int *vert = E2D_info->send_elem_verts[p_index][k_i];
                int nx_k = vert[1] - vert[0] + 1;
                int ny_k = vert[3] - vert[2] + 1;
                int nz_k = vert[5] - vert[4] + 1;
                int nd_k = nx_k * ny_k * nz_k;
                if (k_i+1 < n_overlap_elem) {
                    E2D_info->send_elem_displs[p_index][k_i+1] =
                        E2D_info->send_elem_displs[p_index][k_i] + nd_k * send_ncol;
                }
                sendcount += nd_k * send_ncol;
                // int rank;
                // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                // if (nd_k <= 0 || 1) {
                //     int overlap_elem_ind = elem_mapped_to_coord_index[i].array[k_i];
                //     printf("rank = %d, send to rank %d, overlap_elem_ind = %d, vert_k = [%d,%d,%d,%d,%d,%d]\n",
                //         rank, p, overlap_elem_ind, vert[0], vert[1], vert[2], vert[3], vert[4], vert[5]);
                // }
            }
            E2D_info->sendcounts[p_index] = sendcount;
            E2D_info->sdispls[p_index+1] = E2D_info->sdispls[p_index] + sendcount;
            p_index++; // rank to send local index
        }
    }

    free(target_unique_index);
    free(target_unique_coords);

    // TODO: remove the following lines after implementation is done!
    // free(target_ranks);
    delete_dyarray(&target_coords[0]); // x coords
    delete_dyarray(&target_coords[1]); // y coords
    delete_dyarray(&target_coords[2]); // z coords
    delete_dyarray(&target_coords[3]); // z coords
    delete_dyarray(&elem_inds);
    for (int k = 0; k < nelem; k++) {
        int send_tot = send_elem_nsend_tot[k];
        for (int n = 0; n < send_tot; n++) {
            free(send_elem_verts[k][n]);
            free(send_elem_send_coords[k][n]);
        }
        free(send_elem_verts[k]);
        free(send_elem_send_coords[k]);
    }
    free(send_elem_verts);
    free(send_elem_send_coords);
    free(send_elem_nsend_tot);
    free(send_nstarts);
    free(send_nends);

    for (int i = 0; i < n_unique_index; i++) {
        dyArray *elem_inds_i = &elem_mapped_to_coord_index[i];
        delete_dyarray(elem_inds_i);
    }
    free(elem_mapped_to_coord_index);
}



void find_ranks_to_recv(
    E2D_INFO *E2D_info, int *Edims, int *gridsizes, int *BCs, int Ncol,
    MPI_Comm s_rowcomm, int s_rowsize, MPI_Comm s_colcomm, int s_colsize, int recv_ns, int recv_ncol,
    int *recv_DMVerts
)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // first find which columns to recv from which colcomm
    // int s_rowsize;
    // assert(s_rowcomm != MPI_COMM_NULL);
    // MPI_Comm_size(s_rowcomm, &s_rowsize);

    int recv_ne = recv_ns + recv_ncol - 1;
    int recv_colcomm_sind = block_decompose_rank_BLCYC_fashion(Ncol, s_rowsize, recv_ns);
    int recv_colcomm_eind = block_decompose_rank_BLCYC_fashion(Ncol, s_rowsize, recv_ne);

    int n_recv_colcomm = recv_colcomm_eind - recv_colcomm_sind + 1;
    int *recv_nstarts = malloc(n_recv_colcomm * sizeof(int));
    int *recv_nends = malloc(n_recv_colcomm * sizeof(int));
    assert(recv_nstarts != NULL && recv_nends != NULL);
    for (int n = 0; n < n_recv_colcomm; n++) {
        int colcomm_ind = n + recv_colcomm_sind;
        int ncol = block_decompose_BLCYC_fashion(Ncol, s_rowsize, colcomm_ind);
        int ns = block_decompose_nstart_BLCYC_fashion(Ncol, s_rowsize, colcomm_ind);
        int ne = ns + ncol - 1;
        // find overlapping columns
        ns = max(recv_ns, ns);
        ne = min(recv_ne, ne);
        recv_nstarts[n] = ns;
        recv_nends[n] = ne;
    }

    int nrecv_elem_tot = 1;
    int recv_elem_s[3], recv_elem_e[3];
    // find in each dimension, how many elements the local domain spans over
    for (int d = 0; d < 3; d++) {
        recv_elem_s[d] = element_decompose_rank(gridsizes[d], Edims[d], BCs[d], recv_DMVerts[d*2]);
        recv_elem_e[d] = element_decompose_rank(gridsizes[d], Edims[d], BCs[d], recv_DMVerts[d*2+1]);
        int nrecv_elem_d = (recv_elem_e[d] - recv_elem_s[d] + 1);
        nrecv_elem_tot *= nrecv_elem_d;
    }

#ifdef DEBUG_RANKS_TO_RECV
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("rank = %d, nrecv_elem_tot = %d\n",rank,nrecv_elem_tot);
    if (rank == 0) {
        printf("rank = %d, recv_elem_s[3] = [%d, %d, %d], recv_elem_e[3] = [%d, %d, %d]\n",
            rank, recv_elem_s[0], recv_elem_s[1], recv_elem_s[2],
            recv_elem_e[0], recv_elem_e[1], recv_elem_e[2]);
    }
#endif

    int *recv_elems = (int *)malloc(nrecv_elem_tot * 3 * sizeof(int));
    assert(recv_elems != NULL);
    // find out all the coordinates of the receiver processes in the recv_comm Cart Topology
    c_ndgrid(3, recv_elem_s, recv_elem_e, recv_elems);
    int *recv_elems_ivec = recv_elems;
    int *recv_elems_jvec = recv_elems + nrecv_elem_tot;
    int *recv_elems_kvec = recv_elems + nrecv_elem_tot * 2;


    int *recv_elemcomm_indices = malloc(nrecv_elem_tot * sizeof(int));
    assert(recv_elemcomm_indices != NULL);

    int *recv_elems_indices = malloc(nrecv_elem_tot * sizeof(int));
    assert(recv_elems_indices != NULL);

    // find out which process contains these elements
    // find which rank owns this element in bandcomm
    // int s_colsize;
    // assert(s_colcomm != MPI_COMM_NULL);
    // MPI_Comm_size(s_colcomm, &s_colsize);
    int npelem = s_colsize; // TODO: verify if this is correct!
    int Ne_tot = Edims[0] * Edims[1] * Edims[2];
    for (int i = 0; i < nrecv_elem_tot; i++) {
        int coord_temp[3];
        coord_temp[0] = recv_elems_ivec[i];
        coord_temp[1] = recv_elems_jvec[i];
        coord_temp[2] = recv_elems_kvec[i];
        int elem_index;
        DDBP_Cart_Index(Edims, coord_temp, &elem_index);
        int owner_elemcomm_index = element_index_to_elemcomm_index(Ne_tot, npelem, elem_index);
        recv_elemcomm_indices[i] = owner_elemcomm_index;
        recv_elems_indices[i] = elem_index;
    }

#ifdef DEBUG_RANKS_TO_RECV
    //TODO: remove after check
    int nproc_wrldcomm;
    int rank_wrldcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_wrldcomm);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc_wrldcomm);
    for (int i = 0; i < nproc_wrldcomm; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == rank) {
            // printf("rank = %d, recv_elemcomm_indices [n = %d]: ", rank_wrldcomm, nrecv_elem_tot);
            // print_array(recv_elemcomm_indices, nrecv_elem_tot, sizeof(int));
            printf("rank = %d, recv_elems_indices [n = %d]: ", rank_wrldcomm, nrecv_elem_tot);
            print_array(recv_elems_indices, nrecv_elem_tot, sizeof(int));
        }
        // usleep(100000);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // find unique elemcomm_index
    int *recv_unique_elemcomm_indices = malloc(nrecv_elem_tot * sizeof(int));
    assert(recv_unique_elemcomm_indices != NULL);
    memcpy(recv_unique_elemcomm_indices, recv_elemcomm_indices, nrecv_elem_tot * sizeof(int));
    int n_unique_elemcomm_index = unique(recv_unique_elemcomm_indices, nrecv_elem_tot);

    // for each unique index, find which how many/which elements are mapped to it
    dyArray *elem_mapped_to_elemcomm_index = malloc(n_unique_elemcomm_index * sizeof(dyArray));
    assert(elem_mapped_to_elemcomm_index != NULL);
    for (int i = 0; i < n_unique_elemcomm_index; i++) {
        dyArray *elem_inds_i = &elem_mapped_to_elemcomm_index[i];
        init_dyarray(elem_inds_i);
        int unique_ind = recv_unique_elemcomm_indices[i];
        for (int j = 0; j < nrecv_elem_tot; j++) {
            int elemcomm_index_temp = recv_elemcomm_indices[j];
            if (elemcomm_index_temp == unique_ind) {
                int k_temp = recv_elems_indices[j];
                append_dyarray(elem_inds_i, k_temp);
            }
        }
    }

    // find unique target ranks in the union_comm
    int nproc_to_recv = n_recv_colcomm * n_unique_elemcomm_index;
    int *target_ranks = malloc(nproc_to_recv * sizeof(int));
    assert(target_ranks != NULL);

    colcommind_rowcommind_to_union_rank(
        recv_colcomm_sind, recv_colcomm_eind, n_unique_elemcomm_index,
        s_rowsize, 1, recv_unique_elemcomm_indices, target_ranks
    );

    E2D_info->nproc_to_recv = nproc_to_recv;
    E2D_info->ranks_to_recv = target_ranks;

    // Find:
    // E2D_info->recv_nstarts      :x
    // E2D_info->recv_nends        :x
    // E2D_info->recv_nelems       :x
    // E2D_info->recv_elem_inds    :x (not used)
    // E2D_info->recv_elem_verts   :x
    // E2D_info->recv_elem_displs  :x
    // E2D_info->recvcounts        :x
    // E2D_info->rdispls           :x
    E2D_info->recv_nstarts = malloc(nproc_to_recv * sizeof(int));
    assert(E2D_info->recv_nstarts != NULL);
    E2D_info->recv_nends = malloc(nproc_to_recv * sizeof(int));
    assert(E2D_info->recv_nends != NULL);
    E2D_info->recv_nelems = malloc(nproc_to_recv * sizeof(int));
    assert(E2D_info->recv_nelems != NULL);
    E2D_info->recv_elem_inds = malloc(nproc_to_recv * sizeof(int*));
    assert(E2D_info->recv_elem_inds != NULL);
    E2D_info->recv_elem_verts = malloc(nproc_to_recv * sizeof(int**));
    assert(E2D_info->recv_elem_verts != NULL);
    E2D_info->recv_elem_displs = malloc(nproc_to_recv * sizeof(int*));
    assert(E2D_info->recv_elem_displs != NULL);
    E2D_info->recvcounts = malloc(nproc_to_recv * sizeof(int));
    assert(E2D_info->recvcounts != NULL);
    E2D_info->rdispls = malloc((nproc_to_recv+1) * sizeof(int));
    assert(E2D_info->rdispls != NULL);
    E2D_info->rdispls[0] = 0;
    int p_index = 0;
    for (int n = 0; n < n_recv_colcomm; n++) {
        int p = E2D_info->ranks_to_recv[p_index];
        for (int i = 0; i < n_unique_elemcomm_index; i++) {
            int p = E2D_info->ranks_to_recv[p_index]; // rank to recv (global)
            int colcomm_ind = n + recv_colcomm_sind; // colcomm index (global)
            int elemcomm_index = recv_unique_elemcomm_indices[i];
            E2D_info->recv_nstarts[p_index] = recv_nstarts[n];
            E2D_info->recv_nends[p_index] = recv_nends[n];
            int recv_ncol = recv_nends[n] - recv_nstarts[n] + 1;

            // nelem that overlap with local domain from recv rank p
            int n_overlap_elem = elem_mapped_to_elemcomm_index[i].len;
            E2D_info->recv_nelems[p_index] = n_overlap_elem;
            E2D_info->recv_elem_inds[p_index] = malloc(n_overlap_elem * sizeof(int));
            assert(E2D_info->recv_elem_inds[p_index] != NULL);
            E2D_info->recv_elem_verts[p_index] = malloc(n_overlap_elem * sizeof(int*));
            assert(E2D_info->recv_elem_verts[p_index] != NULL);

            for (int k_i = 0; k_i < n_overlap_elem; k_i++) {
                int overlap_elem_ind = elem_mapped_to_elemcomm_index[i].array[k_i]; // global
                E2D_info->recv_elem_inds[p_index][k_i] = overlap_elem_ind;
                E2D_info->recv_elem_verts[p_index][k_i] = malloc(6 * sizeof(int));
                assert(E2D_info->recv_elem_verts[p_index][k_i] != NULL);
                int coord_temp[3], elem_verts_temp[6];
                DDBP_Index_Cart(Edims, overlap_elem_ind, coord_temp);
                for (int d = 0; d < 3; d++) {
                    elem_verts_temp[2*d] = element_decompose_nstart(gridsizes[d], Edims[d], BCs[d], coord_temp[d]);
                    elem_verts_temp[2*d+1] = element_decompose_nend(gridsizes[d], Edims[d], BCs[d], coord_temp[d]);
                    // find intersect of the element and the local domain
                    elem_verts_temp[2*d] = max(recv_DMVerts[2*d], elem_verts_temp[2*d]);
                    elem_verts_temp[2*d+1] = min(recv_DMVerts[2*d+1], elem_verts_temp[2*d+1]);
                }
                for (int ii = 0; ii < 6; ii++) {
                    assert(elem_verts_temp[ii] >= 0 && elem_verts_temp[ii] < gridsizes[ii/2]);
                    E2D_info->recv_elem_verts[p_index][k_i][ii] = elem_verts_temp[ii];
                }
            }

            E2D_info->recv_elem_displs[p_index] = malloc(n_overlap_elem * sizeof(int));
            assert(E2D_info->recv_elem_displs[p_index] != NULL);
            E2D_info->recv_elem_displs[p_index][0] = 0;
            int recvcount = 0; // total #value to recv to rank p
            for (int k_i = 0; k_i < n_overlap_elem; k_i++) {
                int *vert = E2D_info->recv_elem_verts[p_index][k_i];
                int nx_k = vert[1] - vert[0] + 1;
                int ny_k = vert[3] - vert[2] + 1;
                int nz_k = vert[5] - vert[4] + 1;
                int nd_k = nx_k * ny_k * nz_k;
                if (k_i+1 < n_overlap_elem) {
                    E2D_info->recv_elem_displs[p_index][k_i+1] =
                        E2D_info->recv_elem_displs[p_index][k_i] + nd_k * recv_ncol;
                }
                recvcount += nd_k * recv_ncol;
                // int rank;
                // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                // if (nd_k <= 0 || 1) {
                //     int overlap_elem_ind = elem_mapped_to_elemcomm_index[i].array[k_i];
                //     printf("rank = %d, recv from rank %d, overlap_elem_ind = %d, vert_k = [%d,%d,%d,%d,%d,%d]\n",
                //         rank, p, overlap_elem_ind, vert[0], vert[1], vert[2], vert[3], vert[4], vert[5]);
                // }
            }
            E2D_info->recvcounts[p_index] = recvcount;
            E2D_info->rdispls[p_index+1] = E2D_info->rdispls[p_index] + recvcount;

            p_index++;
        }
    }

    free(recv_nstarts);
    free(recv_nends);
    free(recv_elems);
    free(recv_elems_indices);
    free(recv_elemcomm_indices);
    free(recv_unique_elemcomm_indices);
    
    for (int i = 0; i < n_unique_elemcomm_index; i++) {
        dyArray *elem_inds_i = &elem_mapped_to_elemcomm_index[i];
        delete_dyarray(elem_inds_i);
    }
    free(elem_mapped_to_elemcomm_index);
}



/**
 * @brief Initialize data structure for element parallelization to
 *        domain parallelization data transfer.
 */
void E2D_Init(E2D_INFO *E2D_info, int *Edims, int nelem, DDBP_ELEM *elem_list,
    int *gridsizes, int *BCs, int Ncol,
    int send_ns, int send_ncol, MPI_Comm s_rowcomm, int s_rowsize, int s_rowcomm_index,
    MPI_Comm s_colcomm, int s_colsize, int s_colcomm_index,
    int recv_ns, int recv_ncol, int *recv_DMVerts, MPI_Comm r_rowcomm, int r_rowsize, MPI_Comm r_colcomm,
    int *r_coldims, int r_colcomm_index,
    MPI_Comm union_comm)
{
    int is_sender = (s_colcomm_index < 0 || s_rowcomm_index < 0 || send_ncol <= 0) ? 0 : 1;
    int is_recver = (r_colcomm_index < 0 || r_colcomm == MPI_COMM_NULL || recv_ncol <= 0) ? 0 : 1;

    E2D_info->is_sender = is_sender;
    E2D_info->is_recver = is_recver;
    E2D_info->union_comm = union_comm;
    E2D_info->sendtype = MPI_DOUBLE;
    E2D_info->recvtype = MPI_DOUBLE;

    // global data sizes
    E2D_info->gridsizes[0] = gridsizes[0];
    E2D_info->gridsizes[1] = gridsizes[1];
    E2D_info->gridsizes[2] = gridsizes[2];
    E2D_info->Ncol = Ncol; // global #columns

    // set up data size parameters for senders
    E2D_info->nelem = nelem;
    if (is_sender) {
        // element vertices owned by the sender
        // first allocate memory
        E2D_info->elem_verts = malloc(nelem * sizeof(int *));
        assert(E2D_info->elem_verts != NULL);
        for (int k = 0; k < nelem; k++) {
            E2D_info->elem_verts[k] = malloc(6 * sizeof(int));
            assert(E2D_info->elem_verts[k] != NULL);
        }
        // set up the vertices (global index)
        for (int k = 0; k < nelem; k++) {
            DDBP_ELEM *E_k = &elem_list[k];
            E2D_info->elem_verts[k][0] = E_k->is;
            E2D_info->elem_verts[k][1] = E_k->ie;
            E2D_info->elem_verts[k][2] = E_k->js;
            E2D_info->elem_verts[k][3] = E_k->je;
            E2D_info->elem_verts[k][4] = E_k->ks;
            E2D_info->elem_verts[k][5] = E_k->ke;
        }
        // element band index
        E2D_info->elem_band_nstart = send_ns;
        E2D_info->elem_band_nend = send_ns + send_ncol - 1;
    }

    // set up data size parameters for receivers
    if (is_recver) {
        E2D_info->dm_band_nstart = recv_ns;
        E2D_info->dm_band_nend = recv_ns + recv_ncol - 1;
        E2D_info->dm_vert[0] = recv_DMVerts[0];
        E2D_info->dm_vert[1] = recv_DMVerts[1];
        E2D_info->dm_vert[2] = recv_DMVerts[2];
        E2D_info->dm_vert[3] = recv_DMVerts[3];
        E2D_info->dm_vert[4] = recv_DMVerts[4];
        E2D_info->dm_vert[5] = recv_DMVerts[5];
    }

    // set up target ranks and displacement of data in the buffer
    if (is_sender) {
        find_ranks_to_send(
            E2D_info, nelem, elem_list, gridsizes, Ncol,
            send_ns, send_ncol, r_rowcomm, r_rowsize, r_colcomm, r_coldims,
            r_colcomm_index, union_comm
        );
        E2D_info->send_requests = malloc(E2D_info->nproc_to_send * sizeof(MPI_Request));
        assert(E2D_info->send_requests != NULL);
    } else {
        E2D_info->nproc_to_send = 0;
        E2D_info->elem_band_nstart = 0;
        E2D_info->elem_band_nend = -1;
    }

    // set up receiver ranks and displacement of data in the buffer
    if (is_recver) {
        find_ranks_to_recv(E2D_info, Edims, gridsizes, BCs, Ncol, s_rowcomm,
            s_rowsize, s_colcomm, s_colsize, recv_ns, recv_ncol, recv_DMVerts);
        E2D_info->recv_requests = malloc(E2D_info->nproc_to_recv * sizeof(MPI_Request));
        assert(E2D_info->recv_requests != NULL);
    } else {
        E2D_info->nproc_to_recv = 0;
        E2D_info->dm_band_nstart = 0;
        E2D_info->dm_band_nend = -1;
    }
}

// free E2D_info
void E2D_Finalize(E2D_INFO *E2D_info)
{
    int is_sender = E2D_info->is_sender;
    int is_recver = E2D_info->is_recver;

    if (is_sender) {
        // m-d arrays
        int nelem = E2D_info->nelem;
        for (int k = 0; k < nelem; k++) {
            free(E2D_info->elem_verts[k]);
        }
        free(E2D_info->elem_verts);

        for (int p = 0; p < E2D_info->nproc_to_send; p++) {
            for (int k = 0; k < E2D_info->send_nelems[p]; k++) {
                free(E2D_info->send_elem_verts[p][k]);
            }
            free(E2D_info->send_elem_inds[p]);
            free(E2D_info->send_elem_displs[p]);
            free(E2D_info->send_elem_verts[p]);
        }
        free(E2D_info->send_elem_inds);
        free(E2D_info->send_elem_displs);
        free(E2D_info->send_elem_verts);

        // 1-d arrays
        free(E2D_info->ranks_to_send);
        free(E2D_info->sendcounts);
        free(E2D_info->sdispls);
        free(E2D_info->send_nstarts);
        free(E2D_info->send_nends);
        free(E2D_info->send_nelems);
        free(E2D_info->send_requests);
    }

    if (is_recver) {
        // m-d arrays
        for (int p = 0; p < E2D_info->nproc_to_recv; p++) {
            for (int k = 0; k < E2D_info->recv_nelems[p]; k++) {
                free(E2D_info->recv_elem_verts[p][k]);
            }
            free(E2D_info->recv_elem_inds[p]);
            free(E2D_info->recv_elem_displs[p]);
            free(E2D_info->recv_elem_verts[p]);
        }
        free(E2D_info->recv_elem_inds);
        free(E2D_info->recv_elem_displs);
        free(E2D_info->recv_elem_verts);
        // 1-d arrays
        free(E2D_info->ranks_to_recv);
        free(E2D_info->recvcounts);
        free(E2D_info->rdispls);
        free(E2D_info->recv_nstarts);
        free(E2D_info->recv_nends);
        free(E2D_info->recv_nelems);
        free(E2D_info->recv_requests);
    }
}


// set up sendbuf, pack data from sdata into sendbuf
void E2D_set_sendbuf(E2D_INFO *E2D_info, const void **sdata, void *sendbuf)
{
    MPI_Aint lower_bound;
    MPI_Aint extent_sendtype;
    MPI_Type_get_extent(E2D_info->sendtype, &lower_bound, &extent_sendtype);

    // go over each target rank and pack up the data into sendbuf
    for (int p = 0; p < E2D_info->nproc_to_send; p++) {
        // int sendcount_p = E2D_info->sendcounts[p];
        void *sendbuf_p = E2D_info->sendbuf + E2D_info->sdispls[p] * extent_sendtype;
        int nelem = E2D_info->send_nelems[p]; // create a var in E2D_info for each rank
        for (int k = 0; k < nelem; k++) {
            // send elem index
            int k_pk = E2D_info->send_elem_inds[p][k]; // local element index

            // domain global vertices for element k_pk (whole element)
            int *elem_verts = E2D_info->elem_verts[k_pk];
            int DMnx = elem_verts[1] - elem_verts[0] + 1;
            int DMny = elem_verts[3] - elem_verts[2] + 1;
            int DMnz = elem_verts[5] - elem_verts[4] + 1;
            int DMnd = DMnx * DMny * DMnz;

            // send elem domain global indices (overlapping part)
            int *send_verts_pk = E2D_info->send_elem_verts[p][k];
            int nx_pk = send_verts_pk[1] - send_verts_pk[0] + 1;
            int ny_pk = send_verts_pk[3] - send_verts_pk[2] + 1;
            int nz_pk = send_verts_pk[5] - send_verts_pk[4] + 1;
            int nd_pk = nx_pk * ny_pk * nz_pk;

            int send_displ_k = E2D_info->send_elem_displs[p][k];
            void *sendbuf_pk = sendbuf_p + send_displ_k * extent_sendtype;

            // sdata domain relative indices (overlapping part wrt current element)
            int is_r = send_verts_pk[0] - elem_verts[0];
            int ie_r = send_verts_pk[1] - elem_verts[0];
            int js_r = send_verts_pk[2] - elem_verts[2];
            int je_r = send_verts_pk[3] - elem_verts[2];
            int ks_r = send_verts_pk[4] - elem_verts[4];
            int ke_r = send_verts_pk[5] - elem_verts[4];

            // TODO: set up for multiple columns!
            // sendbuf_pk column info
            int send_ns = E2D_info->send_nstarts[p]; // global send column start index
            int send_ne = E2D_info->send_nends[p]; // global send column end index
            // copy data from sdata_k into sendbuf_pk
            const void *sdata_k = sdata[k_pk]; // TODO: check if casting is required!
            for (int n = send_ns; n <= send_ne; n++) {
                int n_send = n - send_ns;
                int n_r = n - E2D_info->elem_band_nstart;
                // TODO: remove after test?
                assert(n_r >= 0 && n_r <= E2D_info->elem_band_nend - E2D_info->elem_band_nstart);
                // ! at this point, we have to assume the array is a "double array"
                // TODO: generalize this routine for different data types
                extract_subgrid(
                    (double*)sdata_k + n_r * DMnd, DMnx, DMny, DMnz,
                    is_r, ie_r, js_r, je_r, ks_r, ke_r,
                    (double*)sendbuf_pk + n_send * nd_pk, nx_pk, ny_pk, nz_pk,
                    0, nx_pk-1, 0, ny_pk-1, 0, nz_pk-1
                );
            }
        }
    }
}


// start to transfer element distribution to domain distribution, non-blocking
// * set up sendbuf and recvbuf
// * initialize the MPI_Isend and MPI_Irecv
void E2D_Iexec(E2D_INFO *E2D_info, const void **sdata)
{
    int is_sender = E2D_info->is_sender;
    int is_recver = E2D_info->is_recver;
    int nproc_to_send = E2D_info->nproc_to_send;
    int nproc_to_recv = E2D_info->nproc_to_recv;
    MPI_Comm comm = E2D_info->union_comm;
    MPI_Aint lower_bound;
    MPI_Aint extent_sendtype, extent_recvtype;
    MPI_Type_get_extent(E2D_info->sendtype, &lower_bound, &extent_sendtype);
    MPI_Type_get_extent(E2D_info->recvtype, &lower_bound, &extent_recvtype);

    // set up recvbuf and init MPI_Irecv
    if (is_recver) {
        // set up recvbuf (allocate memory)
        int recvcount_tot = E2D_info->rdispls[nproc_to_recv];
        E2D_info->recvbuf = malloc(recvcount_tot * extent_recvtype);
        assert(E2D_info->recvbuf != NULL);
        // start to receive (non-blocking MPI_Irecv)
        for (int i = 0; i < nproc_to_recv; i++) {
            int r = E2D_info->ranks_to_recv[i];
            MPI_Irecv(E2D_info->recvbuf + E2D_info->rdispls[i] * extent_recvtype,
                E2D_info->recvcounts[i], E2D_info->recvtype, r, 123,
                E2D_info->union_comm, &E2D_info->recv_requests[i]);
        }
    }

    // set up sendbuf and init MPI_Isend
    if (is_sender) {
        int sendcount_tot = E2D_info->sdispls[nproc_to_send];
        E2D_info->sendbuf = malloc(sendcount_tot * extent_sendtype);
        assert(E2D_info->sendbuf != NULL);
        E2D_set_sendbuf(E2D_info, sdata, E2D_info->sendbuf);
        // start to send (non-blocking MPI_Isend)
        for (int i = 0; i < nproc_to_send; i++) {
            int r = E2D_info->ranks_to_send[i];
            MPI_Isend(E2D_info->sendbuf + E2D_info->sdispls[i] * extent_sendtype,
                E2D_info->sendcounts[i], E2D_info->sendtype, r, 123,
                E2D_info->union_comm, &E2D_info->send_requests[i]);
        }
    }
}

// TODO: generalize this function for different datatypes
// once receive is completed, copy data from recvbuf to rdata
void E2D_set_rdata(E2D_INFO *E2D_info, void *rdata)
{
    MPI_Aint lower_bound;
    MPI_Aint extent_recvtype;
    MPI_Type_get_extent(E2D_info->recvtype, &lower_bound, &extent_recvtype);

    int DMnx = E2D_info->dm_vert[1] - E2D_info->dm_vert[0] + 1;
    int DMny = E2D_info->dm_vert[3] - E2D_info->dm_vert[2] + 1;
    int DMnz = E2D_info->dm_vert[5] - E2D_info->dm_vert[4] + 1;
    int DMnd = DMnx * DMny * DMnz;
    // go over each trunk of data from different ranks
    for (int p = 0; p < E2D_info->nproc_to_recv; p++) {
        // int recvcount_p = E2D_info->recvcounts[p];
        void *recvbuf_p = E2D_info->recvbuf + E2D_info->rdispls[p] * extent_recvtype;
        int nelem = E2D_info->recv_nelems[p]; // create a var in E2D_info for each rank
        for (int k = 0; k < nelem; k++) {
            int recv_displ_k = E2D_info->recv_elem_displs[p][k];
            void *recvbuf_pk = recvbuf_p + recv_displ_k * extent_recvtype;
            // recvbuf_pk domain indices
            int *recv_verts_pk = E2D_info->recv_elem_verts[p][k];
            int nx_pk = recv_verts_pk[1] - recv_verts_pk[0] + 1;
            int ny_pk = recv_verts_pk[3] - recv_verts_pk[2] + 1;
            int nz_pk = recv_verts_pk[5] - recv_verts_pk[4] + 1;
            int nd_pk = nx_pk * ny_pk * nz_pk;
            // rdata domain indices
            int is_r = recv_verts_pk[0] - E2D_info->dm_vert[0];
            int ie_r = recv_verts_pk[1] - E2D_info->dm_vert[0];
            int js_r = recv_verts_pk[2] - E2D_info->dm_vert[2];
            int je_r = recv_verts_pk[3] - E2D_info->dm_vert[2];
            int ks_r = recv_verts_pk[4] - E2D_info->dm_vert[4];
            int ke_r = recv_verts_pk[5] - E2D_info->dm_vert[4];
            // TODO: set up for multiple columns!
            // recvbuf_pk column info
            int recv_ns = E2D_info->recv_nstarts[p]; // global recv column start index
            int recv_ne = E2D_info->recv_nends[p]; // global recv column start index
            for (int n = recv_ns; n <= recv_ne; n++) {
                int n_recv = n - recv_ns;
                int n_r = n - E2D_info->dm_band_nstart;
                // TODO: remove after test?
                assert(n_r >= 0 && n_r <= E2D_info->dm_band_nend - E2D_info->dm_band_nstart);
                // ! at this point, we have to assume the array is a "double array"
                // TODO: generalize this routine for different data types
                extract_subgrid(
                    (double*)recvbuf_pk + n_recv * nd_pk, nx_pk, ny_pk, nz_pk,
                    0, nx_pk-1, 0, ny_pk-1, 0, nz_pk-1,
                    (double*)rdata + n_r * DMnd, DMnx, DMny, DMnz,
                    is_r, ie_r, js_r, je_r, ks_r, ke_r
                );
            }
        }
    }
}


// wait for nonblocking send and receive in E2D to be completed before moving on
void E2D_Wait(E2D_INFO *E2D_info, void *rdata) {
    // wait for sending to be completed
    if (E2D_info->is_sender)
        MPI_Waitall(E2D_info->nproc_to_send, E2D_info->send_requests,
            MPI_STATUS_IGNORE);
    // wait for receiving to be completed
    if (E2D_info->is_recver)
        MPI_Waitall(E2D_info->nproc_to_recv, E2D_info->recv_requests,
            MPI_STATUS_IGNORE);

    // copy the data from recvbuf to rdata
    if (E2D_info->is_recver)
        E2D_set_rdata(E2D_info, rdata);

    // free buffers
    if (E2D_info->is_sender) free(E2D_info->sendbuf);
    if (E2D_info->is_recver) free(E2D_info->recvbuf);
}


// non-blocking
void E2D_Exec(E2D_INFO *E2D_info, const void **sdata, void *rdata)
{
    E2D_Iexec(E2D_info, sdata);
    E2D_Wait(E2D_info, rdata);
}


void E2D(E2D_INFO *E2D_info, int *Edims, int nelem, DDBP_ELEM *elem_list,
    int *gridsizes, int *BCs, int Ncol, const void **sdata, int send_ns,
    int send_ncol, MPI_Comm s_rowcomm, int s_rowsize, int s_rowcomm_index,
    MPI_Comm s_colcomm, int s_colsize, int s_colcomm_index,
    void *rdata, int recv_ns, int recv_ncol, int *recv_DMVerts,
    MPI_Comm r_rowcomm, int r_rowsize, MPI_Comm r_colcomm, int *r_coldims, int r_colcomm_index,
    MPI_Comm union_comm)
{
    // set up parameters such as target ranks and send/recv counts etc.
    E2D_Init(
        E2D_info, Edims, nelem, elem_list, gridsizes, BCs, Ncol,
        send_ns, send_ncol, s_rowcomm, s_rowsize, s_rowcomm_index,
        s_colcomm, s_colsize, s_colcomm_index,
        recv_ns, recv_ncol, recv_DMVerts, r_rowcomm, r_rowsize, r_colcomm,
        r_coldims, r_colcomm_index, union_comm
    );

    // perform data transfer
    E2D_Exec(E2D_info, sdata, rdata);

    // free E2D_info
    E2D_Finalize(E2D_info);
}


/**
 * @brief Set up sub-communicators for DDBP method.
 *
 */
void Setup_Comms_DDBP(SPARC_OBJ *pSPARC) {
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;

    // set up paral. params, if the user doesn't provide them
    set_default_DDBP_paral_params(DDBP_info, pSPARC->kptcomm);

    // split kptcomm into multiple ddbp elemcomm's
    create_DDBP_elemcomm(DDBP_info, pSPARC->kptcomm);

    // split kptcomm into multiple bandcomm's (bridge between elemcomm's)
    create_DDBP_bandcomm(DDBP_info, pSPARC->kptcomm);

    // split elemcomm into multiple ddbp basiscomm's
    create_DDBP_basiscomm(DDBP_info, DDBP_info->elemcomm);
}

