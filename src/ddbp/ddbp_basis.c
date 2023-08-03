/**
 * @file    ddbp_basis.c
 * @brief   This file contains the functions for the Discrete
 *          Discontinuous Basis Projection (DDBP) basis generation.
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
#include "sq3.h"
#include "cs.h"
#include "ddbp.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define TEMP_TOL 1e-14


/**
 * @brief  Set up nonlocal projectors for all DDBP (extended) elements.
 *
 *         This has to be done every time the atom positions are updated.
 */
void setup_ddbp_element_Vnl(SPARC_OBJ *pSPARC) {
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start setting up nloc projs for DDBP elements ... \n");
    #endif

    int rank, nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int BCs[3], BCs_a[3], BCs_v[3];
    BCs_a[0] = BCs_v[0] = BCs[0] = pSPARC->BCx;
    BCs_a[1] = BCs_v[1] = BCs[1] = pSPARC->BCy;
    BCs_a[2] = BCs_v[2] = BCs[2] = pSPARC->BCz;

    double cell[3];
    cell[0] = pSPARC->range_x;
    cell[1] = pSPARC->range_y;
    cell[2] = pSPARC->range_z;

    int isPerfBC_x = (int)(fabs(DDBP_info->buffer_x) <= TEMP_TOL && DDBP_info->Nex == 1);
    int isPerfBC_y = (int)(fabs(DDBP_info->buffer_y) <= TEMP_TOL && DDBP_info->Ney == 1);
    int isPerfBC_z = (int)(fabs(DDBP_info->buffer_z) <= TEMP_TOL && DDBP_info->Nez == 1);

    // Boundary conditions for assigning atoms to extended elements
    // * when there is a buffer, we include images as if they are independent atoms,
    // * if there is no buffer in one direction, we force Dirichlet BC and not include
    // * those images, which will be included later as real images
    if (isPerfBC_x) BCs_a[0] = 1;
    if (isPerfBC_y) BCs_a[1] = 1;
    if (isPerfBC_z) BCs_a[2] = 1;

    // assign atoms to the elements and extended elements
    ddbp_assign_atoms(pSPARC->atom_pos, pSPARC->n_atom, cell, BCs_a, DDBP_info);

    // Boundary conditions for setting up Vnl
    // * force Dirichlet BC if there is a nonzero buffer in that direction, since
    // * those images are included as independent atoms. Whereas in zero buffer direction,
    // * we include global images as real images, integrals will be summed during Vnl*x
    if (!isPerfBC_x) BCs_v[0] = 1;
    if (!isPerfBC_y) BCs_v[1] = 1;
    if (!isPerfBC_z) BCs_v[2] = 1;

    // go over each element and set up nonlocal projectors
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;

        double cell_orig[3]; // cell origin of the global domain
        cell_orig[0] = -E_k->xs_ex_sg;
        cell_orig[1] = -E_k->ys_ex_sg;
        cell_orig[2] = -E_k->zs_ex_sg;

        // For setting up element nonlocal projectors, we temporarily force BC to DBC
        // This is because we already included atoms/images that have nloc influence to the element
        int BCx_tmp = ESPRC_k->BCx;
        int BCy_tmp = ESPRC_k->BCy;
        int BCz_tmp = ESPRC_k->BCz;

        CalculateNonlocalInnerProductIndex(ESPRC_k);

        // set up nonlocal projectors for basis calculation
        // find atoms that have nonlocal influence on the extended element
        // GetInfluencingAtoms_nloc(ESPRC_k, &ESPRC_k->Atom_Influence_nloc, ESPRC_k->DMVertices_dmcomm,
        //                         ESPRC_k->bandcomm_index < 0 ? MPI_COMM_NULL : ESPRC_k->dmcomm);
        GetInfluencingAtoms_nloc_kernel(ESPRC_k, cell_orig, cell, BCs_v, &ESPRC_k->Atom_Influence_nloc,
            ESPRC_k->DMVertices_dmcomm, ESPRC_k->bandcomm_index < 0 ? MPI_COMM_NULL : ESPRC_k->dmcomm);

        // calculate nonlocal projectors in psi-domain
        if (pSPARC->isGammaPoint)
            CalculateNonlocalProjectors(ESPRC_k, &ESPRC_k->nlocProj, ESPRC_k->Atom_Influence_nloc,
                                        ESPRC_k->DMVertices_dmcomm, ESPRC_k->bandcomm_index < 0 ? MPI_COMM_NULL : ESPRC_k->dmcomm);
        else
            CalculateNonlocalProjectors_kpt(ESPRC_k, &ESPRC_k->nlocProj, ESPRC_k->Atom_Influence_nloc,
                                            ESPRC_k->DMVertices_dmcomm, ESPRC_k->bandcomm_index < 0 ? MPI_COMM_NULL : ESPRC_k->dmcomm);

        // set up nonlocal projectors for Lanczos
        // find atoms that have nonlocal influence the process domain (of kptcomm_topo)
        // GetInfluencingAtoms_nloc(ESPRC_k, &ESPRC_k->Atom_Influence_nloc_kptcomm, ESPRC_k->DMVertices_kptcomm,
        //                         ESPRC_k->kptcomm_index < 0 ? MPI_COMM_NULL : ESPRC_k->kptcomm_topo);
        GetInfluencingAtoms_nloc_kernel(ESPRC_k, cell_orig, cell, BCs_v, &ESPRC_k->Atom_Influence_nloc_kptcomm,
            ESPRC_k->DMVertices_kptcomm, ESPRC_k->kptcomm_index < 0 ? MPI_COMM_NULL : ESPRC_k->kptcomm_topo);

        // calculate nonlocal projectors in kptcomm_topo
        if (pSPARC->isGammaPoint)
            CalculateNonlocalProjectors(ESPRC_k, &ESPRC_k->nlocProj_kptcomm, ESPRC_k->Atom_Influence_nloc_kptcomm,
                                        ESPRC_k->DMVertices_kptcomm,
                                        ESPRC_k->kptcomm_index < 0 ? MPI_COMM_NULL : ESPRC_k->kptcomm_topo);
        else
            CalculateNonlocalProjectors_kpt(ESPRC_k, &ESPRC_k->nlocProj_kptcomm, ESPRC_k->Atom_Influence_nloc_kptcomm,
                                            ESPRC_k->DMVertices_kptcomm,
                                            ESPRC_k->kptcomm_index < 0 ? MPI_COMM_NULL : ESPRC_k->kptcomm_topo);

        // recover original BC
        ESPRC_k->BCx = BCx_tmp;
        ESPRC_k->BCy = BCy_tmp;
        ESPRC_k->BCz = BCz_tmp;

        // set up nonlocal projectors for global Vnl restricted on the element
        // *for H_DDBP construction (Vnl_DDBP part)
        int DMVert_E_k[6];
        DMVert_E_k[0] = E_k->is; DMVert_E_k[1] = E_k->ie;
        DMVert_E_k[2] = E_k->js; DMVert_E_k[3] = E_k->je;
        DMVert_E_k[4] = E_k->ks; DMVert_E_k[5] = E_k->ke;
        // find atoms that have nonlocal influence the element domain
        GetInfluencingAtoms_nloc(pSPARC, &E_k->AtmNloc, DMVert_E_k,
            DDBP_info->elemcomm_index < 0 ? MPI_COMM_NULL : DDBP_info->elemcomm);

        // calculate global nonlocal projectors restricted to element
        if (pSPARC->isGammaPoint)
            CalculateNonlocalProjectors(pSPARC, &E_k->nlocProj, E_k->AtmNloc, DMVert_E_k,
                DDBP_info->elemcomm_index < 0 ? MPI_COMM_NULL : DDBP_info->elemcomm);
        else
            CalculateNonlocalProjectors_kpt(pSPARC, &E_k->nlocProj, E_k->AtmNloc, DMVert_E_k,
                DDBP_info->elemcomm_index < 0 ? MPI_COMM_NULL : DDBP_info->elemcomm);

        // initialize the DDBP nonlocal projectors (will be calculated in each SCF)
        int Ntypes = ESPRC_k->Ntypes;
        int nALB = E_k->nALB;
        DDBP_HAMILT_ERBLKS *H_DDBP_Ek = &E_k->H_DDBP_Ek;
        DDBP_VNL *Vnl_DDBP = &H_DDBP_Ek->Vnl_DDBP;
        init_nlocProj_DDBP(nALB, Ntypes, E_k->nlocProj, E_k->AtmNloc,
            &Vnl_DDBP->nlocProj, &Vnl_DDBP->AtmNloc,
            (int) (DDBP_info->elemcomm != MPI_COMM_NULL && DDBP_info->elemcomm_index >= 0));
        init_Vnl_DDBP(Vnl_DDBP, Ntypes, pSPARC->n_atom, pSPARC->nAtomv,
            pSPARC->localPsd, pSPARC->IP_displ, pSPARC->psd, pSPARC->dV);
    }

    // #ifdef DEBUG
    // // TODO: remove after check
    // for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
    //     DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
    //     // TODO: remove after check
    //     for (int ii = 0; ii < nproc; ii++) {
    //         MPI_Barrier(DDBP_info->elemcomm);
    //         if (ii == rank) print_Element(E_k);
    //     }
    // }
    // #endif
}

/**
 * @brief Broadcast an array from active processes to inactive processes using
 *        inter-communicator broadcasting.
 * 
 * @param buffer Buffer to be broadcasted.
 * @param count Count of values in buffer.
 * @param datatype Datatype of buffer to be broadcasted.
 * @param root Rank of root process in the send group.
 * @param kptcomm Communicator including both active and inactive processes.
 * @param proc_active Flag indicating if a process is in the active group.
 */
void intercomm_bcast(
    void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm kptcomm, int proc_active)
{
    int rank_kpt;
    MPI_Comm_rank(kptcomm, &rank_kpt);

    // Split the kpt_comm for all active processes in pSPARC->kptcomm
    MPI_Comm kpt_comm; // kpt_comm split by whether it's active
    MPI_Comm_split(kptcomm, proc_active, rank_kpt, &kpt_comm);

    // create a inter comm between the active processes and idle ones in kptcomm
    MPI_Comm kpt_inter;
    create_Cart_inter_comm(kptcomm,
        proc_active ? kpt_comm : MPI_COMM_NULL, &kpt_inter);

    if (kpt_inter != MPI_COMM_NULL) {
        if (proc_active) { // sender comm
            MPI_Bcast(buffer, count, datatype, rank_kpt == root ? MPI_ROOT : MPI_PROC_NULL, kpt_inter);
        } else { // receiver comm
            MPI_Bcast(buffer, count, datatype, root, kpt_inter);
        }
    }

    MPI_Comm_free(&kpt_comm);
    if (kpt_inter != MPI_COMM_NULL)
        MPI_Comm_free(&kpt_inter);
}


void collect_Veff_global(SPARC_OBJ *pSPARC, double *Veff_global)
{
    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start collecting Veff_global ... \n");
    #endif

    double t_comm = 0.0, t_bcast = 0.0;
    double t1, t2;

    int proc_active = (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) ? 0 : 1;
    int rank_kpt;
    MPI_Comm_rank(pSPARC->kptcomm, &rank_kpt);
    int rank_dmcomm = -1;
    if (pSPARC->dmcomm != MPI_COMM_NULL)
        MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);

    // global grid size
    int Nx = pSPARC->Nx;
    int Ny = pSPARC->Ny;
    int Nz = pSPARC->Nz;
    int Nd = Nx * Ny * Nz;

    // start collecting Veff_global
    if (proc_active == 1) {
        t1 = MPI_Wtime();
        MPI_Comm recv_comm = MPI_COMM_NULL;
        if (rank_dmcomm == 0) {
            int dims[3] = {1,1,1}, periods[3] = {1,1,1};
            // create a cartesian topology on one process (rank 0)
            MPI_Cart_create(MPI_COMM_SELF, 3, dims, periods, 0, &recv_comm);
        }
        t2 = MPI_Wtime();
        t_comm += t2 - t1;

        int gridsizes[3], sdims[3], rdims[3], rDMVert[6];
        gridsizes[0] = Nx; gridsizes[1] = Ny; gridsizes[2] = Nz;
        sdims[0] = pSPARC->npNdx;
        sdims[1] = pSPARC->npNdy;
        sdims[2] = pSPARC->npNdz;
        rdims[0] = rdims[1] = rdims[2] = 1;
        rDMVert[0] = 0; rDMVert[1] = Nx-1;
        rDMVert[2] = 0; rDMVert[3] = Ny-1;
        rDMVert[4] = 0; rDMVert[5] = Nz-1;

        D2D_OBJ d2d_sender, d2d_recvr;
        Set_D2D_Target(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices_dmcomm, rDMVert,
            pSPARC->dmcomm, sdims, recv_comm, rdims, pSPARC->bandcomm);
        D2D(&d2d_sender, &d2d_recvr, gridsizes, pSPARC->DMVertices_dmcomm, pSPARC->Veff_loc_dmcomm,
            rDMVert, Veff_global, pSPARC->dmcomm, sdims, recv_comm, rdims, pSPARC->bandcomm);
        Free_D2D_Target(&d2d_sender, &d2d_recvr, pSPARC->dmcomm, recv_comm);

        t1 = MPI_Wtime();
        MPI_Bcast(Veff_global, Nd, MPI_DOUBLE, 0, pSPARC->dmcomm);
        t2 = MPI_Wtime();
        t_bcast += t2 - t1;

        if (rank_dmcomm == 0) MPI_Comm_free(&recv_comm);
    }

    t1 = MPI_Wtime();
    // bcast Veff_global from active processes within kptcomm to idle processes (inter-comm bcast)
    intercomm_bcast(Veff_global, Nd, MPI_DOUBLE, 0, pSPARC->kptcomm, proc_active);
    t2 = MPI_Wtime();
    t_bcast += t2 - t1;

    #ifdef DEBUG
    if (rank_t == 0) printf("== Setup Veff_k ==: set up comms took %.3f ms\n", t_comm*1e3);
    if (rank_t == 0) printf("== Setup Veff_k ==: Bcast took %.3f ms\n", t_bcast*1e3);
    #endif
}



/**
 * @brief Extract values in a local domain from the global vector
 *
 * @param gridsizes Global grid sizes.
 * @param BCs Global boundary conditions.
 * @param V_global Global vector.
 * @param DMVerts Domain vertices (relative) for the local domain.
 * @param is Start index in x dir (can be different from DMVerts[0]).
 * @param js Start index in y dir (can be different from DMVerts[2]).
 * @param ks Start index in z dir (can be different from DMVerts[4]).
 * @param V_loc Local vector (output).
 */
void extract_local_from_global(
    int gridsizes[3], int BCs[3], const double *V_global,
    int DMVerts[6], int is, int js, int ks, double *V_loc)
{
    int Nx = gridsizes[0];
    int Ny = gridsizes[1];
    int Nz = gridsizes[2];
    int BCx = BCs[0];
    int BCy = BCs[1];
    int BCz = BCs[2];
    int nx = DMVerts[1] - DMVerts[0] + 1;
    int ny = DMVerts[3] - DMVerts[2] + 1;
    int nz = DMVerts[5] - DMVerts[4] + 1;

    char isOut = 'F'; // is index outside the domain
    for (int kk = 0; kk < nz; kk++) {
        int kk_g = kk + ks;
        if (BCz == 0) {
            kk_g %= Nz; // map index back for PBC
            if (kk_g < 0) kk_g += Nz;
        } else if (kk_g >= Nz || kk_g < 0) {
            isOut = 'T';
        }
        for (int jj = 0; jj < ny; jj++) {
            int jj_g = jj + js;
            if (BCy == 0) {
                jj_g %= Ny; // map index back for PBC
                if (jj_g < 0) jj_g += Ny;
            } else if (jj_g >= Ny || jj_g < 0) {
                isOut = 'T';
            }
            for (int ii = 0; ii < nx; ii++) {
                int ii_g = ii + is;
                if (BCx == 0) {
                    ii_g %= Nx; // map index back for PBC
                    if (ii_g < 0) ii_g += Nx;
                } else if (ii_g >= Nx || ii_g < 0) {
                    isOut = 'T';
                }
                int ind_DM = (kk * ny + jj) * nx + ii;
                int ind_g  = (kk_g * Ny + jj_g) * Nx + ii_g;
                // need to map index back to domain if it goes out
                V_loc[ind_DM] = (isOut == 'T') ? 0.0 : V_global[ind_g];
            }
        }
    }
}



/**
 * @brief  Set up local effective potentials for all DDBP (extended) elements.
 *
 *         This has to be done every time density is updated (in each SCF).
 */
void setup_ddbp_element_Veff(SPARC_OBJ *pSPARC) {
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start setting up Veff_k for DDBP elements ... \n");
    #endif

    int proc_active = (pSPARC->bandcomm_index < 0 || pSPARC->dmcomm == MPI_COMM_NULL) ? 0 : 1;

    // global grid size
    int Nx = pSPARC->Nx;
    int Ny = pSPARC->Ny;
    int Nz = pSPARC->Nz;
    int Nd = Nx * Ny * Nz;

    // first collect Veff into one process
    // we note that Veff is transferred to dmcomm
    double *Veff_global = NULL;
    Veff_global = (double *)malloc(Nd * sizeof(double));
    assert(Veff_global != NULL);

    // TODO: collect Veff from dmcomm to one process, and bcast to idle dmcomm's
    // TODO: use inter-communicator to Bcast from the active dmcomm to idle dmcomm
    double t1, t2;
    t1 = MPI_Wtime();
    if (pSPARC->npNd == 1) {
        if (proc_active) {
            memcpy(Veff_global, pSPARC->Veff_loc_dmcomm, Nd * sizeof(double));
        }
        intercomm_bcast(Veff_global, Nd, MPI_DOUBLE, 0, pSPARC->kptcomm, proc_active);
    } else {
        // collect the Veff_global by D2D and then Bcast to idel processes in kptcomm
        collect_Veff_global(pSPARC, Veff_global);
    }
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if (rank_t == 0)
        printf("== Setup Veff_k ==: collecting Veff_global took: %.3f ms\n", (t2-t1)*1e3);
    #endif

    if (Veff_global == NULL) {
        printf("[FATAL] BUG in setup_ddbp_element_Veff: Veff = NULL!\n");
        assert(Veff_global != NULL);
    }

    int BCx = DDBP_info->BCx;
    int BCy = DDBP_info->BCy;
    int BCz = DDBP_info->BCz;
    char isOut = 'F'; // is index outside the domain

    // go over each element and set up local effective potentials
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;

        int gridsizes[3] = {Nx, Ny, Nz};
        int BCs[3] = {BCx, BCy, BCz};
        
        // if (ESPRC_k->dmcomm != MPI_COMM_NULL) {
            // before updating Veff_loc_dmcomm, we store the history
            memcpy(E_k->Veff_loc_dmcomm_prev, ESPRC_k->Veff_loc_dmcomm,
                E_k->nd_ex_d*sizeof(double));

            // set up Veff for dmcomm
            extract_local_from_global(gridsizes, BCs, Veff_global, E_k->DMVert_ex,
                E_k->is_ex, E_k->js_ex, E_k->ks_ex, ESPRC_k->Veff_loc_dmcomm);
        // }
        #ifdef DEBUG_VEFF
        double *v_ref = malloc(E_k->nd_ex_d * sizeof(double));
        assert(v_ref != NULL);

        // set up Veff_loc_dmcomm for basis calculation
        for (int kk = 0; kk < E_k->nz_ex_d; kk++) {
            int kk_g = kk + E_k->ks_ex;
            if (BCz == 0) {
                kk_g %= Nz; // map index back for PBC
                if (kk_g < 0) kk_g += Nz;
            } else if (kk_g >= Nz || kk_g < 0) {
                isOut = 'T';
            }
            for (int jj = 0; jj < E_k->ny_ex_d; jj++) {
                int jj_g = jj + E_k->js_ex;
                if (BCy == 0) {
                    jj_g %= Ny; // map index back for PBC
                    if (jj_g < 0) jj_g += Ny;
                } else if (jj_g >= Ny || jj_g < 0) {
                    isOut = 'T';
                }
                for (int ii = 0; ii < E_k->nx_ex_d; ii++) {
                    int ii_g = ii + E_k->is_ex;
                    if (BCx == 0) {
                        ii_g %= Nx; // map index back for PBC
                        if (ii_g < 0) ii_g += Nx;
                    } else if (ii_g >= Nx || ii_g < 0) {
                        isOut = 'T';
                    }
                    int ind_DM = (kk * E_k->ny_ex_d + jj) * E_k->nx_ex_d + ii;
                    int ind_g  = (kk_g * Ny + jj_g) * Nx + ii_g;
                    // need to map index back to domain if it goes out
                    v_ref[ind_DM] = (isOut == 'T') ? 0.0 : Veff_global[ind_g];
                }
            }
        }

        int info = double_check_arrays(v_ref, ESPRC_k->Veff_loc_dmcomm, E_k->nd_ex_d);
        free(v_ref);
        assert(info == 0);
        #endif

        ESPRC_k->req_veff_loc = MPI_REQUEST_NULL;

        // set up Veff for Lanczos
        // perform Lanczos in a Cartesian topology embeded in elemcomm
        // instead of each dmcomm to utilize more processes
        extract_local_from_global(gridsizes, BCs, Veff_global, E_k->DMVert_ex_topo,
            E_k->is_ex, E_k->js_ex, E_k->ks_ex, ESPRC_k->Veff_loc_kptcomm_topo);
    }

    free(Veff_global);

    #ifdef DEBUG
    if (rank_t == 0) printf("Done.\n");
    #endif
}



/**
 * @brief   Find DDBP basis in the extended element of E_k using CheFSI.
 */
void find_extended_element_basis_CheFSI(DDBP_INFO *DDBP_info, DDBP_ELEM *E_k,
    double *x0, int count, int kpt, int spn_i)
{
    if (E_k->nALB == 0) return;

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start basis calcuation using CheFSI for element %d ... \n", E_k->index);
    double t_s,t_e;
    t_s = MPI_Wtime();
    #endif

    SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    int nspn = ESPRC_k->Nspin_spincomm;
    int nkpt = ESPRC_k->Nkpts_kptcomm;
    int nrow = E_k->nd_ex_d;
    int ncol = DDBP_info->n_basis_basiscomm;
    // size of each matrix
    int size_k = nrow * ncol;
    int size_s = size_k * nkpt;

    double lambda_cutoff = 0.0;

    // determine the constants for performing chebyshev filtering
    // TODO: need to set up d2d for spin case
    Chebyshevfilter_constants(ESPRC_k, x0, &lambda_cutoff,
        &ESPRC_k->eigmin[spn_i], &ESPRC_k->eigmax[spn_i], count, kpt, spn_i);

    double *v_tilde_f = (double *)malloc(size_s * sizeof(*v_tilde_f)); // filtered basis
    assert(v_tilde_f != NULL);
    ESPRC_k->Yorb = v_tilde_f;

    double t1, t2, t_temp;
    t1 = MPI_Wtime();
    int m = ESPRC_k->ChebDegree;
    // Apply Chebyshev filtering: v_tilde_f = p_m(H_k) * v_tilde
    ChebyshevFiltering(ESPRC_k, E_k->DMVert_ex, E_k->v_tilde + spn_i*size_s,
                       v_tilde_f, ncol, m, lambda_cutoff,
                       ESPRC_k->eigmax[spn_i], ESPRC_k->eigmin[spn_i], kpt, spn_i,
                       ESPRC_k->dmcomm, &t_temp);
    t2 = MPI_Wtime();
    #ifdef DEBUG
    if (rank_t == 0) 
        printf("Chebyshev filtering on basis (ncol = %2d, degree = %d) took: %.3f ms.\n",
            ncol, ESPRC_k->ChebDegree, (t2-t1)*1e3);
    #endif
    ESPRC_k->Hp = (double *)calloc(ESPRC_k->nr_Hp_BLCYC * ESPRC_k->nc_Hp_BLCYC , sizeof(double));
    ESPRC_k->Mp = (double *)calloc(ESPRC_k->nr_Mp_BLCYC * ESPRC_k->nc_Mp_BLCYC , sizeof(double));
    ESPRC_k->Q  = (double *)calloc(ESPRC_k->nr_Q_BLCYC * ESPRC_k->nc_Q_BLCYC , sizeof(double));
    ESPRC_k->Yorb_BLCYC = (double *) malloc(ESPRC_k->nr_orb_BLCYC * ESPRC_k->nc_orb_BLCYC * sizeof(double));
    ESPRC_k->Xorb_BLCYC = (double *) malloc(ESPRC_k->nr_orb_BLCYC * ESPRC_k->nc_orb_BLCYC * sizeof(double));
    assert(ESPRC_k->Hp != NULL && ESPRC_k->Mp != NULL && ESPRC_k->Q != NULL);
    assert(ESPRC_k->Yorb_BLCYC != NULL && ESPRC_k->Xorb_BLCYC != NULL);

    // project the element Hamiltonian H_k onto the basis
    Project_Hamiltonian(ESPRC_k, E_k->DMVert_ex, v_tilde_f,
                        ESPRC_k->Hp, ESPRC_k->Mp, kpt, spn_i, ESPRC_k->dmcomm);

    // solve subspace eigenproblem for H_k
    Solve_Generalized_EigenProblem(ESPRC_k, kpt, spn_i);
    int nproc_kptcomm;
    MPI_Comm_size(ESPRC_k->kptcomm, &nproc_kptcomm);
    if (ESPRC_k->useLAPACK == 1 && nproc_kptcomm > 1) {
        MPI_Bcast(ESPRC_k->lambda, ESPRC_k->Nstates * ESPRC_k->Nspin_spincomm,
                  MPI_DOUBLE, 0, ESPRC_k->kptcomm); // TODO: bcast in blacscomm if possible
    }

    // perform subspace rotation
    // find Y * Q, store the result in Xorb (band+domain) and Xorb_BLCYC (block cyclic format)
    Subspace_Rotation(ESPRC_k, ESPRC_k->Yorb_BLCYC, ESPRC_k->Q,
                      ESPRC_k->Xorb_BLCYC, E_k->v_tilde + spn_i*size_s, kpt, spn_i);

    free(ESPRC_k->Hp); ESPRC_k->Hp = NULL;
    free(ESPRC_k->Mp); ESPRC_k->Mp = NULL;
    free(ESPRC_k->Q);  ESPRC_k->Q  = NULL;
    free(ESPRC_k->Yorb_BLCYC); ESPRC_k->Yorb_BLCYC = NULL;
    free(ESPRC_k->Xorb_BLCYC); ESPRC_k->Xorb_BLCYC = NULL;
    free(v_tilde_f);

    #ifdef DEBUG
    t_e = MPI_Wtime();
    if (rank_t == 0)
    printf("Finished CheFSI on basis in extended element %d, which took %.3f ms.\n\n", E_k->index, (t_e-t_s)*1e3);
    #endif
}


/**
 * @brief   Restrict DDBP basis from the extended element to the element.
 */
void restrict_basis_to_element(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, int kpt, int spn_i
)
{
    if (DDBP_info->elemcomm_index == -1 || DDBP_info->basiscomm_index == -1 ||
        DDBP_info->dmcomm == MPI_COMM_NULL) {
        return;
    }

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start restricting basis to element %d ... \n", E_k->index);
    #endif

    SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    int nrow_ex = E_k->nd_ex_d;
    int nrow = E_k->nd_d;
    int ncol = DDBP_info->n_basis_basiscomm;
    int nkpt = ESPRC_k->Nkpts_kptcomm;
    // size of each matrix
    int size_s_ex = nrow_ex * ncol * nkpt;
    int size_s = nrow * ncol * nkpt;
    double *v_tilde = E_k->v_tilde + spn_i*size_s_ex;
    double *v = E_k->v + spn_i*size_s;
    for (int n = 0; n < ncol; n++) {
        double *v_ex_n = v_tilde + n * nrow_ex;
        double *v_n = v + n * nrow;
        restrict_to_element(E_k, v_ex_n, v_n);
    }

    #ifdef DEBUG
    if (rank_t == 0) printf("Done.\n");
    #endif
}


/**
 * @brief   Orthogonalize the restricted DDBP basis within the element.
 *
 *          Warning: This function assumes no domain paral for basis calculation.
 */
void orth_restricted_basis(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, int kpt, int spn_i
)
{
    if (DDBP_info->elemcomm_index == -1 || DDBP_info->basiscomm_index == -1 ||
        DDBP_info->dmcomm == MPI_COMM_NULL) {
        return;
    }

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start orthogonalizing the restriced basis in element %d ... ", E_k->index);
    #endif

    SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    int nrow = E_k->nd_d;
    int ncol = DDBP_info->n_basis_basiscomm;
    int nkpt = ESPRC_k->Nkpts_kptcomm;
    // size of each matrix
    int size_s = nrow * ncol * nkpt;

    void orthChol(
        SPARC_OBJ *pSPARC, int Ns, int Nt, double *V,
        int ncol, int *descV, MPI_Comm rowcomm
    );

    //::debug -- Orth the extended basis (no effect, since it's already orthogonal)
    // int nrow_ex = E_k->nd_ex_d;
    // int size_s_ex = nrow_ex * ncol * nkpt;
    // orthChol(ESPRC_k, E_k->nd_ex, E_k->nALB, E_k->v_tilde + spn_i*size_s_ex,
    //         ncol, ESPRC_k->desc_orbitals, DDBP_info->blacscomm);

    // use Cholesky QR method to orthogonalize the restricted basis
    // Warning: assuming no domain parallelization for the basis v
    orthChol(ESPRC_k, E_k->nd, E_k->nALB, E_k->v + spn_i*size_s,
        ncol, E_k->desc_v, DDBP_info->blacscomm);

    #ifdef DEBUG
    if (rank_t == 0) printf("Done.\n");
    #endif
}


/**
 * @brief Update the DDBP basis history vectors.
 *        v_prev = v.
 * 
 * @param DDBP_info DDBP info object.
 * @param nspin Local number of spin assigned to the process.
 * @param nkpt Local number of kpoints assigned to the process.
 * @param isGammaPoint Flag indicating whether it's gamma-point.
 * @param isInitSCF Flag indicating if it's the very first SCF.
 * @param kpt K-point index (local).
 * @param spn_i Spin index (local).
 */
void Update_basis_history(
    DDBP_INFO *DDBP_info, int nspin, int nkpt, int isGammaPoint,
    int isInitSCF, int kpt, int spn_i)
{
    if (isInitSCF) return;

    // go over each element and update DDBP basis history
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];

        int nrow = E_k->nd_d;
        int ncol = DDBP_info->n_basis_basiscomm;
        int size_k = nrow * ncol;
        int size_s = size_k * nkpt;
        if (isGammaPoint) {
            double *v = E_k->v + spn_i*size_s + kpt*size_k;
            double *v_prev = E_k->v_prev + spn_i*size_s + kpt*size_k;
            memcpy(v_prev, v, size_k * sizeof(*v));
        } else {
            double _Complex *v = E_k->v_cmplx + spn_i*size_s + kpt*size_k;
            double _Complex *v_prev = E_k->v_prev_cmplx + spn_i*size_s + kpt*size_k;
            memcpy(v_prev, v, size_k * sizeof(*v));
        }
    }
}



/**
 * @brief   Calculate DDBP basis for all elements.
 */
void Calculate_DDBP_basis(SPARC_OBJ *pSPARC, int count, int kpt, int spn_i) {
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
    int isFirstEGS = (pSPARC->elecgs_Count - pSPARC->StressCount) == 0 ? 1 : 0;
    // flag to check if this is the very first SCF of the entire simulation
    int isInitSCF = (isFirstEGS && count == 0) ? 1 : 0;
    int SCF_iter = count < pSPARC->rhoTrigger ? 1 : count - pSPARC->rhoTrigger + 2;

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start DDBP basis calcuation ... \n");
    #endif

    // go over each element and calculate DDBP basis
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;

        // set up ESPRC_k for basis calculation
        // setup_element_SPARC_obj(pSPARC, E_k);

        double *x0 = ESPRC_k->Lanczos_x0;

        if (count == 0) {
            double cellsizes[3], meshes[3];
            cellsizes[0] = ESPRC_k->range_x;
            cellsizes[1] = ESPRC_k->range_y;
            cellsizes[2] = ESPRC_k->range_z;
            meshes[0] = ESPRC_k->delta_x;
            meshes[1] = ESPRC_k->delta_y;
            meshes[2] = ESPRC_k->delta_z;
            int gridsizes[3];
            gridsizes[0] = ESPRC_k->Nx;
            gridsizes[1] = ESPRC_k->Ny;
            gridsizes[2] = ESPRC_k->Nz;
            int RandFlag = (ESPRC_k->cell_typ != 0 ||
                ESPRC_k->chefsibound_flag == 0 ||
                ESPRC_k->chefsibound_flag == 1);
            // set up initial guess for Lanczos
            init_guess_Lanczos(x0, cellsizes, gridsizes, meshes, E_k->DMVert_ex_topo,
                RandFlag, DDBP_info->elemcomm_topo);
        }

        // find DDBP basis in the extended element of E_k: v_tilde
        // TODO: consider repeating this step multiple times to obtain a more
        //       accurate set of eigenvectors of H_k as basis
        int nrepeat = isInitSCF ? 4 : 1;
        for (int r = 0; r < nrepeat; r++)
            find_extended_element_basis_CheFSI(DDBP_info, E_k, x0, count, kpt, spn_i);

        // restrict the basis functions to E_k: v = v_tilde(isInEk)
        restrict_basis_to_element(DDBP_info, E_k, kpt, spn_i);

        // orthogonalize the restriced basis functions
        orth_restricted_basis(DDBP_info, E_k, kpt, spn_i);
    }
}


/**
 * @brief Find overlap matrix of a basis Vk and another basis Wk within
 *        element E_k. We assume the result is global, i.e., every process
 *        will have a full copy of the resulting matrix Mk.
 *          Mk = Vk^T * Wk.
 * 
 * @param Nd Number of grid points of Vk in element E_k (local).
 * @param nALB Number of basis in element E_k (global).
 * @param Vk Basis Vk.
 * @param descVk Descriptor for Vk.
 * @param Wk Basis Wk.
 * @param descWk Descriptor for Wk.
 * @param Mk Overlap matrix (global).
 * @param descMk Descriptor for Mk.
 * @param rowcomm Row communicator where row-wise distribution of Vk/Wk happens.
 * @param colcomm Column communicator where column-wise distribution of Vk/Wk happens.
 *                Note: in the current implementation, colcomm only has one process
 *                for basis generation, although this function doesn't requre that.)
 */
void element_basis_overlap_matrix(
    int Nd, int nALB, const double *Vk, int *descVk, const double *Wk,
    int *descWk, double *Mk, int *descMk, MPI_Comm rowcomm, MPI_Comm colcomm
)
{
    // Vk^T * Wk in a subcomm within rowcomm
    // TODO: if no column paral., skip this and do a local dgemm!
    pdgemm_subcomm("T", "N", nALB, nALB, Nd, 1.0, Vk, descVk, Wk, descWk,
        0.0, Mk, descMk, rowcomm, best_max_nproc(Nd, nALB, "pdgemm"));

    int nproc_rowcomm, nproc_colcomm;
    MPI_Comm_size(rowcomm, &nproc_rowcomm);
    MPI_Comm_size(colcomm, &nproc_colcomm);

    if (nproc_colcomm > 1) { // in the current implementation, this won't happen
        MPI_Allreduce(MPI_IN_PLACE, Mk, nALB*nALB, MPI_DOUBLE, MPI_SUM, colcomm);
    }

    if (nproc_rowcomm > 1)
        MPI_Bcast(Mk, nALB*nALB, MPI_DOUBLE, 0, rowcomm);
}



/**
 * @brief Calculate the basis overlap matrix, i.e.,
 *                   Mvvp = v^T * v_prev,
 *        where v is the current DDBP basis, v_prev is the basis from
 *        the previous SCF step. The basis overlap matrix is used to
 *        align any function expressed in the previous basis to the
 *        current basis.
 * 
 * @param DDBP_info DDBP info object.
 * @param nkpt Number of kpoints assigned (local).
 * @param isInitSCF Flag indicating if it's the very first SCF.
 * @param kpt K-point index (local).
 * @param spn_i Spin index (local).
 */
void Calculate_basis_overlap_matrix(
    DDBP_INFO *DDBP_info, int nkpt, int isInitSCF, int kpt, int spn_i)
{
    if (isInitSCF) return;
    if (DDBP_info->elemcomm_index == -1 || DDBP_info->basiscomm_index == -1 ||
        DDBP_info->dmcomm == MPI_COMM_NULL) {
        return;
    }

    // go over each element and find DDBP basis overlap matrix
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        // SPARC_OBJ *ESPRC_k = E_k->ESPRC;

        int Nd = E_k->nd_d;
        int nALB = E_k->nALB;

        int nrow = E_k->nd_d;
        int ncol = DDBP_info->n_basis_basiscomm;
        // int nkpt = ESPRC_k->Nkpts_kptcomm;
        // size of each basis matrix
        int size_s = nrow * ncol * nkpt;
        int size_k = nrow * ncol;
        #if defined(USE_MKL) || defined(USE_SCALAPACK)
        int desc_Mvvp[9];
        int ZERO = 0, info;
        int mb = max(1, nALB);
        int nb = max(1, nALB);
        int llda = max(1, nALB);
        descinit_(desc_Mvvp, &nALB, &nALB, &mb, &nb, &ZERO, &ZERO, &DDBP_info->ictxt_blacs, &llda, &info);

        double *v_i = E_k->v + spn_i*size_s + kpt * size_k;
        double *v_im1 = E_k->v_prev + spn_i*size_s + kpt * size_k;
        // if Mvvp is contiguous memory, use the following
        // double *Mvvp_i = E_k->Mvvp + spn_i*nALB*nALB*nkpt + kpt*nALB*nALB;
        double *Mvvp_i = E_k->Mvvp[spn_i][kpt];
        // find Mvvp_i = v_i^T * v_im1
        element_basis_overlap_matrix(
            Nd, nALB, v_i, E_k->desc_v, v_im1, E_k->desc_v, Mvvp_i,
            desc_Mvvp, DDBP_info->blacscomm, DDBP_info->dmcomm
        );
        #else
        // TODO: implement corresponding routines without MKL/ScaLAPACK
        assert(0);
        #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    }
}


/**
 * @brief Align the orbitals with the current basis.
 *
 *        Given a set of orbitals expressed in the DDBP basis from the
 *        previous SCF, we align it the the current basis using the basis
 *        overlap matrix:
 *                 X = Mvvp * X,
 *        where Mvvp is the basis overlap matrix, Mvvp = v^T * v_prev.
 * 
 * @param DDBP_info DDBP info object.
 * @param X Orbitals expressed in the DDBP basis (input/output).
 * @param isInitSCF Flag indicating if it's the very first SCF.
 * @param kpt K-point index (local).
 * @param spn_i Spin index (local).
 */
void align_orbitals_with_current_basis(
    DDBP_INFO *DDBP_info, DDBP_ARRAY *X, int isInitSCF, int kpt, int spn_i)
{
    if (isInitSCF) return;
    if (DDBP_info->elemcomm_index == -1 || DDBP_info->basiscomm_index == -1 ||
        DDBP_info->dmcomm == MPI_COMM_NULL) {
        return;
    }

    int ncol = X->ncol;
    int nelem = X->nelem;
    if (ncol == 0) return;

    DDBP_ARRAY Aux;
    DDBP_ARRAY *MX = &Aux;
    duplicate_DDBP_Array_template(X, MX);

    // go over each element, and find Mvvp_k * x_k
    for (int k = 0; k < nelem; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        int nALB = E_k->nALB;
        double *Mvvp_k = E_k->Mvvp[spn_i][kpt];
        double *xk = X->array[k];
        double *Mxk = MX->array[k];
        double alpha = 1.0, beta = 0.0;
        adaptive_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            nALB, ncol, nALB, alpha, Mvvp_k, nALB, xk, nALB, beta, Mxk, nALB);
    }

    // X = MX
    axpby_DDBP_Array(1.0, MX, 0.0, X);
    
    // delete MX
    delete_DDBP_Array(MX);
}
