/**
 * @file    ddbp_hamiltonian.c
 * @brief   This file contains the functions for the Discrete
 *          Discontinuous Basis Projection (DDBP) Hamiltonian routines.
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
#include "linearAlgebra.h"
#include "sq3.h"
#include "cs.h"
#include "ddbp.h"

// for Lap-Vec routines
#include "lapVecRoutines.h"
#include "lapVecRoutinesKpt.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define TEMP_TOL 1e-14


/**
 * @brief   Calculate (-1/2 D^2 + Veff_loc + c * I) times a bunch of vectors in
 *          a matrix-free way.
 *
 *          This function is copied from hamiltonianVecRoutines.c and modified
 *          that it skips the non-local part.
 */
void Hamiltonian_loc_vectors_mult(
    const SPARC_OBJ *pSPARC, int DMnd, int *DMVertices, double *Veff_loc,
    int ncol, double c, double *x, double *Hx, MPI_Comm comm
)
{
    int nproc;
    MPI_Comm_size(comm, &nproc);

    // TODO: make dims an input parameter
    int dims[3], periods[3], my_coords[3];
    if (nproc > 1)
        MPI_Cart_get(comm, 3, dims, periods, my_coords);
    else
        dims[0] = dims[1] = dims[2] = 1;

    // first find (-0.5 * Lap + Veff + c) * x
    if (pSPARC->cell_typ == 0) { // orthogonal cell
        for (int i = 0; i < ncol; i++) {
            Lap_plus_diag_vec_mult_orth(
                pSPARC, DMnd, DMVertices, 1, -0.5, 1.0, c, Veff_loc,
                x+i*DMnd, Hx+i*DMnd, comm, dims
            );
        }
        // Lap_plus_diag_vec_mult_orth(
        //         pSPARC, DMnd, DMVertices, ncol, -0.5, 1.0, c, Veff_loc,
        //         x, Hx, comm, dims
        // ); // slower than the for loop above
    } else {  // non-orthogonal cell
        MPI_Comm comm2;
        if (comm == pSPARC->kptcomm_topo)
            comm2 = pSPARC->kptcomm_topo_dist_graph; // pSPARC->comm_dist_graph_phi
        else
            comm2 = pSPARC->comm_dist_graph_psi;
        for (int i = 0; i < ncol; i++) {
            Lap_plus_diag_vec_mult_nonorth(
                pSPARC, DMnd, DMVertices, 1, -0.5, 1.0, c, Veff_loc,
                x+i*DMnd, Hx+i*DMnd, comm, comm2, dims
            );
        }
    }
}



/**
 * @brief   Calculate the local part of the global Hamiltonian multiplied by the
 *          basis functions vk in element k. I.e., find
 *                             Hv_ex = H_loc * v_ex,
 *          where v_ex(x) = v(x), if x is within element, and 0 otherwise.
 *
 *          The local part of the global Hamiltonian is matrix-free (and sparse),
 *          and the basis functions are fully localized. The resulting functions
 *          is also localized, except it extends to outside of the element by FDn
 *          grid points in each direction, where FDn is half of the finite
 *          difference order.
 */
void H_loc_vk_mult(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, int ncol, const double *v, double *Hv_ex
)
{
    SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    int FDn = ESPRC_k->order / 2;
    int DMnx = E_k->nx_d;
    int DMny = E_k->ny_d;
    int DMnz = E_k->nz_d;
    int DMnd = DMnx * DMny * DMnz;
    int DMnx_ex = DMnx + ESPRC_k->order;
    int DMny_ex = DMny + ESPRC_k->order;
    int DMnz_ex = DMnz + ESPRC_k->order;
    int DMnd_ex = DMnx_ex * DMny_ex * DMnz_ex;

    // due to the FD stencil, the nnz will extend up to FDn nodes outside
    double *v_ex = calloc(ncol * DMnd_ex, sizeof(*v_ex));
    assert(v_ex != NULL);

    // copy v into extended v_ex (fill the extended part by 0's)
    for (int n = 0; n < ncol; n++) {
        // double *v_n = E_k->v + spn_i*size_s + n * DMnd;
        const double *v_n = v + n * DMnd;
        double *v_ex_n = v_ex + n * DMnd_ex;
        extract_subgrid(
            v_n, DMnx, DMny, DMnz,
            0, DMnx-1, 0, DMny-1, 0, DMnz-1,
            v_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
            FDn, FDn+DMnx-1, FDn, FDn+DMny-1, FDn, FDn+DMnz-1
        );
    }

    // TODO:   Instead of mapping data back, a better/faster way might
    // TODO: be to simply not extend the element in that direction,
    // TODO: and apply the global BC in that direciton. This way the
    // TODO: size of the vector will be reduced. However, this will
    // TODO: change the vector size, and when copying data to buffer,
    // TODO: one needs to modify the indices for these situations.
    // if there's only one element in one direction, then we should
    // fill the extended part by mapping the v values back if the
    // global BC is periodic!
    int BCs[3] = {DDBP_info->BCx,DDBP_info->BCy,DDBP_info->BCz};
    int Nes[3] = {DDBP_info->Nex,DDBP_info->Ney,DDBP_info->Nez};
    // set up (send) start and end block indices for v
    int isv[6] = {0,      DMnx-FDn, 0,      0,         0,      0       };
    int iev[6] = {FDn-1,  DMnx-1,   DMnx-1, DMnx-1,    DMnx-1, DMnx-1  };
    int jsv[6] = {0,      0,        0,      DMny-FDn,  0,      0       };
    int jev[6] = {DMny-1, DMny-1,   FDn-1,  DMny-1,    DMny-1, DMny-1  };
    int ksv[6] = {0,      0,        0,      0,         0,      DMnz-FDn};
    int kev[6] = {DMnz-1, DMnz-1,   DMnz-1, DMnz-1,    FDn-1,  DMnz-1  };
    // set up (recv) start and end block indices for v_ex
    int isv_ex[6] = {FDn+DMnx,   0         , FDn,        FDn       , FDn,        FDn       };
    int iev_ex[6] = {DMnx_ex-1,  FDn-1     , FDn+DMnx-1, FDn+DMnx-1, FDn+DMnx-1, FDn+DMnx-1};
    int jsv_ex[6] = {FDn,        FDn       , FDn+DMny,   0         , FDn,        FDn       };
    int jev_ex[6] = {FDn+DMny-1, FDn+DMny-1, DMny_ex-1,  FDn-1     , FDn+DMny-1, FDn+DMny-1};
    int ksv_ex[6] = {FDn,        FDn       , FDn,        FDn       , FDn+DMnz,   0         };
    int kev_ex[6] = {FDn+DMnz-1, FDn+DMnz-1, FDn+DMnz-1, FDn+DMnz-1, DMnz_ex-1,  FDn-1     };
    for (int d = 0; d < 3; d++) {
        if (Nes[d] == 1 && BCs[d] == 0) {
            // map v into extended v_ex
            for (int n = 0; n < ncol; n++) {
                const double *v_n = v + n * DMnd;
                double *v_ex_n = v_ex + n * DMnd_ex;
                int di = 2*d;
                extract_subgrid(
                    v_n, DMnx, DMny, DMnz,
                    // 0, DMnx-1, 0, DMny-1, 0, FDn-1,
                    isv[di], iev[di], jsv[di], jev[di], ksv[di], kev[di],
                    v_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
                    // FDn, FDn+DMnx-1, FDn, FDn+DMny-1, FDn+DMnz, DMnz_ex-1
                    isv_ex[di], iev_ex[di], jsv_ex[di], jev_ex[di], ksv_ex[di], kev_ex[di]
                );
                int dip = 2*d+1;
                extract_subgrid(
                    v_n, DMnx, DMny, DMnz,
                    // 0, DMnx-1, 0, DMny-1, DMnz-FDn, DMnz-1,
                    isv[dip], iev[dip], jsv[dip], jev[dip], ksv[dip], kev[dip],
                    v_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
                    // FDn, FDn+DMnx-1, FDn, FDn+DMny-1, 0, FDn-1
                    isv_ex[dip], iev_ex[dip], jsv_ex[dip], jev_ex[dip], ksv_ex[dip], kev_ex[dip]
                );
            }
        }
    }

    // extend Veff to extended domain (fill the extended part by 0's)
    double *Veff_loc_ex = calloc(DMnd_ex, sizeof(*v_ex));
    assert(Veff_loc_ex != NULL);
    extract_subgrid(
        ESPRC_k->Veff_loc_dmcomm, E_k->nx_ex_d, E_k->ny_ex_d, E_k->nz_ex_d,
        E_k->DMVert[0],E_k->DMVert[1],E_k->DMVert[2],
        E_k->DMVert[3],E_k->DMVert[4],E_k->DMVert[5],
        Veff_loc_ex, DMnx_ex, DMny_ex, DMnz_ex,
        FDn, FDn+DMnx-1, FDn, FDn+DMny-1, FDn, FDn+DMnz-1
    );

    // Warning: here we again assume no domain paral over elements, i.e., the
    //     domain owned by the current process is the entire element domain
    int DMVertices_ex[6];
    DMVertices_ex[0] = 0;
    DMVertices_ex[1] = DMnx_ex - 1;
    DMVertices_ex[2] = 0;
    DMVertices_ex[3] = DMny_ex - 1;
    DMVertices_ex[4] = 0;
    DMVertices_ex[5] = DMnz_ex - 1;

    // first find (-0.5 * Lap + Veff + c) * x
    // borrow ESPRC_k for E_k_tilde for element E_k
    // Note: need to set BC to Dirichlet for local Hamiltonian!
    // *Note: in principle, we need to set Dirichlet BC, however, in the case
    // *      we fill the extended part by 0, PBC will give the same result
    // int EBCs_t[3] = {ESPRC_k->BCx, ESPRC_k->BCy, ESPRC_k->BCz};
    // ESPRC_k->BCx = 1;
    // ESPRC_k->BCy = 1;
    // ESPRC_k->BCz = 1;
    Hamiltonian_loc_vectors_mult(
        ESPRC_k, DMnd_ex, DMVertices_ex, Veff_loc_ex, ncol, 0.0,
        v_ex, Hv_ex, ESPRC_k->dmcomm
    );
    // ESPRC_k->BCx = EBCs_t[0];
    // ESPRC_k->BCy = EBCs_t[1];
    // ESPRC_k->BCz = EBCs_t[2];

    free(v_ex);
    free(Veff_loc_ex);
}


/**
 * @brief   Copy Hv_ex to Hv and copy the extended part to sendbuf.
 *
 *          Hv_ex contains the part within the current element E_k, as
 *          well as the extended part that extends to neighbor elements.
 *          The extended part needs to be send to the corresponding
 *          elements. We first copy them into a buffer array.
 */
void copy_Hv_ex_to_buf(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, int ncol, const double *Hv_ex
    //! TODO: , double *Hv, double *sendbuf
)
{
    SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    int FDn = ESPRC_k->order / 2;
    int DMnx = E_k->nx_d;
    int DMny = E_k->ny_d;
    int DMnz = E_k->nz_d;
    int DMnd = E_k->nd_d;
    int DMnx_ex = DMnx + ESPRC_k->order;
    int DMny_ex = DMny + ESPRC_k->order;
    int DMnz_ex = DMnz + ESPRC_k->order;
    int DMnd_ex = DMnx_ex * DMny_ex * DMnz_ex;

    // TODO: free the memory after the local DDBP Hamiltonain is done
    // i.e., after vj' * Hv, for all j = neighbors of E_k
    E_k->Hv = malloc(DMnd * ncol * sizeof(*(E_k->Hv)));
    assert(E_k->Hv != NULL);

    // TODO: decide the order of buffers based on the graph topology.
    // TODO: test whether it's faster to transfer one column at a time.
    for (int n = 0; n < ncol; n++) {
        // double *v_n = E_k->v + spn_i*size_s + n * DMnd;
        const double *Hv_ex_n = Hv_ex + n * DMnd_ex;
        double *Hv_n = E_k->Hv + n * DMnd;
        // copy Hv
        extract_subgrid(
            Hv_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
            FDn, FDn+DMnx-1, FDn, FDn+DMny-1, FDn, FDn+DMnz-1,
            Hv_n, DMnx, DMny, DMnz,
            0, DMnx-1, 0, DMny-1, 0, DMnz-1
        );
    }

    // set up send buffer based on the ordering of the neighbors
    int nxex_in = DMnx_ex - FDn;
    int nyex_in = DMny_ex - FDn;
    int nzex_in = DMnz_ex - FDn;
    int istart[6] = {0,         nxex_in,   FDn,       FDn,        FDn,       FDn},
          iend[6] = {FDn-1,     DMnx_ex-1, nxex_in-1, nxex_in-1,  nxex_in-1, nxex_in-1},
        jstart[6] = {FDn,       FDn,       0,         nyex_in,    FDn,       FDn},
          jend[6] = {nyex_in-1, nyex_in-1, FDn-1,     DMny_ex-1,  nyex_in-1, nyex_in-1},
        kstart[6] = {FDn,       FDn,       FDn,       FDn,        0,         nzex_in},
          kend[6] = {nzex_in-1, nzex_in-1, nzex_in-1, nzex_in-1,  FDn-1,     DMnz_ex-1};

    int count = 0;
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
        // if dims[i] < 3 and periods[i] == 1, switch send buffer for left and right neighbors
        // int nbrcount = nbr_i + (1 - 2 * (nbr_i % 2)) *
        //              (int)(dims[nbr_i / 2] < 3 && periods[nbr_i / 2]);
        // TODO: do the swap at recv time!
        int nbrcount = nbr_i;
        int k_s = kstart[nbrcount];
        int k_e = kend  [nbrcount];
        int j_s = jstart[nbrcount];
        int j_e = jend  [nbrcount];
        int i_s = istart[nbrcount];
        int i_e = iend  [nbrcount];
        for (int n = 0; n < ncol; n++) {
            for (int k = k_s; k <= k_e; k++) {
                for (int j = j_s; j <= j_e; j++) {
                    for (int i = i_s; i <= i_e; i++) {
                        // E_k->sendbuf[count++] = Hv_ex(n,i,j,k);
                        int ind = n * DMnd_ex + (k * DMny_ex + j) * DMnx_ex + i;
                        E_k->sendbuf[count++] = Hv_ex[ind];
                    }
                }
            }
        }
    }

#ifdef DEBUG_DOUBLE_CHECK
    // use another way to copy the buffers and compare the results
    double *test_sendbuf = malloc(n_out * sizeof(*(E_k->Hv)));
    assert(test_sendbuf != NULL);

    int size_xbuff = FDn * DMny * DMnz;
    int size_ybuff = FDn * DMnx * DMnz;
    int size_zbuff = FDn * DMnx * DMny;
    // double *sendbuf_xl = E_k->sendbuf;
    double *sendbuf_xl = test_sendbuf;
    double *sendbuf_xr = sendbuf_xl + size_xbuff * ncol;
    double *sendbuf_yl = sendbuf_xr + size_xbuff * ncol;
    double *sendbuf_yr = sendbuf_yl + size_ybuff * ncol;
    double *sendbuf_zl = sendbuf_yr + size_ybuff * ncol;
    double *sendbuf_zr = sendbuf_zl + size_zbuff * ncol;

    // copy data for x-left neighbor element
    for (int n = 0; n < ncol; n++) {
        const double *Hv_ex_n = Hv_ex + n * DMnd_ex;
        double *buf_n = sendbuf_xl + n * size_xbuff;
        extract_subgrid(
            Hv_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
            0, FDn-1, FDn, FDn+DMny-1, FDn, FDn+DMnz-1,
            buf_n, FDn, DMny, DMnz,
            0, FDn-1, 0, DMny-1, 0, DMnz-1
        );
    }

    // copy data for x-right neighbor element
    for (int n = 0; n < ncol; n++) {
        const double *Hv_ex_n = Hv_ex + n * DMnd_ex;
        double *buf_n = sendbuf_xr + n * size_xbuff;
        extract_subgrid(
            Hv_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
            FDn+DMnx, DMnx_ex-1, FDn, FDn+DMny-1, FDn, FDn+DMnz-1,
            buf_n, FDn, DMny, DMnz,
            0, FDn-1, 0, DMny-1, 0, DMnz-1
        );
    }

    // copy data for y-left neighbor element
    for (int n = 0; n < ncol; n++) {
        const double *Hv_ex_n = Hv_ex + n * DMnd_ex;
        double *buf_n = sendbuf_yl + n * size_ybuff;
        extract_subgrid(
            Hv_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
            FDn, FDn+DMnx-1, 0, FDn-1, FDn, FDn+DMnz-1,
            buf_n, DMnx, FDn, DMnz,
            0, DMnx-1, 0, FDn-1, 0, DMnz-1
        );
    }

    // copy data for y-right neighbor element
    for (int n = 0; n < ncol; n++) {
        const double *Hv_ex_n = Hv_ex + n * DMnd_ex;
        double *buf_n = sendbuf_yr + n * size_ybuff;
        extract_subgrid(
            Hv_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
            FDn, FDn+DMnx-1, FDn+DMny, DMny_ex-1, FDn, FDn+DMnz-1,
            buf_n, DMnx, FDn, DMnz,
            0, DMnx-1, 0, FDn-1, 0, DMnz-1
        );
    }

    // copy data for z-left neighbor element
    for (int n = 0; n < ncol; n++) {
        const double *Hv_ex_n = Hv_ex + n * DMnd_ex;
        double *buf_n = sendbuf_zl + n * size_zbuff;
        extract_subgrid(
            Hv_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
            FDn, FDn+DMnx-1, FDn, FDn+DMny-1, 0, FDn-1,
            buf_n, DMnx, DMny, FDn,
            0, DMnx-1, 0, DMny-1, 0, FDn-1
        );
    }

    // copy data for z-right neighbor element
    for (int n = 0; n < ncol; n++) {
        const double *Hv_ex_n = Hv_ex + n * DMnd_ex;
        double *buf_n = sendbuf_zr + n * size_zbuff;
        extract_subgrid(
            Hv_ex_n, DMnx_ex, DMny_ex, DMnz_ex,
            FDn, FDn+DMnx-1, FDn, FDn+DMny-1, FDn+DMnz, DMnz_ex-1,
            buf_n, DMnx, DMny, FDn,
            0, DMnx-1, 0, DMny-1, 0, FDn-1
        );
    }

    //::debug Compare test_sendbuf and E_k->sendbuf
    double err = 0.0;
    for (int i = 0; i < n_out; i++) {
        // printf("E_k->sendbuf[%d] = %18.10f, test_sendbuf[%d] = %18.10f\n",
            // i,E_k->sendbuf[i],i,test_sendbuf[i]);
        err = max(err, fabs(E_k->sendbuf[i] - test_sendbuf[i]));
    }

    free(test_sendbuf);

    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf(RED "err = %.3e\n" RESET, err);
    assert(err < 1e-14);
#endif // #ifdef DEBUG_DOUBLE_CHECK
}


/**
 * @brief   (For Debugging) Check H*vk using another way.
 *
 *          In this routine, we directly calculate H * vk within the element
 *          to varify the Hv part is correctly calculated within E_k.
 */
void check_Hv(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, int ncol, double *v, const double *Hv,
    const double *Hv_ex
)
{
    SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    int FDn = ESPRC_k->order / 2;
    int DMnx = E_k->nx_d;
    int DMny = E_k->ny_d;
    int DMnz = E_k->nz_d;
    int DMnd = DMnx * DMny * DMnz;

    int DMnx_ex = DMnx + ESPRC_k->order;
    int DMny_ex = DMny + ESPRC_k->order;
    int DMnz_ex = DMnz + ESPRC_k->order;
    int DMnd_ex = DMnx_ex * DMny_ex * DMnz_ex;

    double *Hv_ref = calloc(DMnd*ncol, sizeof(double));
    assert(Hv_ref != NULL);

    double *Veff_loc = calloc(DMnd, sizeof(double));
    assert(Veff_loc != NULL);

    extract_subgrid(
        ESPRC_k->Veff_loc_dmcomm, E_k->nx_ex_d, E_k->ny_ex_d, E_k->nz_ex_d,
        E_k->DMVert[0], E_k->DMVert[1], E_k->DMVert[2],
        E_k->DMVert[3], E_k->DMVert[4], E_k->DMVert[5],
        Veff_loc, DMnx, DMny, DMnz,
        0, DMnx-1, 0, DMny-1, 0, DMnz-1
    );

    // borrow ESPRC_k for E_k_tilde for element E_k
    // Note: need to set BC to Dirichlet for local Hamiltonian!
    int EBCs_t[3] = {ESPRC_k->BCx, ESPRC_k->BCy, ESPRC_k->BCz};
    ESPRC_k->BCx = 1;
    ESPRC_k->BCy = 1;
    ESPRC_k->BCz = 1;
    // if there's only one element in a direction, set BC to global BC
    if (DDBP_info->Nex == 1) ESPRC_k->BCx = DDBP_info->BCx;
    if (DDBP_info->Ney == 1) ESPRC_k->BCy = DDBP_info->BCy;
    if (DDBP_info->Nez == 1) ESPRC_k->BCz = DDBP_info->BCz;
    int DMVert_dmcomm[6] = {0,DMnx-1,0,DMny-1,0,DMnz-1};
    Hamiltonian_loc_vectors_mult(
        ESPRC_k, DMnd, DMVert_dmcomm, Veff_loc, ncol, 0.0,
        v, Hv_ref, ESPRC_k->dmcomm
    );
    ESPRC_k->BCx = EBCs_t[0];
    ESPRC_k->BCy = EBCs_t[1];
    ESPRC_k->BCz = EBCs_t[2];

    //::debug Compare results obtained from 2 ways
    double err = 0.0;
    for (int n = 0; n < ncol; n++) {
        for (int k = 0; k < DMnz; k++) {
            for (int j = 0; j < DMny; j++) {
                for (int i = 0; i < DMnx; i++) {
                    int ip = i + FDn;
                    int jp = j + FDn;
                    int kp = k + FDn;
                    int ind = n * DMnd + (k * DMny + j) * DMnx + i;
                    int ind_ex = n * DMnd_ex + (kp * DMny_ex + jp) * DMnx_ex + ip;
                    err = max(err, fabs(Hv_ref[ind]-Hv[ind]));
                    // printf("v[%3d] = %14.10f, Hv[%3d] = %14.10f, Hv_ref[%3d] = %14.10f, Hv_ex[%3d] = %14.10f\n",
                        // ind,v[ind],ind,Hv[ind],ind,Hv_ref[ind],ind_ex,Hv_ex[ind_ex]);
                }
            }
        }
    }

    free(Veff_loc);
    free(Hv_ref);

    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Hv err = %.3e\n", err);
    assert(err < 1e-14);
}


/**
 * @brief   Start the non-blocking data transfer (halo exchange) between neighbor
 *          elements.
 *
 * @param sendbuf  The buffer array with data that will be sent out.
 * @param recvbuf  The buffer array which will be used to receive data.
 * @param kptcomm  The global kptcomm that includes all the elemcomm's.
 */
void element_haloX_Hvk(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, int ncol, const double *sendbuf,
    double *recvbuf, MPI_Comm kptcomm
)
{
    // move this to initialization
    // set up send and recv parameters
    // set_haloX_Hvk_params(DDBP_info, E_k, ncol, kptcomm);
    // print_haloX(E_k, &E_k->haloX_Hv, kptcomm);

    // init the element halo exchange of Hvk data
    // *perform all the data transfer through MPI_Isend, including local ones
    DDBP_element_Ineighbor_alltoallv(&E_k->haloX_Hv, sendbuf, recvbuf, kptcomm);

    // *only perform the data transfer that are across processes, do the local copying separately
    // DDBP_remote_element_Ineighbor_alltoallv(&E_k->haloX_Hv, sendbuf, recvbuf, kptcomm);
}



/**
 * @brief   Calculate the diagonal blocks of the local part of the DDBP Hamiltonian.
 *
 *          The diagonal blocks of the local part of DDBP Hailtonian is given by
 *                 H_ddbp_kk := vk^T * Hvk + beta * H_ddbp_kk,
 *          where H_loc := -1/2 D^2 + Veff is the global local Hamiltonian, vk is
 *          the DDBP basis functions.
 */
void calculate_H_DDBP_diag_block(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, double *v, double *Hv, double beta)
{
    DDBP_HAMILT_ERBLKS *H_DDBP_Ek = &E_k->H_DDBP_Ek;
    int Nd = E_k->nd_d;
    int nALB = E_k->nALB;

    // double *h_kk = (double *)malloc(nALB * nALB * sizeof(double));
    double *h_kk = H_DDBP_Ek->h_kj[6];
    assert(h_kk != NULL);

#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int deschij[9];
    int ZERO = 0, info;
    int mb = max(1, nALB);
    int nb = max(1, nALB);
    int llda = max(1, nALB);
    descinit_(deschij, &nALB, &nALB, &mb, &nb, &ZERO, &ZERO, &DDBP_info->ictxt_blacs, &llda, &info);

    // v' * Hv in a subcomm within blascomm
    // TODO: if no basis paral., skip this and do a local dgemm!
    pdgemm_subcomm("T", "N", nALB, nALB, Nd, 1.0, v, E_k->desc_v, Hv, E_k->desc_v,
        beta, h_kk, deschij, DDBP_info->blacscomm, best_max_nproc(Nd, nALB, "pdgemm"));
#else
    // TODO: implement corresponding routine without MKL/ScaLAPACK
    assert(0);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

    if (DDBP_info->npdm > 1) { // in the current implementation, this won't happen
        MPI_Allreduce(MPI_IN_PLACE, h_kk, nALB*nALB, MPI_DOUBLE, MPI_SUM, DDBP_info->dmcomm);
    }

    if (DDBP_info->npbasis > 1)
        MPI_Bcast(h_kk, nALB*nALB, MPI_DOUBLE, 0, DDBP_info->blacscomm);

#ifdef DEBUG_DOUBLE_CHECK
    // print the block to check
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) {
        void show_mat(double *array, int m, int n);
        printf("E_%d\n",E_K->index);
        show_mat(h_kk, nALB, nALB);
    }
#endif

}



/**
 * @brief   Copy the Hvj value for element j after receiving the data from the neighbor
 *          which contains element j. The data is added to the existing value in Hvj.
 *
 * @param haloX   The halo exchange info object.
 * @param recvbuf Receive buffer.
 * @param Hvj     The array to be written, where we store the Hvj data (other part filled by 0).
 * @param ncol    Number of columns of Hvj data.
 * @param DMnx    Number of nodes of element k (not j!) in the x dir.
 * @param DMny    Number of nodes of element k (not j!) in the y dir.
 * @param DMnz    Number of nodes of element k (not j!) in the z dir.
 * @param dim     Which direction (0 - x dir, 1 - y dir, 2 - z dir).
 * @param face    Which face of the domain (0 - left, 1 - right).
 */
void copy_buf_to_Hv(
    const haloX_t *haloX, const double* recvbuf, double *Hvj, int ncol,
    int DMnx, int DMny, int DMnz, int dim, int face)
{
    int nbr_i = 2 * dim + face;
    // copy Hvj value from recvbuf
    int is_recv = haloX->isrecv[nbr_i];
    int ie_recv = haloX->ierecv[nbr_i];
    int js_recv = haloX->jsrecv[nbr_i];
    int je_recv = haloX->jerecv[nbr_i];
    int ks_recv = haloX->ksrecv[nbr_i];
    int ke_recv = haloX->kerecv[nbr_i];
    int nx_recv = ie_recv - is_recv + 1;
    int ny_recv = je_recv - js_recv + 1;
    int nz_recv = ke_recv - ks_recv + 1;
    int nd_recv = nx_recv * ny_recv * nz_recv;
    int nrow = DMnx * DMny * DMnz;
    for (int n = 0; n < ncol; n++) {
        const double *recvbuf_n = recvbuf + haloX->rdispls[nbr_i] + nd_recv * n;
        double *Hvj_n = Hvj + nrow * n;
        sum_subgrid(1.0, recvbuf_n, nx_recv, ny_recv, nz_recv,
            0, nx_recv-1, 0, ny_recv-1, 0, nz_recv-1,
            Hvj_n, DMnx, DMny, DMnz,
            is_recv, ie_recv, js_recv, je_recv, ks_recv, ke_recv);
    }
}



/**
 * @brief   Calculate the diagonal blocks of the local part of the DDBP Hamiltonian.
 *
 *          The off diagonal blocks of the local part of DDBP Hailtonian is given by
 *                        H_ddbp_kj := vk^T * H_loc * vj, (j /= k)
 *          where H_loc := -1/2 D^2 + Veff is the global local Hamiltonian, vk is
 *          the DDBP basis functions assigned to element k.
 *
 */
void calculate_H_DDBP_offdiag_blocks(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, haloX_t *haloX, double *v, double *recvbuf)
{
// TODO: Potential improvement: instead of copying the nonzero part of Hvj into the
// TODO: whole matrix with other parts filled with 0, another way is to extract the
// TODO: corresponding locations of vk into a sperate dense vector, and multiply
// TODO: vk(m:n,:)' Hvj(m:n,:), which will be of MUCH smaller size.
// TODO: If one chooses to do this, be careful when there are 2 elements in a dir.
// TODO: In that case, the left and right neighbor are the same element, so one
// TODO: would need to combine the nonzeros (sum the values if they overlap).
    DDBP_HAMILT_ERBLKS *H_DDBP_Ek = &E_k->H_DDBP_Ek;
    // global matrix size
    int Nd = E_k->nd_d;
    int nALB = E_k->nALB;
    // local matrix size
    int DMnx = E_k->nx_d;
    int DMny = E_k->ny_d;
    int DMnz = E_k->nz_d;
    int nrow = E_k->nd_d;
    int ncol = DDBP_info->n_basis_basiscomm;
    // descriptor for storing hij blocks
    int deschkj[9];
#if defined(USE_MKL) || defined(USE_SCALAPACK)
    int ZERO = 0, info;
    int mb = max(1, nALB);
    int nb = max(1, nALB);
    int llda = max(1, nALB);
    descinit_(deschkj, &nALB, &nALB, &mb, &nb, &ZERO, &ZERO,
        &DDBP_info->ictxt_blacs, &llda, &info);
#else
    // TODO: implement corresponding action without MKL/ScaLAPACK
    assert(0);
#endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)
    int BCs[3] = {DDBP_info->BCx,DDBP_info->BCy,DDBP_info->BCz};
    int Nes[3] = {DDBP_info->Nex,DDBP_info->Ney,DDBP_info->Nez};
    for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
        // which direction
        int dim = nbr_i / 2;
        // which face of neighbor, 0:left, 1:right
        int face = nbr_i % 2;

        // *if there's only one element in this direction, skip this direction
        if (Nes[dim] == 1) continue;

        // *if there're two elements in this direction, there's only one neighbor
        // *element. For PBC, we can merge the non-zeros and skip any of the
        // *neighbors. For DBC, we'll have to skip the corresponding neighbor,
        // *whether left or right depends on the element coordinates.
        if (Nes[dim] == 2) { // coord can only be 0 or 1 in this case
            if (E_k->coords[dim] == face)
                continue;
        }

        // double *h_kj = (double *)malloc(nALB * nALB * sizeof(double));
        double *h_kj = H_DDBP_Ek->h_kj[nbr_i];
        assert(h_kj != NULL);

        double *Hvj = (double *)calloc(nrow * ncol, sizeof(double));
        assert(Hvj != NULL);

        // copy Hvj value from recvbuf
        copy_buf_to_Hv(haloX, recvbuf, Hvj, ncol, DMnx, DMny, DMnz, dim, face);

        // copy the other face of the Hvj value from recvbuf
        // *Note: if the Hvj position already has a value, we add the new value to it
        if (Nes[dim] == 2) { // combine the two neighbors since they're the same element
            copy_buf_to_Hv(haloX, recvbuf, Hvj, ncol, DMnx, DMny, DMnz, dim, 1-face);
        }

#ifdef DEBUG_DOUBLE_CHECK
        double *Hvj_ref = (double *)calloc(nrow * ncol, sizeof(double));
        assert(Hvj_ref != NULL);
        // copy Hvj value from recvbuf
        int is_recv = haloX->isrecv[nbr_i];
        int ie_recv = haloX->ierecv[nbr_i];
        int js_recv = haloX->jsrecv[nbr_i];
        int je_recv = haloX->jerecv[nbr_i];
        int ks_recv = haloX->ksrecv[nbr_i];
        int ke_recv = haloX->kerecv[nbr_i];
        int nx_recv = ie_recv - is_recv + 1;
        int ny_recv = je_recv - js_recv + 1;
        int nz_recv = ke_recv - ks_recv + 1;
        int nd_recv = nx_recv * ny_recv * nz_recv;

        for (int n = 0; n < ncol; n++) {
            double *recvbuf_n = recvbuf + haloX->rdispls[nbr_i] + nd_recv * n;
            double *Hvj_n = Hvj_ref + nrow * n;
            extract_subgrid(recvbuf_n, nx_recv, ny_recv, nz_recv,
                0, nx_recv-1, 0, ny_recv-1, 0, nz_recv-1,
                Hvj_n, DMnx, DMny, DMnz,
                is_recv, ie_recv, js_recv, je_recv, ks_recv, ke_recv);
        }

        double err = 0.0;
        for (int i = 0; i < nrow*ncol; i++) {
            err = max(err, fabs(Hvj_ref[i] - Hvj[i]));
        }
        printf("Hvj err = %.2e\n", err);
        assert(err < 1e-14);
        free(Hvj_ref);
#endif

        // vk' * Hvj in a subcomm within blascomm
        // TODO: if no basis paral., skip this and do a local dgemm!
        pdgemm_subcomm("T", "N", nALB, nALB, Nd, 1.0, v, E_k->desc_v, Hvj, E_k->desc_v,
            0.0, h_kj, deschkj, DDBP_info->blacscomm, best_max_nproc(Nd, nALB, "pdgemm"));

        if (DDBP_info->npdm > 1) { // in the current implementation, this won't happen
            MPI_Allreduce(MPI_IN_PLACE, h_kj, nALB*nALB, MPI_DOUBLE, MPI_SUM, DDBP_info->dmcomm);
        }

        if (DDBP_info->npbasis > 1)
            MPI_Bcast(h_kj, nALB*nALB, MPI_DOUBLE, 0, DDBP_info->blacscomm);

        free(Hvj);
        // free(h_kj);
    }
}



/**
 * @brief   Calculate the local part of the DDBP Hamiltonian.
 *
 *          The local part of DDBP Hailtonian is defined by
 *                        H_ddbp := V^T * H_loc * V,
 *          where H_loc := -1/2 D^2 + Veff is the global local Hamiltonian, V is
 *          the DDBP basis functions (a block-diagonal matrix).
 */
void Calculate_local_DDBP_Hamiltonian(
    SPARC_OBJ *pSPARC, int count, int kpt, int spn_i
)
{
// ! TODO: instead of giving pSPARC, simply use DDBP_info and kptcomm, since that's all required from pSPARC
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start local DDBP Hamiltonian calculation ... \n");
    #endif

    double t1,t2;
    double t_cpy_buf = 0.0, t_H_loc_vk = 0.0, t_HaloX = 0.0, t_h_ii = 0.0,
        t_h_ij = 0.0, t_malloc_free = 0.0;

    // perform H*vk_ex, and transfer Hvk_ex between elements
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;
        int nrow = E_k->nd_d;
        int ncol = DDBP_info->n_basis_basiscomm;
        int nkpt = ESPRC_k->Nkpts_kptcomm;
        int size_s = nrow * ncol * nkpt;
        int DMnx_ex = E_k->nx_d + ESPRC_k->order;
        int DMny_ex = E_k->ny_d + ESPRC_k->order;
        int DMnz_ex = E_k->nz_d + ESPRC_k->order;
        int DMnd_ex = DMnx_ex * DMny_ex * DMnz_ex;
        t1 = MPI_Wtime();
        double *Hv_ex = calloc(ncol * DMnd_ex, sizeof(*Hv_ex));
        assert(Hv_ex != NULL);
        t2 = MPI_Wtime();
        t_malloc_free += t2 - t1;

        t1 = MPI_Wtime();
        // first find H_loc * vk = (-1/2 D^2 + Veff) * vk
        H_loc_vk_mult(DDBP_info, E_k, ncol, E_k->v + spn_i*size_s, Hv_ex);
        t2 = MPI_Wtime();
        t_H_loc_vk += t2 - t1;

        int FDn = ESPRC_k->order / 2;
        int DMnx = E_k->nx_d;
        int DMny = E_k->ny_d;
        int DMnz = E_k->nz_d;
        int buf_sz = 2 * FDn * (DMnx*DMny + DMnx*DMnz + DMny*DMnz) * ncol;
        t1 = MPI_Wtime();
        E_k->sendbuf = (double *)calloc(buf_sz, sizeof(double));
        assert(E_k->sendbuf != NULL);
        t2 = MPI_Wtime();
        t_malloc_free += t2 - t1;

        t1 = MPI_Wtime();
        // copy Hv_ex to Hv, as well as sendbuf
        copy_Hv_ex_to_buf(DDBP_info, E_k, ncol, Hv_ex);

        #ifdef DEBUG_DOUBLE_CHECK
        check_Hv(DDBP_info, E_k, ncol, E_k->v + spn_i*size_s, E_k->Hv, Hv_ex);
        #endif
        t2 = MPI_Wtime();
        t_cpy_buf += t2 - t1;

        t1 = MPI_Wtime();
        free(Hv_ex);
        // communicate the Hv data to/from neighbohr elements, non-blocking
        E_k->recvbuf = (double *)calloc(buf_sz, sizeof(double));
        assert(E_k->recvbuf != NULL);
        t2 = MPI_Wtime();
        t_malloc_free += t2 - t1;

        t1 = MPI_Wtime();
        // init the element halo exchange of Hv data
        element_haloX_Hvk(
            DDBP_info, E_k, ncol, E_k->sendbuf, E_k->recvbuf, pSPARC->kptcomm
        );
        t2 = MPI_Wtime();
        t_HaloX += t2 - t1;
    }

    // ! Warning: this is not required if local halo exchange is done through MPI_Isend
    // go over the elments again to perform local halo exchange
    // for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
    //     DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
    //     SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    //     haloX_t *haloX = &E_k->haloX_Hv;
    //     t1 = MPI_Wtime();
    //     DDBP_local_element_Ineighbor_alltoallv(
    //         DDBP_info, E_k, haloX, E_k->sendbuf, E_k->recvbuf, pSPARC->kptcomm
    //     );
    //     t2 = MPI_Wtime();
    //     t_HaloX += t2 - t1;
    // }

    // while halo exhange is going on, we first compute the local vk' H vk,
    // which doesn't depend on the data from other elements, this overlaps
    // the communication with computation
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;
        int nrow = E_k->nd_d;
        int ncol = DDBP_info->n_basis_basiscomm;
        int nkpt = ESPRC_k->Nkpts_kptcomm;
        int size_s = nrow * ncol * nkpt;
        double beta = 0.0;
        t1 = MPI_Wtime();
        calculate_H_DDBP_diag_block(DDBP_info, E_k, E_k->v + spn_i*size_s, E_k->Hv, beta);
        t2 = MPI_Wtime();
        t_h_ii += t2 - t1;
    }

    // go over the elments again to compute the off diagonal blocks vk' H vj (j/=k)
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;
        int nrow = E_k->nd_d;
        int ncol = DDBP_info->n_basis_basiscomm;
        int nkpt = ESPRC_k->Nkpts_kptcomm;
        int size_s = nrow * ncol * nkpt;
        haloX_t *haloX = &E_k->haloX_Hv;

        t1 = MPI_Wtime();
        // wait for the element halo exchange of Hv data to be completed here
        MPI_Waitall(2*haloX->n_neighbors, haloX->requests, MPI_STATUS_IGNORE);
        t2 = MPI_Wtime();
        t_HaloX += t2 - t1;

        t1 = MPI_Wtime();
        calculate_H_DDBP_offdiag_blocks(
            DDBP_info, E_k, haloX, E_k->v + spn_i*size_s, E_k->recvbuf
        );
        t2 = MPI_Wtime();
        t_h_ij += t2 - t1;

        free(E_k->Hv);
        free(E_k->sendbuf);
        free(E_k->recvbuf);
    }

    #ifdef DEBUG
    if (rank_t == 0) {
        printf("== Construct H_DDBP ==: Total time for H_loc*vk: %.3f ms\n",t_H_loc_vk*1e3);
        printf("== Construct H_DDBP ==: Total time for packing buffer: %.3f ms\n",t_cpy_buf*1e3);
        printf("== Construct H_DDBP ==: Total time for haloX: %.3f ms\n",t_HaloX*1e3);
        printf("== Construct H_DDBP ==: Total time for vk' H vk: %.3f ms\n",t_h_ii*1e3);
        printf("== Construct H_DDBP ==: Total time for vk' H vj: %.3f ms\n",t_h_ij*1e3);
        printf("== Construct H_DDBP ==: Total time for malloc/free: %.3f ms\n",t_malloc_free*1e3);
    }
    #endif
}



/**
 * @brief   Calculate DDBP Hamiltonian.
 *
 *          The DDBP Hailtonian is defined by
 *                        H_ddbp := V^T * H * V,
 *          where H := -1/2 D^2 + Veff + Vnl is the global Hamiltonian, V is the
 *          DDBP basis (a block-diagonal matrix).
 */
void Calculate_DDBP_Hamiltonian(SPARC_OBJ *pSPARC, int count, int kpt, int spn_i)
{
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
    if (DDBP_info->elemcomm_index == -1 || DDBP_info->basiscomm_index == -1 ||
        DDBP_info->dmcomm == MPI_COMM_NULL) {
        return;
    }

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start DDBP Hamiltonian calcuation ... \n");
    #endif

    double t_h_ddbp_loc = 0.0, t_h_ddbp_nloc = 0.0;

    double t1,t2;
    t1 = MPI_Wtime();
    Calculate_local_DDBP_Hamiltonian(pSPARC, count, kpt, spn_i);
    t2 = MPI_Wtime();
    t_h_ddbp_loc += t2 - t1;

    t1 = MPI_Wtime();
    Calculate_nloc_projectors_DDBP_Hamiltonian(pSPARC, count, kpt, spn_i);
    t2 = MPI_Wtime();
    t_h_ddbp_nloc += t2 - t1;

    #ifdef DEBUG
    if (rank_t == 0) {
        printf("--------------------------------------------------------------\n");
        printf("== Construct H_DDBP ==: Construction of H_DDBP_loc: %.3f ms\n",t_h_ddbp_loc*1e3);
        printf("== Construct H_DDBP ==: Construction of Vnl_DDBP: %.3f ms\n",t_h_ddbp_nloc*1e3);
    }
    #endif
}



/**
 * @brief   Calculate the local part of the global Hamiltonian multiplied by the
 *          basis functions vk in element k. I.e., find
 *                             Hv = diag(H_diag) * vk,
 *          where diag(H_diag) is a diagonal matrix corresponding to element E_k.
 */
void H_diag_vk_mult(
    int Nd_k, const double *H_diag, int ncol, const double *v, double *Hv
)
{
    // find Hv = diag(H_diag) * v
    for (int n = 0; n < ncol; n++) {
        const double *v_n = v + n * Nd_k;
        double *Hv_n = Hv + n * Nd_k;
        for (int i = 0; i < Nd_k; i++) {
            Hv_n[i] = H_diag[i] * v_n[i];
        }
    }
}



/**
 * @brief   Update Veff part of DDBP Hamiltonian.
 *
 *          The DDBP Hailtonian is defined by
 *                        H_ddbp := V^T * H * V,
 *          where H := -1/2 D^2 + Veff + Vnl is the global Hamiltonian, V is the
 *          DDBP basis (a block-diagonal matrix).
 * 
 *          When the basis is not updated, only Veff part in H is changed. We rewrite
 *          H_ddbp as follows:
 *                H_ddbp = V^T * (-1/2 D^2 + Vnl) * V + V^T * Veff * V.
 *          The first term is fixed if V is not updated, whereas the second part, due
 *          to the change of Veff, needs to be updated.
 * 
 *          This routine thus does the following:
 *                H_ddbp_new = H_ddbp + V^T * (Veff_new - Veff_old) * V.
 */
void Update_Veff_part_of_DDBP_Hamiltonian(SPARC_OBJ *pSPARC, int count, int kpt, int spn_i)
{
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
    if (DDBP_info->elemcomm_index == -1 || DDBP_info->basiscomm_index == -1 ||
        DDBP_info->dmcomm == MPI_COMM_NULL) {
        return;
    }

    #ifdef DEBUG
    int rank_t;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_t);
    if (rank_t == 0) printf("Start updating Veff part of DDBP Hamiltonian calcuation ... \n");
    #endif

    double t1,t2;
    double t_cpy_buf = 0.0, t_H_loc_vk = 0.0, t_HaloX = 0.0, t_h_ii = 0.0,
        t_h_ij = 0.0, t_malloc_free = 0.0;

    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;
        int nrow = E_k->nd_d;
        int ncol = DDBP_info->n_basis_basiscomm;
        int nkpt = ESPRC_k->Nkpts_kptcomm;
        int size_s = nrow * ncol * nkpt;
        int DMnx = E_k->nx_d;
        int DMny = E_k->ny_d;
        int DMnz = E_k->nz_d;
        int DMnd = DMnx * DMny * DMnz;

        t1 = MPI_Wtime();
        double *Hv = calloc(ncol * DMnd, sizeof(*Hv));
        assert(Hv != NULL);
        double *Veff_diff_k = malloc(DMnd * sizeof(double));
        assert(Veff_diff_k != NULL);
        t2 = MPI_Wtime();
        t_malloc_free += t2 - t1;

        // find Veff_new - Veff_old
        // Veff_diff_k = Veff_new(isInEk)
        extract_subgrid(
            ESPRC_k->Veff_loc_dmcomm, E_k->nx_ex_d, E_k->ny_ex_d, E_k->nz_ex_d,
            E_k->DMVert[0],E_k->DMVert[1],E_k->DMVert[2],
            E_k->DMVert[3],E_k->DMVert[4],E_k->DMVert[5],
            Veff_diff_k, DMnx, DMny, DMnz,
            0, DMnx-1, 0, DMny-1, 0, DMnz-1
        );

        double *Veff_old = E_k->Veff_loc_dmcomm_prev;
        // Veff_diff_k += -1.0 * Veff_old(isInEk)
        sum_subgrid(-1.0, Veff_old, E_k->nx_ex_d, E_k->ny_ex_d, E_k->nz_ex_d,
            E_k->DMVert[0],E_k->DMVert[1],E_k->DMVert[2],
            E_k->DMVert[3],E_k->DMVert[4],E_k->DMVert[5],
            Veff_diff_k, DMnx, DMny, DMnz,
            0, DMnx-1, 0, DMny-1, 0, DMnz-1
        );

        t1 = MPI_Wtime();
        // first find Veff_diff_k * vk
        H_diag_vk_mult(DMnd, Veff_diff_k, ncol, E_k->v + spn_i*size_s, Hv);
        t2 = MPI_Wtime();
        t_H_loc_vk += t2 - t1;

        t1 = MPI_Wtime();
        double beta = 1.0; // add the changed part to the diagonal blocks
        calculate_H_DDBP_diag_block(DDBP_info, E_k, E_k->v + spn_i*size_s, Hv, beta);
        t2 = MPI_Wtime();
        t_h_ii += t2 - t1;

        t1 = MPI_Wtime();
        free(Hv);
        free(Veff_diff_k);
        t2 = MPI_Wtime();
        t_malloc_free += t2 - t1;
    }

    #ifdef DEBUG
    if (rank_t == 0) {
        printf("== Construct H_DDBP ==: Total time for H_loc*vk: %.3f ms\n",t_H_loc_vk*1e3);
        printf("== Construct H_DDBP ==: Total time for packing buffer: %.3f ms\n",t_cpy_buf*1e3);
        printf("== Construct H_DDBP ==: Total time for haloX: %.3f ms\n",t_HaloX*1e3);
        printf("== Construct H_DDBP ==: Total time for vk' H vk: %.3f ms\n",t_h_ii*1e3);
        printf("== Construct H_DDBP ==: Total time for vk' H vj: %.3f ms\n",t_h_ij*1e3);
        printf("== Construct H_DDBP ==: Total time for malloc/free: %.3f ms\n",t_malloc_free*1e3);
    }
    #endif
}
