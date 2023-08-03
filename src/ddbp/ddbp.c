/**
 * @file    ddbp.h
 * @brief   This file contains the function declarations for the Discrete
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

extern double t_haloX;
extern double t_densmat;
extern double t_nloc;
extern double t_malloc;

/**
 * @brief   Find the neighbors of a DDBP element.
 *
 * @param E_k    Pointer to the element structure.
 * @param E_dims Number of elements in each dimension.
 */
void find_DDBP_neighbor_list(
    DDBP_ELEM *E_k, const int E_dims[3], const int BCs[3])
{
    const int ii_E = E_k->coords[0];
    const int jj_E = E_k->coords[1];
    const int kk_E = E_k->coords[2];
    // const int BCx = BCs[0];
    // const int BCy = BCs[1];
    // const int BCz = BCs[2];
    const int Nex = E_dims[0];
    const int Ney = E_dims[1];
    const int Nez = E_dims[2];

    int k = E_k->index;

    // TODO: consider what to do for non-orthogonal case

    int *nbhd_list = malloc(30 * sizeof(*nbhd_list));
    int n_nbhd = 0;
    nbhd_list[n_nbhd++] = k; // first include the element itself

    // append the neighors in x dir
    if (Nex > 1) {
        int coords_nbhd_l[3] = {(ii_E-1+Nex) % Nex,jj_E,kk_E}; // TODO: verify what to do for DBC
        int coords_nbhd_r[3] = {(ii_E+1) % Nex,jj_E,kk_E}; // TODO: verify what to do for DBC

        int index_nbhd_l, index_nbhd_r;
        DDBP_Cart_Index(E_dims, coords_nbhd_l, &index_nbhd_l);
        DDBP_Cart_Index(E_dims, coords_nbhd_r, &index_nbhd_r);
        int nbhd_list_1d[2] = {index_nbhd_l, index_nbhd_r};
        int n_nbhd_1d = unique(nbhd_list_1d, 2);
        for (int i = 0; i < n_nbhd_1d; ++i) {
            nbhd_list[n_nbhd++] = nbhd_list_1d[i];
        }
    }
    // append the neighors in y dir
    if (Ney > 1) {
        int coords_nbhd_l[3] = {ii_E,(jj_E-1+Ney) % Ney,kk_E}; // TODO: verify what to do for DBC
        int coords_nbhd_r[3] = {ii_E,(jj_E+1) % Ney,kk_E}; // TODO: verify what to do for DBC

        int index_nbhd_l, index_nbhd_r;
        DDBP_Cart_Index(E_dims, coords_nbhd_l, &index_nbhd_l);
        DDBP_Cart_Index(E_dims, coords_nbhd_r, &index_nbhd_r);
        int nbhd_list_1d[2] = {index_nbhd_l, index_nbhd_r};
        int n_nbhd_1d = unique(nbhd_list_1d, 2);
        for (int i = 0; i < n_nbhd_1d; ++i) {
            nbhd_list[n_nbhd++] = nbhd_list_1d[i];
        }
    }
    // append the neighors in z dir
    if (Nez > 1) {
        int coords_nbhd_l[3] = {ii_E,jj_E,(kk_E-1+Nez) % Nez}; // TODO: verify what to do for DBC
        int coords_nbhd_r[3] = {ii_E,jj_E,(kk_E+1) % Nez}; // TODO: verify what to do for DBC

        int index_nbhd_l, index_nbhd_r;
        DDBP_Cart_Index(E_dims, coords_nbhd_l, &index_nbhd_l);
        DDBP_Cart_Index(E_dims, coords_nbhd_r, &index_nbhd_r);
        int nbhd_list_1d[2] = {index_nbhd_l, index_nbhd_r};
        int n_nbhd_1d = unique(nbhd_list_1d, 2);
        for (int i = 0; i < n_nbhd_1d; ++i) {
            nbhd_list[n_nbhd++] = nbhd_list_1d[i];
        }
    }

    n_nbhd = unique(nbhd_list, n_nbhd);
    nbhd_list = realloc(nbhd_list, n_nbhd*sizeof(*nbhd_list));

    E_k->element_nbhd_list = nbhd_list;
    E_k->n_element_nbhd = n_nbhd;
}



/**
 * @brief   Create a SPARC_OBJ for the given element.
 */
void create_element_SPARC_obj(SPARC_OBJ *pSPARC, DDBP_ELEM *E_k)
{
    E_k->ESPRC = malloc(sizeof(*E_k->ESPRC));
    SPARC_OBJ *ESPRC_k = E_k->ESPRC;

    // copy the data from global SPARC object (shallow copy) to element SPARC object
    // since it's a shallow copy, pointers are directly pointed to the same memory
    *ESPRC_k = *pSPARC;

    // TODO: set FD order to a different value if necessary, note that if FD orders
    // are set differently, the stencil weights must be recalculated too

    // customize the element SPARC object
    ESPRC_k->BCx = E_k->EBCx;
    ESPRC_k->BCy = E_k->EBCy;
    ESPRC_k->BCz = E_k->EBCz;

    ESPRC_k->range_x = E_k->xe_ex_sg - E_k->xs_ex_sg;
    ESPRC_k->range_y = E_k->ye_ex_sg - E_k->ys_ex_sg;
    ESPRC_k->range_z = E_k->ze_ex_sg - E_k->zs_ex_sg;

    ESPRC_k->Nx = E_k->nx_ex;
    ESPRC_k->Ny = E_k->ny_ex;
    ESPRC_k->Nz = E_k->nz_ex;
    ESPRC_k->Nd = ESPRC_k->Nx * ESPRC_k->Ny * ESPRC_k->Nz;

    ESPRC_k->Nstates = E_k->nALB;

    // atoms need to be found for every MD, so will do it later

}



/**
 * @brief   Assign atoms to the elements and extended elements.
 */
void ddbp_assign_atoms(
    const double *atom_pos, const int n_atom, const double cell[3],
    const int BCs[3], DDBP_INFO *DDBP_info)
{
    double buf = 2.0; // buffer for deciding influencing range of an atom
    // go over each element assigned to this process and search through all atoms
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;

        // make sure the lists are empty before we start
        clear_dyarray(&E_k->atom_list);
        clear_dyarray(&E_k->atom_list_ex);
        assert(E_k->atom_list.len == 0);
        assert(E_k->atom_list_ex.len == 0);

        double vert[6];
        vert[0] = E_k->xs; vert[1] = E_k->xe;
        vert[2] = E_k->ys; vert[3] = E_k->ye;
        vert[4] = E_k->zs; vert[5] = E_k->ze;
        double vert_ex[6];
        vert_ex[0] = E_k->xs_ex_sg; vert_ex[1] = E_k->xe_ex_sg;
        vert_ex[2] = E_k->ys_ex_sg; vert_ex[3] = E_k->ye_ex_sg;
        vert_ex[4] = E_k->zs_ex_sg; vert_ex[5] = E_k->ze_ex_sg;

        // E_k->n_atom = 0;
        // E_k->n_atom_ex = 0;
        // printf("E[%d]->vert = [%f,%f,%f,%f,%f,%f]\n",E_k->index,
        //     vert[0],vert[1],vert[2],vert[3],vert[4],vert[5]);
        // printf("E[%d]->vert_ex = [%f,%f,%f,%f,%f,%f]\n",E_k->index,
        //     vert_ex[0],vert_ex[1],vert_ex[2],vert_ex[3],vert_ex[4],vert_ex[5]);

        // check if the atom is within this element
        for (int i = 0; i < n_atom; i++) {
            if (is_atom_in_region(&atom_pos[3*i], vert)) {
                // printf("Bingo! Atom in Element!\n");
                append_dyarray(&E_k->atom_list, i);
            }
        }
        E_k->n_atom = E_k->atom_list.len;

        // find which atoms have nonlocal influence on the element (entire element)
        int n_atom_ex = 0;
        for (int J = 0; J < n_atom; J++) {
            // int ityp = DDBP_info->atom_types[i];
            double rcbox[3] = {0.0,0.0,0.0};
            // TODO: confirm whether to include atoms not in extended element but have nloc influence!
            // TODO: including them might improve the quality of the basis (especially for EBC = 'D'),
            // TODO: but for EBC = 'P', there's a chance that the atoms might overlap with the image
            // TODO: of an already existent atom within the extended element, causing the image atom
            // TODO: to be counted twice (think about the case when buffer = 0)!
            // * Another idea is to include them only for EBC = 'D'
            rcbox[0] = DDBP_info->rcs_x[J] + buf;
            rcbox[1] = DDBP_info->rcs_y[J] + buf;
            rcbox[2] = DDBP_info->rcs_z[J] + buf;

            // printf("atom %d position = (%f,%f,%f)\n",J,atom_pos[3*J],atom_pos[3*J+1],atom_pos[3*J+2]);
            // printf("rcbox = [%.2f, %.2f, %.2f]\n",rcbox[0],rcbox[1],rcbox[2]);
            // rcbox[0] = rcbox[1] = rcbox[2] = 0.1; // for debugging
            // check if the atom (including its images) has influence in the extended element
            // these atoms will be treated as if they are independent atoms, no summation of integrals in Vnl*x
            double *image_coords;
            int nimage_in = atom_images_in_region(&atom_pos[3*J], rcbox, cell, BCs, vert_ex, &image_coords);
            free(image_coords); // memory allocation happens within atom_images_in_region
            if (nimage_in > 0) {
                append_dyarray(&E_k->atom_list_ex, J);
                n_atom_ex += nimage_in; // we count images too
            }
        }
        E_k->n_atom_ex = n_atom_ex;

        // go over the list of atoms that influence the extended atom again and save the atom+image coords
        // TODO: do it on-the-fly above using dynamic array for double array type!
        double *atom_pos_local = (double *)malloc(E_k->n_atom_ex * 3 * sizeof(double));
        assert(atom_pos_local != NULL);
        int *nAtomv = (int *)calloc(ESPRC_k->Ntypes, sizeof(int));
        assert(nAtomv != NULL);
        // find local coordinates in element E_k, note x0_local can be out of the extended element
        int count = 0;
        for (int i = 0; i < E_k->atom_list_ex.len; i++) {
            int J = E_k->atom_list_ex.array[i]; // global atom index
            int ityp = DDBP_info->atom_types[J];
            double rcbox[3] = {0.0,0.0,0.0};
            rcbox[0] = DDBP_info->rcs_x[J] + buf;
            rcbox[1] = DDBP_info->rcs_y[J] + buf;
            rcbox[2] = DDBP_info->rcs_z[J] + buf;
            double *image_coords;
            int nimage_in = atom_images_in_region(&atom_pos[3*J], rcbox, cell, BCs, vert_ex, &image_coords);
            for (int j = 0; j < nimage_in; j++) {
                // save relative coords of every image
                atom_pos_local[count*3  ] = image_coords[j*3  ] - E_k->xs_ex_sg;
                atom_pos_local[count*3+1] = image_coords[j*3+1] - E_k->ys_ex_sg;
                atom_pos_local[count*3+2] = image_coords[j*3+2] - E_k->zs_ex_sg;
                count++; // count every image
                nAtomv[ityp]++; // count the type
            }
            free(image_coords); // memory allocation happens within atom_images_in_region
        }
        assert(count == E_k->n_atom_ex);
        ESPRC_k->n_atom = E_k->n_atom_ex;
        ESPRC_k->atom_pos = atom_pos_local;
        ESPRC_k->nAtomv = nAtomv;
    }
}



/**
 * @brief   Initialize DDBP.
 */
void init_DDBP(SPARC_OBJ *pSPARC) {
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;

    int rank, nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #ifdef DEBUG
    if (rank == 0) print_DDBP_info(DDBP_info);
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    double dx = pSPARC->delta_x;
    double dy = pSPARC->delta_y;
    double dz = pSPARC->delta_z;
    int BCs[3];
    BCs[0] = pSPARC->BCx;
    BCs[1] = pSPARC->BCy;
    BCs[2] = pSPARC->BCz;
    // double cell[3];
    // cell[0] = pSPARC->range_x;
    // cell[1] = pSPARC->range_y;
    // cell[2] = pSPARC->range_z;
    int E_dims[3];
    E_dims[0] = DDBP_info->Nex;
    E_dims[1] = DDBP_info->Ney;
    E_dims[2] = DDBP_info->Nez;

    // go over each element and set up the elements
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];

        // find the global element index
        E_k->index = k + DDBP_info->elem_start_index;

        // find the global coordinates of the element
        DDBP_Index_Cart(E_dims, E_k->index, E_k->coords);
        int ii_E = E_k->coords[0];
        int jj_E = E_k->coords[1];
        int kk_E = E_k->coords[2];

        // set the element BC, currently set to the same for all elements
        E_k->EBCx = DDBP_info->EBCx;
        E_k->EBCy = DDBP_info->EBCy;
        E_k->EBCz = DDBP_info->EBCz;

        // find the corner coordinates of the element
        E_k->xs = ii_E * DDBP_info->Lex;
        E_k->xe = (ii_E + 1) * DDBP_info->Lex;
        E_k->ys = jj_E * DDBP_info->Ley;
        E_k->ye = (jj_E + 1) * DDBP_info->Ley;
        E_k->zs = kk_E * DDBP_info->Lez;
        E_k->ze = (kk_E + 1) * DDBP_info->Lez;

        // find the starting and ending grid indices of the grids in the element
        // note we don't include the end point, so that elements don't overlap
        E_k->is = ceil(E_k->xs / dx - TEMP_TOL);
        E_k->ie = ceil(E_k->xe / dx - TEMP_TOL) - 1;
        E_k->js = ceil(E_k->ys / dy - TEMP_TOL);
        E_k->je = ceil(E_k->ye / dy - TEMP_TOL) - 1;
        E_k->ks = ceil(E_k->zs / dz - TEMP_TOL);
        E_k->ke = ceil(E_k->ze / dz - TEMP_TOL) - 1;
        E_k->nx = E_k->ie - E_k->is + 1;
        E_k->ny = E_k->je - E_k->js + 1;
        E_k->nz = E_k->ke - E_k->ks + 1;
        E_k->nd = E_k->nx * E_k->ny * E_k->nz;

        // find the corner coordinates of the element (snap to grid)
        E_k->xs_sg = E_k->is * dx;
        E_k->ys_sg = E_k->js * dy;
        E_k->zs_sg = E_k->ks * dz;
        E_k->xe_sg = (E_k->ie + 1 - E_k->EBCx) * dx; // no grid here for PBC
        E_k->ye_sg = (E_k->je + 1 - E_k->EBCy) * dy; // no grid here for PBC
        E_k->ze_sg = (E_k->ke + 1 - E_k->EBCz) * dz; // no grid here for PBC

        // Extended Element //
        // set the element buffer size, currently set to the same for all elements
        E_k->buffer_x = DDBP_info->buffer_x; // buffer size in x dir
        E_k->buffer_y = DDBP_info->buffer_y; // buffer size in y dir
        E_k->buffer_z = DDBP_info->buffer_z; // buffer size in z dir

        // find the corner coordinates of the extended element
        E_k->xs_ex = E_k->xs - DDBP_info->buffer_x;
        E_k->xe_ex = E_k->xe + DDBP_info->buffer_x;
        E_k->ys_ex = E_k->ys - DDBP_info->buffer_y;
        E_k->ye_ex = E_k->ye + DDBP_info->buffer_y;
        E_k->zs_ex = E_k->zs - DDBP_info->buffer_z;
        E_k->ze_ex = E_k->ze + DDBP_info->buffer_z;

        // find the grid index of the corner coordinates of the extended element (snap to grid)
        // Caution: remember to map it back to domain for PBC if it goes out
        // Method 1 (original):
        // E_k->is_ex = ceil((E_k->xs - DDBP_info->buffer_x) / dx - TEMP_TOL);
        // E_k->ie_ex = floor((E_k->xe + DDBP_info->buffer_x) / dx + TEMP_TOL) - 1 + DDBP_info->EBCx;
        // E_k->js_ex = ceil((E_k->ys - DDBP_info->buffer_y) / dy - TEMP_TOL);
        // E_k->je_ex = floor((E_k->ye + DDBP_info->buffer_y) / dy + TEMP_TOL) - 1 + DDBP_info->EBCy;
        // E_k->ks_ex = ceil((E_k->zs - DDBP_info->buffer_z) / dz - TEMP_TOL);
        // E_k->ke_ex = floor((E_k->ze + DDBP_info->buffer_z) / dz + TEMP_TOL) - 1 + DDBP_info->EBCz;
        // Method 2:
        // E_k->is_ex = floor(E_k->xs_ex / dx + TEMP_TOL);
        // E_k->js_ex = floor(E_k->ys_ex / dy + TEMP_TOL);
        // E_k->ks_ex = floor(E_k->zs_ex / dz + TEMP_TOL);
        // E_k->ie_ex = ceil(E_k->xe_ex / dx - TEMP_TOL) - 1 + E_k->EBCx;
        // E_k->je_ex = ceil(E_k->ye_ex / dy - TEMP_TOL) - 1 + E_k->EBCy;
        // E_k->ke_ex = ceil(E_k->ze_ex / dz - TEMP_TOL) - 1 + E_k->EBCz;
        // Method 3:
        E_k->is_ex = E_k->is - ceil(E_k->buffer_x / dx - TEMP_TOL);
        E_k->ie_ex = E_k->ie + ceil(E_k->buffer_x / dx - TEMP_TOL);
        E_k->js_ex = E_k->js - ceil(E_k->buffer_y / dy - TEMP_TOL);
        E_k->je_ex = E_k->je + ceil(E_k->buffer_y / dy - TEMP_TOL);
        E_k->ks_ex = E_k->ks - ceil(E_k->buffer_z / dz - TEMP_TOL);
        E_k->ke_ex = E_k->ke + ceil(E_k->buffer_z / dz - TEMP_TOL);

        E_k->nx_ex = E_k->ie_ex - E_k->is_ex + 1;
        E_k->ny_ex = E_k->je_ex - E_k->js_ex + 1;
        E_k->nz_ex = E_k->ke_ex - E_k->ks_ex + 1;
        E_k->nd_ex = E_k->nx_ex * E_k->ny_ex * E_k->nz_ex;

        // find the corner coordinates of the extended element (snap to grid)
        E_k->xs_ex_sg = E_k->is_ex * dx;
        E_k->ys_ex_sg = E_k->js_ex * dy;
        E_k->zs_ex_sg = E_k->ks_ex * dz;
        E_k->xe_ex_sg = (E_k->ie_ex + 1 - E_k->EBCx) * dx; // no grid here for PBC
        E_k->ye_ex_sg = (E_k->je_ex + 1 - E_k->EBCy) * dy; // no grid here for PBC
        E_k->ze_ex_sg = (E_k->ke_ex + 1 - E_k->EBCz) * dz; // no grid here for PBC

        // initialize number of atoms assigned to this element/extended element
        E_k->n_atom = 0;
        E_k->n_atom_ex = 0;
        init_dyarray(&E_k->atom_list);
        init_dyarray(&E_k->atom_list_ex);

        // Number of basis functions in this element
        // Method 1: assign nALB according to #atoms in each element
        // Warning: this will create complication in the paral. of basis funcs (element band paral.) as
        //          well as slow convergence wrt #ALB.
        // E_k->nALB = 0; // calculate later based on the assigned atoms
        // Method 2: Take average
        E_k->nALB = DDBP_info->nALB_tot / DDBP_info->Ne_tot;
        // also need to find which indices these basis func's correspond to
        // in general, need to do a cumulative sum over all elements, using MPI_Scan with MPI_SUM operation
        // if we know in advance, all nALB will be the same for all elements, then we can calculate them
        E_k->ALB_ns = E_k->nALB * E_k->index;
        E_k->ALB_ne = E_k->nALB * (E_k->index + 1) - 1;

        // find neighbor element list
        find_DDBP_neighbor_list(E_k, E_dims, BCs);

        // create the element SPARC object
        create_element_SPARC_obj(pSPARC, E_k);
    }

    // assign atoms to the elements and extended elements, TODO: need to do it for every MD, move to another place!
    // ddbp_assign_atoms(pSPARC->atom_pos, pSPARC->n_atom, cell, BCs, DDBP_info);


    // print all elements
    // for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
    //     DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        // print_Element(E_k); // TODO: remove after check
        // TODO: remove after check
        // for (int ii = 0; ii < nproc; ii++) {
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     if (ii == rank) print_Element(E_k);
        // }
    // }
}



/**
 * @brief   Create the DDBP_info structure and store the relevant parameters.
 */
void Create_DDBP_info(SPARC_OBJ *pSPARC) {
    pSPARC->DDBP_info = malloc( sizeof(*(pSPARC->DDBP_info)) );
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;

    // store the DDBP input parameters in the structure
    DDBP_info->Nex = pSPARC->DDBP_Nex;
    DDBP_info->Ney = pSPARC->DDBP_Ney;
    DDBP_info->Nez = pSPARC->DDBP_Nez;
    DDBP_info->buffer_x = pSPARC->DDBP_buffer_x;
    DDBP_info->buffer_y = pSPARC->DDBP_buffer_y;
    DDBP_info->buffer_z = pSPARC->DDBP_buffer_z;
    DDBP_info->BCx = pSPARC->BCx;
    DDBP_info->BCy = pSPARC->BCy;
    DDBP_info->BCz = pSPARC->BCz;
    DDBP_info->EBCx = pSPARC->DDBP_EBCx;
    DDBP_info->EBCy = pSPARC->DDBP_EBCy;
    DDBP_info->EBCz = pSPARC->DDBP_EBCz;
    DDBP_info->nALB_atom = pSPARC->DDBP_nALB_atom;
    // TODO: add the following input options
    DDBP_info->npelem  = pSPARC->DDBP_npelem;
    DDBP_info->npbasis = pSPARC->DDBP_npbasis;
    // DDBP_info->npdm = pSPARC->DDBP_npdm;
    // DDBP_info->npelem  = -1; // TODO: remove after check
    // DDBP_info->npbasis = -1; // TODO: remove after check
    DDBP_info->npband = DDBP_info->npbasis; // TODO: remove after check
    DDBP_info->npdm = -1; // TODO: remove after check

    // copy number of states
    DDBP_info->Nstates = pSPARC->Nstates;

    // find total number of DDBP elements
    DDBP_info->Ne_tot = DDBP_info->Nex * DDBP_info->Ney * DDBP_info->Nez;

    // set finite-difference order to the same as the global problem
    DDBP_info->fd_order = pSPARC->order;

    // find the lengths of the elements in each direction
    DDBP_info->Lex = pSPARC->range_x / DDBP_info->Nex;
    DDBP_info->Ley = pSPARC->range_y / DDBP_info->Ney;
    DDBP_info->Lez = pSPARC->range_z / DDBP_info->Nez;

    DDBP_info->n_atom = pSPARC->n_atom;
    // create atom type vector and the corresponding rc box (nloc projector influence)
    DDBP_info->atom_types = (int *) malloc(DDBP_info->n_atom * sizeof(int));
    DDBP_info->rcs_x = (double *) malloc(DDBP_info->n_atom * sizeof(double));
    DDBP_info->rcs_y = (double *) malloc(DDBP_info->n_atom * sizeof(double));
    DDBP_info->rcs_z = (double *) malloc(DDBP_info->n_atom * sizeof(double));
    assert(DDBP_info->atom_types != NULL);
    assert(DDBP_info->rcs_x != NULL);
    assert(DDBP_info->rcs_y != NULL);
    assert(DDBP_info->rcs_z != NULL);
    int atmcount = 0;
    for (int ityp = 0; ityp < pSPARC->Ntypes; ityp++) {
        for (int n = 0; n < pSPARC->nAtomv[ityp]; n++) {
            double rc = 0.0, rcbox_x, rcbox_y, rcbox_z;
            // find max rc
            for (int i = 0; i <= pSPARC->psd[ityp].lmax; i++) {
                rc = max(rc, pSPARC->psd[ityp].rc[i]);
            }
            if(pSPARC->cell_typ == 0) {
                rcbox_x = rcbox_y = rcbox_z = rc;
            } else {
                rcbox_x = pSPARC->CUTOFF_x[ityp];
                rcbox_y = pSPARC->CUTOFF_y[ityp];
                rcbox_z = pSPARC->CUTOFF_z[ityp];
            }
            // save the data into the array
            DDBP_info->atom_types[atmcount] = ityp;
            DDBP_info->rcs_x[atmcount] = rcbox_x;
            DDBP_info->rcs_y[atmcount] = rcbox_y;
            DDBP_info->rcs_z[atmcount] = rcbox_z;
            atmcount++;
        }
    }

    // calculate total number of basis functions
    DDBP_info->nALB_tot = DDBP_info->nALB_atom * pSPARC->n_atom;
    // TODO: since we take average over number of elements, the total number
    //       of basis might be changed slightly
    DDBP_info->nALB_tot = ceil((double)DDBP_info->nALB_tot / DDBP_info->Ne_tot) * DDBP_info->Ne_tot;
}



/**
 * @brief   Free the nonlocal influencing atoms object.
 */
void free_nloc_influence_atoms(
    ATOM_NLOC_INFLUENCE_OBJ *AtmNloc, int Ntypes, int proc_active)
{
    if (proc_active == 0) return; // return if not active
    if (AtmNloc == NULL) return;
    // free atom influence struct components
    for (int i = 0; i < Ntypes; i++) {
        if (AtmNloc[i].n_atom > 0) {
            if (AtmNloc[i].coords != NULL) free(AtmNloc[i].coords);
            if (AtmNloc[i].atom_index != NULL) free(AtmNloc[i].atom_index);
            if (AtmNloc[i].xs != NULL) free(AtmNloc[i].xs);
            if (AtmNloc[i].xe != NULL) free(AtmNloc[i].xe);
            if (AtmNloc[i].ys != NULL) free(AtmNloc[i].ys);
            if (AtmNloc[i].ye != NULL) free(AtmNloc[i].ye);
            if (AtmNloc[i].zs != NULL) free(AtmNloc[i].zs);
            if (AtmNloc[i].ze != NULL) free(AtmNloc[i].ze);
            if (AtmNloc[i].ndc != NULL) free(AtmNloc[i].ndc);
            for (int j = 0; j < AtmNloc[i].n_atom; j++) {
                if (AtmNloc[i].grid_pos[j] != NULL) free(AtmNloc[i].grid_pos[j]);
            }
            if (AtmNloc[i].grid_pos != NULL) free(AtmNloc[i].grid_pos);
        }
    }
    free(AtmNloc);
}


/**
 * @brief   Free nonlocal objects, including the nonlocal influencing atoms
 *          and the nonlocal projectors.
 */
void free_nloc_projectors(
    const ATOM_NLOC_INFLUENCE_OBJ *AtmNloc, NLOC_PROJ_OBJ *nlocProj,
    int isGammaPoint, int Ntypes, int proc_active)
{
    if (proc_active == 0) return; // return if not active

    for (int ityp = 0; ityp < Ntypes; ityp++) {
        if (nlocProj[ityp].nproj > 0) {
            for (int iat = 0; iat < AtmNloc[ityp].n_atom; iat++) {
                isGammaPoint ? free(nlocProj[ityp].Chi[iat]) : free(nlocProj[ityp].Chi_c[iat]);
            }
        }
        isGammaPoint ? free(nlocProj[ityp].Chi) : free(nlocProj[ityp].Chi_c);
    }
    free(nlocProj);
}



/**
 * @brief   Free nonlocal objects, including the nonlocal influencing atoms
 *          and the nonlocal projectors.
 */
void free_nloc_objs(
    ATOM_NLOC_INFLUENCE_OBJ *AtmNloc, NLOC_PROJ_OBJ *nlocProj,
    int isGammaPoint, int Ntypes, int proc_active)
{
    if (proc_active == 0) return; // return if not active

    // deallocate nonlocal projectors
    free_nloc_projectors(AtmNloc, nlocProj, isGammaPoint, Ntypes, proc_active);

    // free atom influence struct components
    free_nloc_influence_atoms(AtmNloc, Ntypes, proc_active);
}


/**
 * @brief   Free the DDBP related variables that has to be reset for SCF.
 */
void free_DDBP_scfvar(SPARC_OBJ *pSPARC) {
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;

    // go over each element and set up the elements
    for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
        DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
        SPARC_OBJ *ESPRC_k = E_k->ESPRC;
        free(ESPRC_k->atom_pos);
        free(ESPRC_k->nAtomv);
        free(ESPRC_k->IP_displ);

        free_nloc_objs(ESPRC_k->Atom_Influence_nloc, ESPRC_k->nlocProj,
            ESPRC_k->isGammaPoint, ESPRC_k->Ntypes,
            (int) (ESPRC_k->dmcomm != MPI_COMM_NULL && ESPRC_k->bandcomm_index >= 0));

        free_nloc_objs(ESPRC_k->Atom_Influence_nloc_kptcomm, ESPRC_k->nlocProj_kptcomm,
            ESPRC_k->isGammaPoint, ESPRC_k->Ntypes,
            (int) (ESPRC_k->kptcomm_topo != MPI_COMM_NULL && ESPRC_k->kptcomm_index >= 0));

        DDBP_HAMILT_ERBLKS *H_DDBP_Ek = &E_k->H_DDBP_Ek;
        DDBP_VNL *Vnl_DDBP = &H_DDBP_Ek->Vnl_DDBP;
        free_nloc_objs(Vnl_DDBP->AtmNloc, Vnl_DDBP->nlocProj, ESPRC_k->isGammaPoint, ESPRC_k->Ntypes,
            (int) (DDBP_info->elemcomm != MPI_COMM_NULL && DDBP_info->elemcomm_index >= 0));

        free_nloc_objs(E_k->AtmNloc, E_k->nlocProj, pSPARC->isGammaPoint, pSPARC->Ntypes,
            (int) (DDBP_info->elemcomm != MPI_COMM_NULL && DDBP_info->elemcomm_index >= 0));
    }
}


void free_DDBP_Hamiltonian(DDBP_HAMILTONIAN *H_DDBP)
{
    for (int k = 0; k < H_DDBP->nelem; k++) {
        DDBP_HAMILT_ERBLKS *H_DDBP_Ek = H_DDBP->H_DDBP_Ek_list[k];
        for (int i = 0; i < 7; i++) {
            if (H_DDBP_Ek->h_kj[i] != NULL) free(H_DDBP_Ek->h_kj[i]);
        }
    }
    free(H_DDBP->H_DDBP_Ek_list);
}

/**
 * @brief   Free the DDBP_info structure.
 */
void free_DDBP_info(SPARC_OBJ *pSPARC) {
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
    int Nkpts = pSPARC->Nkpts_kptcomm;
    int Nspin = pSPARC->Nspin_spincomm;
    int nelem = DDBP_info->n_elem_elemcomm;
    // TODO: free any sub items of DDBP_info here ...
        free(DDBP_info->atom_types);
        free(DDBP_info->rcs_x);
        free(DDBP_info->rcs_y);
        free(DDBP_info->rcs_z);

        // free DDBP Hamiltonian
        free_DDBP_Hamiltonian(&DDBP_info->H_DDBP);
        delete_DDBP_Array(&DDBP_info->Lanczos_x0);

        // free eigenvalues of DDBP Hamiltonian
        // free(DDBP_info->lambda); // now pointed to pSPARC->lambda

        // free subspace matrices
        if (pSPARC->isGammaPoint) {
            free(DDBP_info->Hp);
            free(DDBP_info->Mp);
            free(DDBP_info->Q);
        } else {
            free(DDBP_info->Hp_kpt);
            free(DDBP_info->Mp_kpt);
            free(DDBP_info->Q_kpt);
        }

        // free DDBP KS orbitals
        // free xorb (2d array over kpts and spin)
        for(int spn_i = 0; spn_i < Nspin; spn_i++) {
            for (int kpt = 0; kpt < Nkpts; kpt++) {
                DDBP_ARRAY *X_ks = &DDBP_info->xorb[spn_i][kpt];
                delete_DDBP_Array(X_ks);
            }
            free(DDBP_info->xorb[spn_i]);
        }
        free(DDBP_info->xorb);
        // delete yorb
        delete_DDBP_Array(&DDBP_info->yorb);

        // free KS orbitals on original FD grid (over elements)
        for (int spn_i = 0; spn_i < Nspin; spn_i++) {            
            for (int kpt = 0; kpt < Nkpts; kpt++) {
                for (int k = 0; k < nelem; k++) {
                    free(DDBP_info->psi[spn_i][kpt][k]);
                }
                free(DDBP_info->psi[spn_i][kpt]);
            }
            free(DDBP_info->psi[spn_i]);
        }
        free(DDBP_info->psi);

        // free descriptor for KS orbital on original FD grid (over elements)
        for (int k = 0; k < nelem; k++) {
            free(DDBP_info->desc_psi[k]);
        }
        free(DDBP_info->desc_psi);

        // free electron densities (over elements)
        for (int k = 0; k < nelem; k++) {
            free(DDBP_info->rho[k]);
        }
        free(DDBP_info->rho);

        // TODO: free any sub-fields of elem_list here ...
        // go over each element and set up the elements
        for (int k = 0; k < DDBP_info->n_elem_elemcomm; k++) {
            DDBP_ELEM *E_k = &DDBP_info->elem_list[k];
            SPARC_OBJ *ESPRC_k = E_k->ESPRC;
            delete_dyarray(&E_k->atom_list);
            delete_dyarray(&E_k->atom_list_ex);
            free(E_k->element_nbhd_list);
            if (ESPRC_k->isGammaPoint) {
                free(E_k->v);
                free(E_k->v_prev);
                free(E_k->v_tilde);
            } else {
                free(E_k->v_cmplx);
                free(E_k->v_prev_cmplx);
                free(E_k->v_tilde_cmplx);
            }
            
            // free memory for basis overlap matrix
            for(int spn_i = 0; spn_i < Nspin; spn_i++) {
                for (int kpt_i = 0; kpt_i < Nkpts; kpt_i++) {
                    free(E_k->Mvvp[spn_i][kpt_i]);
                }
                free(E_k->Mvvp[spn_i]);
            }
            free(E_k->Mvvp);

            // free DDBP Hamiltonian element row blocks
            // free_DDBP_Hamiltonian_erblks(&E_k->H_DDBP_Ek);
            free(ESPRC_k->Veff_loc_kptcomm_topo);
            // if (ESPRC_k->dmcomm != MPI_COMM_NULL) {
                free(ESPRC_k->Veff_loc_dmcomm);
                free(E_k->Veff_loc_dmcomm_prev); // history
            // }
            // free initial guess vector for Lanczos
            if (ESPRC_k->isGammaPoint)
                free(ESPRC_k->Lanczos_x0);
            else
                free(ESPRC_k->Lanczos_x0_complex);

            free(ESPRC_k->lambda);
            free(ESPRC_k->eigmin);
            free(ESPRC_k->eigmax);
            free(ESPRC_k);
        }

        if(DDBP_info->dmcomm != MPI_COMM_NULL)
            MPI_Comm_free(&DDBP_info->dmcomm);
        if(DDBP_info->bandcomm != MPI_COMM_NULL)
            MPI_Comm_free(&DDBP_info->bandcomm);
        if(DDBP_info->basiscomm != MPI_COMM_NULL)
            MPI_Comm_free(&DDBP_info->basiscomm);
        if(DDBP_info->elemcomm_topo != MPI_COMM_NULL)
            MPI_Comm_free(&DDBP_info->elemcomm_topo);
        if(DDBP_info->elemcomm_topo_inter != MPI_COMM_NULL)
            MPI_Comm_free(&DDBP_info->elemcomm_topo_inter);
        if(DDBP_info->blacscomm != MPI_COMM_NULL)
            MPI_Comm_free(&DDBP_info->blacscomm);
        if(DDBP_info->elemcomm != MPI_COMM_NULL)
            MPI_Comm_free(&DDBP_info->elemcomm);

        #if defined(USE_MKL) || defined(USE_SCALAPACK)
        Cblacs_gridexit(DDBP_info->ictxt_blacs);
        Cblacs_gridexit(DDBP_info->ictxt_blacs_topo);
        #endif // #if defined(USE_MKL) || defined(USE_SCALAPACK)

        free(DDBP_info->elem_list);

    free(DDBP_info);
}


void test_H_DDBP_X_mult(DDBP_INFO *DDBP_info, MPI_Comm kptcomm)
{
    if (DDBP_info->bandcomm_index < 0) return;

    DDBP_HAMILTONIAN *H_DDBP = &DDBP_info->H_DDBP;

    double t1, t2;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // for (int i = 1; i < 100; i++) {
    for (int i = 301; i < 302; i++) {
        int ncol = i;
        // testing H_DDBP * X
        double alpha = 1.0, c = 0.0, beta = 0.0;
        DDBP_ARRAY X, HX;
        init_DDBP_Array(DDBP_info, ncol, &X, DDBP_info->bandcomm);
        init_DDBP_Array(DDBP_info, ncol, &HX, DDBP_info->bandcomm);
        randomize_DDBP_Array(&X, kptcomm);

        t_haloX = t_densmat = t_nloc = t_malloc = 0.0; // re-init timing to 0

        t1 = MPI_Wtime();
        for (int m = 0; m < 1; m++) {
            // DDBP_Hamiltonian_vectors_mult(
            //     alpha, H_DDBP, c, &X, beta, &HX, pSPARC->kptcomm);
            DDBP_Hamiltonian_vectors_mult(
                alpha, H_DDBP, c, &X, beta, &HX, DDBP_info->bandcomm);
            // DDBP_Hamiltonian_vectors_mult_selectcol(
            //     alpha, H_DDBP, c, &X, beta, &HX, pSPARC->kptcomm, 0, ncol/2);
        }
        t2 = MPI_Wtime();
        #ifdef DEBUG
        int nrow = X.nrows[0];
        // if (rank == 0) printf("== H_DDBP*X ==: %.3f ms\n", (t2-t1)*1e3);
        if (rank == 0) printf("== H_DDBP*X: size(X) = (%3d, %3d), t = %7.3f ms, "
            "{haloX: %5.3f ms, loc: %5.3f ms, nloc = %5.3f ms, malloc = %5.3f ms} => %6.3f ms/col\n",
            nrow, ncol, (t2-t1)*1e3,
            t_haloX/ncol*1e3, t_densmat/ncol*1e3, t_nloc/ncol*1e3, t_malloc/ncol*1e3, (t2-t1)/ncol*1e3);
        #endif

        delete_DDBP_Array(&X);
        delete_DDBP_Array(&HX);
    }
}


/**
 * @brief   CheFSI with DDBP method.
 */
void CheFSI_DDBP(
    SPARC_OBJ *pSPARC, double lambda_cutoff, double *x0, int count,
    int kpt, int spn_i)
{
    int rank, nproc, rank_spincomm, nproc_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
    int Nstates = pSPARC->Nstates;
    int nspin = pSPARC->Nspin_spincomm;
    int nkpt = pSPARC->Nkpts_kptcomm;
    // flag to check if this is the first electronic ground state calculation
    // ! for restarted test, the first one is also regarded as first EGS
    int isFirstEGS = (pSPARC->elecgs_Count - pSPARC->StressCount) == 0 ? 1 : 0;
    // flag to check if this is the very first SCF of the entire simulation
    int isInitSCF = (isFirstEGS && count == 0) ? 1 : 0;

    int size_s = pSPARC->Nd_d_dmcomm * pSPARC->Nband_bandcomm;

    #ifdef DEBUG
    if (rank == 0) printf("Entering CheFSI with DDBP method ... \n");
    #endif

    double t1, t2, t3, t_temp;

    // calculate DDBP Basis and H_DDBP, align orbital to current basis
    // Calculate_DDBP_Basis_DDBP_Hamiltonian(pSPARC, count, kpt, spn_i);

    // test_H_DDBP_X_mult(DDBP_info, pSPARC->kptcomm);

    DDBP_ARRAY *X_ks = &DDBP_info->xorb[spn_i][kpt];

    int ChebRepeat = (isInitSCF) ? 1 : 1;
    for (int rpt = 0; rpt < ChebRepeat; rpt++) {
        t1 = MPI_Wtime();
        double lambda_cutoff_ddbp = lambda_cutoff;
        double eigmin_ddbp = DDBP_info->eigmin[spn_i];
        double eigmax_ddbp = DDBP_info->eigmax[spn_i];
        DDBP_ARRAY *X0 = &DDBP_info->Lanczos_x0;
        double *lambda_prev = DDBP_info->lambda+spn_i*Nstates;
        Chebyshevfilter_constants_DDBP(pSPARC, X0, &lambda_cutoff_ddbp,
            &eigmin_ddbp, &eigmax_ddbp, lambda_prev, count, kpt, spn_i);
        t2 = MPI_Wtime();
        if(!rank) printf("Total time for Chebyshevfilter_constants_DDBP: %.3f ms\n", (t2-t1)*1e3);
        #ifdef DEBUG
            if (!rank) {
                printf("\n Chebfilt %d, in DDBP Chebyshev filtering, lambda_cutoff = %f,"
                    " lowerbound = %f, upperbound = %f\n\n",
                    count+1, lambda_cutoff_ddbp, eigmin_ddbp, eigmax_ddbp);
            }
            // TODO: verify if this is necessary!
            // MPI_Bcast(&eigmax_ddbp, 1, MPI_DOUBLE, 0, pSPARC->kptcomm);
            // MPI_Bcast(&eigmin_ddbp, 1, MPI_DOUBLE, 0, pSPARC->kptcomm);
            // MPI_Bcast(&lambda_cutoff_ddbp, 1, MPI_DOUBLE, 0, pSPARC->kptcomm);
        #endif

        //! store the DDBP Chebyshev filtering bounds and cutoff
        DDBP_info->eigmin[spn_i] = eigmin_ddbp;
        DDBP_info->eigmax[spn_i] = eigmax_ddbp;
        lambda_cutoff = lambda_cutoff_ddbp;
        
        // ** Chebyshev filtering on H_DDBP ** //
        t1 = MPI_Wtime();
        DDBP_HAMILTONIAN *H_DDBP = &DDBP_info->H_DDBP;
        DDBP_ARRAY *Y = &DDBP_info->yorb;
        // int m = pSPARC->ChebDegree;
        // use spectrum of H_DDBP to find the appropriate Chebyshev polynomial degree
        int m = Spectrum2ChebDegree(eigmax_ddbp - eigmin_ddbp);
        // m = (isInitSCF) ? m : Spectrum2ChebDegree(eigmax_ddbp);
        m = (m < 17) ? 17 : m;
        t_haloX = t_densmat = t_nloc = t_malloc = 0.0; // re-init timing to 0
        ChebyshevFiltering_DDBP(H_DDBP, X_ks, Y, m,
            lambda_cutoff_ddbp, DDBP_info->eigmax[spn_i], DDBP_info->eigmin[spn_i],
            kpt, spn_i, DDBP_info->bandcomm);
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rank && spn_i == 0)
            printf("Total time for DDBP Chebyshev filtering (%d columns, degree = %d): %.3f ms\n",
                    X_ks->ncol, m, (t2-t1)*1e3);
        int nrow = X_ks->nrows[0];
        int ncol = X_ks->ncol;
        if (rank == 0) printf("== p(H_DDBP)*X: npl = %d, size(X) = (%3d, %3d), t = %7.3f ms, "
            "{haloX: %5.3f ms, loc: %5.3f ms, nloc = %5.3f ms, malloc = %5.3f ms} => %6.3f ms/col\n",
            m, nrow, ncol, (t2-t1)*1e3,
            t_haloX/ncol*1e3, t_densmat/ncol*1e3, t_nloc/ncol*1e3, t_malloc/ncol*1e3, (t2-t1)/ncol*1e3);
        #endif

        t1 = MPI_Wtime();
        // ** calculate projected DDBP Hamiltonian and overlap matrix ** //
        double *Hp = DDBP_info->Hp; // for kpoint, point to the corresponding subspace matrix
        double *Mp = DDBP_info->Mp; // for kpoint, point to the corresponding subspace matrix
        Project_Hamiltonian_DDBP(
            DDBP_info, H_DDBP, DDBP_info->Nstates, Y, DDBP_info->desc_xorb, X_ks,
            DDBP_info->desc_xorb, Hp, DDBP_info->desc_Hp_BLCYC, Mp, DDBP_info->desc_Mp_BLCYC,
            kpt, spn_i, DDBP_info->elemcomm, DDBP_info->bandcomm
            // kpt, spn_i, DDBP_info->blacscomm, DDBP_info->bandcomm
        );
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rank && spn_i == 0) printf("Total time for projection (DDBP): %.3f ms\n", (t2-t1)*1e3);
        #endif

        t1 = MPI_Wtime();
        // ** subspace eigenproblem for DDBP method ** //
        double *Q = DDBP_info->Q;
        double *lambda_ks = DDBP_info->lambda+spn_i*Nstates;

        int proc_active = (int) DDBP_info->elemcomm_index >= 0;
        Solve_Subspace_EigenProblem_DDBP(
            Nstates, Hp, DDBP_info->desc_Hp_BLCYC, Mp, DDBP_info->desc_Mp_BLCYC,
            lambda_ks, Q, DDBP_info->desc_Q_BLCYC, "gen", pSPARC->useLAPACK,
            DDBP_info->elemcomm, pSPARC->eig_paral_blksz, pSPARC->eig_paral_maxnp, proc_active);
        t2 = MPI_Wtime();

        // if eigvals are calculated in root process, then bcast the eigvals
        int nproc_elemcomm;
        MPI_Comm_size(DDBP_info->elemcomm, &nproc_elemcomm);
        // for serial eigensolver, bcast through elemcomm
        if (pSPARC->useLAPACK == 1 && nproc_elemcomm > 1) {
            MPI_Bcast(lambda_ks, Nstates, MPI_DOUBLE, 0, DDBP_info->elemcomm);
        }

        // bcast to inactive processes
        intercomm_bcast(lambda_ks, Nstates, MPI_DOUBLE, 0, pSPARC->kptcomm, proc_active);

        t3 = MPI_Wtime();

        #ifdef DEBUG
        if(!rank) {
            // print eigenvalues
            printf("    (DDBP) first calculated eigval = %.15f\n"
                "    (DDBP) last  calculated eigval = %.15f\n",
                lambda_ks[0], lambda_ks[Nstates-1]);
            int neig_print = min(20,pSPARC->Nstates - pSPARC->Nelectron/2 + 10);
            neig_print = min(neig_print, pSPARC->Nstates);
            // neig_print = pSPARC->Nstates;
            printf("The last %d (DDBP) eigenvalues of kpoints #%d and spin #%d are (Nelectron = %d):\n", neig_print, 1, spn_i, pSPARC->Nelectron);
            int i;
            for (i = 0; i < neig_print; i++) {
                printf("lambda[%4d] = %18.14f\n",
                        pSPARC->Nstates - neig_print + i + 1,
                        lambda_ks[pSPARC->Nstates - neig_print + i]);
            }
            printf("==DDBP subpsace eigenproblem: bcast eigvals took %.3f ms\n", (t3-t2)*1e3);
            printf("Total time for solving subspace eigenvalue problem for DDBP: %.3f ms\n",
                    (t2-t1)*1e3);
        }
        #endif

        t1 = MPI_Wtime();
        // ** subspace rotation for DDBP orbitals ** //
        Subspace_Rotation_DDBP(
            Nstates, Y, DDBP_info->desc_xorb, Q, DDBP_info->desc_Q_BLCYC,
            X_ks, DDBP_info->desc_xorb, DDBP_info->elemcomm
        );
        t2 = MPI_Wtime();
        #ifdef DEBUG
        if(!rank) printf("Total time for subspace rotation (DDBP): %.3f ms\n", (t2-t1)*1e3);
        #endif

        // #define DEBUG_ROTATION
        #ifdef DEBUG_ROTATION
            // find X^T * X and see if the result is identity
            Project_Hamiltonian_DDBP(
                DDBP_info, H_DDBP, DDBP_info->Nstates, X_ks, DDBP_info->desc_xorb, Y,
                DDBP_info->desc_xorb, Hp, DDBP_info->desc_Hp_BLCYC, Mp, DDBP_info->desc_Mp_BLCYC,
                kpt, spn_i, DDBP_info->elemcomm, DDBP_info->bandcomm
            );
            int ZERO = 0;
            double trace = 0.0;
            if (DDBP_info->desc_Mp_BLCYC[1] >= 0) {
                trace = pdlatra(&Nstates, Mp, &ZERO, &ZERO, DDBP_info->desc_Mp_BLCYC);
            }
            void show_mat(double *array, int m, int n);
            for (int i = 0; i < nproc_kptcomm; i++) {
                MPI_Barrier(MPI_COMM_WORLD);
                if (i == rank) {
                    printf("rank = %d, desc_M = [%d,%d,%d,%d,%d,%d,%d,%d,%d]\n", rank, 
                        DDBP_info->desc_Mp_BLCYC[0],
                        DDBP_info->desc_Mp_BLCYC[1],
                        DDBP_info->desc_Mp_BLCYC[2],
                        DDBP_info->desc_Mp_BLCYC[3],
                        DDBP_info->desc_Mp_BLCYC[4],
                        DDBP_info->desc_Mp_BLCYC[5],
                        DDBP_info->desc_Mp_BLCYC[6],
                        DDBP_info->desc_Mp_BLCYC[7],
                        DDBP_info->desc_Mp_BLCYC[8]);

                    double tot_sum = 0.0;
                    if (trace > 0.5) {
                        for (int j = 0; j < DDBP_info->nc_Mp_BLCYC; j++) {
                            double col_sum = 0.0;
                            for (int i = 0; i < DDBP_info->nr_Mp_BLCYC; i++) {
                                col_sum += Mp[i+j*DDBP_info->nr_Mp_BLCYC];
                            }
                            tot_sum += col_sum;
                            if (fabs(col_sum - 1) > 1e-6)
                                printf("debug_warning: column %d in rank %d (loc size: %d x %d, col_sum = %.16g) is all zero!\n",
                                    j, rank, DDBP_info->nr_Mp_BLCYC, DDBP_info->nc_Mp_BLCYC, col_sum);
                        }
                    }
                    printf("rank = %d, trace(M := X'*X) = %.16f, tot_sum = %.16f, M = \n",
                        rank, trace, tot_sum);

                    show_mat(Mp, DDBP_info->nr_Mp_BLCYC, DDBP_info->nc_Mp_BLCYC);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }

        #endif

    }

/*
    extern void sleep();
    extern void usleep();
    int nproc_wrldcomm;
    int rank_wrldcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_wrldcomm);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc_wrldcomm);
    usleep(500000);
    for (int i = 0; i < nproc_wrldcomm; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == rank) {
            int rank_dmcomm = -1;
            int coords[3] = {-1,-1,-1};
            if (pSPARC->dmcomm != MPI_COMM_NULL) {
                MPI_Comm_rank(pSPARC->dmcomm, &rank_dmcomm);
                MPI_Cart_coords(pSPARC->dmcomm, rank_dmcomm, 3, coords);
            }
            printf("band+domain: bandcomm_index = %d, coords = (%d,%d,%d), rank_dmcomm = %d, rank = %2d\n",
                pSPARC->bandcomm_index, coords[0], coords[1], coords[2], rank_dmcomm, rank_wrldcomm);
        }
        // usleep(100000);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
*/
}




/**
 * @brief Calculate DDBP Basis and the DDBP Hamiltonian to prepare for
 *        CheFSI_DDBP.
 *
 *        This routine does the following things to prepare for CheFSI
 *        on the DDBP Hamiltonian:
 *        1. Update DDBP Basis history from the previous step.
 *        2. Calculate new DDBP Basis.
 *        3. Align the DDBP orbitals to the new basis.
 *        4. Calculate new DDBP Hamiltonian.
 * 
 * @param pSPARC The global SPARC_OBJ.
 * @param count Global CheFSI count.
 * @param kpt Local kpoint index.
 * @param spn_i Local Spin index.
 */
void Calculate_DDBP_Basis_DDBP_Hamiltonian(
    SPARC_OBJ *pSPARC, int count, int kpt, int spn_i)
{

    int rank, nproc, rank_spincomm, nproc_kptcomm;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(pSPARC->spincomm, &rank_spincomm);
    MPI_Comm_size(pSPARC->kptcomm, &nproc_kptcomm);
    DDBP_INFO *DDBP_info = pSPARC->DDBP_info;
    int Nstates = pSPARC->Nstates;
    int nspin = pSPARC->Nspin_spincomm;
    int nkpt = pSPARC->Nkpts_kptcomm;
    DDBP_ARRAY *X_ks = &DDBP_info->xorb[spn_i][kpt];

    // flag to check if this is the first electronic ground state calculation
    // ! for restarted test, the first one is also regarded as first EGS
    int isFirstEGS = (pSPARC->elecgs_Count - pSPARC->StressCount) == 0 ? 1 : 0;
    // flag to check if this is the very first SCF of the entire simulation
    int isInitSCF = (isFirstEGS && count == 0) ? 1 : 0;
    int SCF_iter = count < pSPARC->rhoTrigger ? 1 : count - pSPARC->rhoTrigger + 2;
    double scf_err = pSPARC->scf_err; // SCF error from the previous SCF step, for first SCF it's 1+TOL_SCF
    double tol_updatebasis_first_EGS = pSPARC->DDBP_tol_updatebasis_first_EGS; // for the very first electronic ground-state
    double tol_updatebasis = pSPARC->DDBP_tol_updatebasis; // for general electronic ground-state steps

    int updateBasisFlag;
    // if ((isFirstEGS && SCF_iter <= 4) || (!isFirstEGS && SCF_iter == 1)) {
    if ((SCF_iter == 1) || // always calculate basis for the first SCF iteration
        (isFirstEGS && scf_err > tol_updatebasis_first_EGS) ||
        (!isFirstEGS && scf_err > tol_updatebasis)
    ) {
        updateBasisFlag = 1;
    } else {
        updateBasisFlag = 0;
    }

    #ifdef DEBUG
    if (rank == 0) printf("Entering CheFSI with DDBP method ... \n");
    #endif

    double t1, t2, t3, t_temp;
    t1 = MPI_Wtime();
    // set up Veff in the (extended) element E_k
    setup_ddbp_element_Veff(pSPARC);
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Setting up Veff_k in all elements took %.3f ms\n", (t2-t1)*1e3);
    #endif

    t1 = MPI_Wtime();
    if (updateBasisFlag) {
        // update previous basis history: v_prev = v
        Update_basis_history(DDBP_info, nspin, nkpt, pSPARC->isGammaPoint, 
            isInitSCF, kpt, spn_i);
    }
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Updating DDBP basis history took %.3f ms\n", (t2-t1)*1e3);
    #endif

    t1 = MPI_Wtime();
    if (updateBasisFlag) {
        // create DDBP basis functions for all elements
        Calculate_DDBP_basis(pSPARC, count, kpt, spn_i);
    }
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Calculating DDBP basis took %.3f ms\n", (t2-t1)*1e3);
    #endif

    t1 = MPI_Wtime();
    if (updateBasisFlag) {
        // calculate overlap matrix of current basis and previous basis Mvvp = v^T * v_prev
        // this is for aligning the orbital coefficients to current basis
        Calculate_basis_overlap_matrix(DDBP_info, nkpt, isInitSCF, kpt, spn_i);
    }
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Calculating DDBP basis overlap matrix took %.3f ms\n", (t2-t1)*1e3);
    #endif

    t1 = MPI_Wtime();
    if (updateBasisFlag) {
        // X_ks := v^T * v_prev * X_ks = Mvvp * X_ks
        align_orbitals_with_current_basis(DDBP_info, X_ks, isInitSCF, kpt, spn_i);
    }
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Aligning orbitals with current basis took %.3f ms\n", (t2-t1)*1e3);
    #endif

    t1 = MPI_Wtime();
    // calculate DDBP Hamiltonian
    if (updateBasisFlag) {
        // project the global Hamiltonian (-1/2 D^2 + Veff + Vnl) onto the DDBP basis
        Calculate_DDBP_Hamiltonian(pSPARC, count, kpt, spn_i);
    } else {
        // if basis is not updated, only Veff part of H_DDBP needs to be updated
        Update_Veff_part_of_DDBP_Hamiltonian(pSPARC, count, kpt, spn_i);
    }
    #ifdef DEBUG
    t2 = MPI_Wtime();
    if (rank == 0) printf("Calculating DDBP Hamiltonian took %.3f ms\n", (t2-t1)*1e3);
    #endif
}
