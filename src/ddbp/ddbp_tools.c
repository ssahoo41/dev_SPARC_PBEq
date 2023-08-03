/**
 * @file    ddbp_tools.c
 * @brief   This file contains tool functions for the Discrete
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
#include "linearAlgebra.h"
#include "ddbp_types.h"
#include "ddbp_paral.h"
#include "ddbp_tools.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define TEMP_TOL 1e-14


/**
 * @brief   Find the index of a DDBP element based on its Cartesian coordinates.
 *                      (ii_E, jj_E, kk_E) -> index.
 *
 * @param dims   The dimensions of the elements (number of elements in all dir's).
 * @param coords The coordinates of the element.
 * @param index  The index of the the element.
 */
void DDBP_Cart_Index(const int dims[3], const int coords[3], int *index)
{
    *index = coords[2] * dims[1] * dims[0] + coords[1] * dims[0] + coords[0];
}

/**
 * @brief   Find the Cartesian coordinates of a DDBP element based on its index.
 *                      index -> (ii_E, jj_E, kk_E).
 *
 * @param dims   The dimensions of the elements (number of elements in all dir's).
 * @param index  The index of the the element.
 * @param coords The coordinates of the element.
 */
void DDBP_Index_Cart(const int dims[3], const int index, int coords[3]) {
    const int Nex = dims[0];
    const int Ney = dims[1];
    int kk = index / (Nex * Ney);
    int jj = (index - kk * Nex * Ney) / Nex;
    int ii = index - kk * Nex * Ney - jj * Nex;
    coords[0] = ii;
    coords[1] = jj;
    coords[2] = kk;
}


/**
 * @brief   Find the neighbor element index of a DDBP element, given a shift
 *          direction and amount.
 *                      index -> neighbor index.
 *
 * @param dims          The dimensions of the elements (number of elements in all dir's).
 * @param periods       Array specifying whether the element partition is periodic (true)
 *                      or not (false) in each dimension.
 * @param dir           Coordinate dimension of shift (integer).
 * @param disp          Displacement ( > 0: upward shift, < 0: downward shift) (integer).
 * @param index_source  The index of the the element.
 * @param index_dext    The index of the neighbor element.
 */
void DDBP_Cart_shift(
    const int dims[3], const int periods[3], int dir, int disp,
    int index_source, int *index_dest
)
{
    assert(dir >= 0 && dir < 3);
    int coords[3];
    // find element Cartesian coordinates
    DDBP_Index_Cart(dims, index_source, coords);

    // shift coords to find the neighbor coords
    coords[dir] += disp;
    if (periods[dir] == 1)
        coords[dir] = (coords[dir] + dims[dir]) % dims[dir]; // map coords back

    // check if coords runs out of dimension (only meaningful if not periodic)
    if (coords[dir] >= dims[dir]) {
        *index_dest = -1;
    } else {
        DDBP_Cart_Index(dims, coords, index_dest);
    }
}


/**
 * @brief   Calculates start node of an element owned by  
 *          the process (in one direction).
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param bc    Boundary condition in the given direction. 0 - PBC, 1 - DBC.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int element_decompose_nstart(const int n, const int p, const int bc, const int rank)
{
    return ceil(rank * (n-bc) / (double) p - TEMP_TOL);
}


/**
 * @brief   Calculates end node of an element owned by  
 *          the process (in one direction).
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param bc    Boundary condition in the given direction. 0 - PBC, 1 - DBC.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int element_decompose_nend(const int n, const int p, const int bc, const int rank)
{
    return element_decompose_nstart(n, p, bc, rank+1) - 1;
}


/**
 * @brief   Calculates numbero of nodes of an element owned by  
 *          the process (in one direction).
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param bc    Boundary condition in the given direction. 0 - PBC, 1 - DBC.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int element_decompose(const int n, const int p, const int bc, const int rank)
{
    int ns = element_decompose_nstart(n, p, bc, rank);
    int ne = element_decompose_nend(n, p, bc, rank);
    return ne - ns + 1;
}


/**
 * @brief   Calculates which process owns the provided node of an
 *          element (in one direction).
 */
int element_decompose_rank(
    const int n, const int p, const int bc, const int node_indx)
{
    return floor(node_indx * p / (double) (n - bc) + TEMP_TOL);
}


/**
 * @brief   Determine which process owns a specific basis function.
 *
 *          This is accompanied by the routines to create elemcomm, basiscomm,
 *          and dmcomm. If the way these sub-communicators are changed, this has
 *          to be modified so that they're consistent. Note that we assume no
 *          domain paral.
 *
 * @param DDBP_info       Pointer to a DDBP_INFO object.
 * @param k               Index of element E_k that the basis belongs to.
 * @param n               Index of a basis within element E_k. (Not used if
 *                        basiscomm_index is given.)
 * @param basiscomm_index Index of basiscomm. (Set to -1 to ask the program to
 *                        figure out, which requires the value of n.)
 * @param dmcomm_rank     Rank of process in the dmcomm.
 */
int DDBP_basis_owner(
    DDBP_INFO *DDBP_info, int k, int n, int basiscomm_index,
    int dmcomm_rank, MPI_Comm kptcomm
){
    if (basiscomm_index < 0 && n < 0) {
        printf("Please provide a valid basis index or basiscomm_index!\n");
        exit(EXIT_FAILURE);
    }

    int elemcomm_index = element_index_to_elemcomm_index(
        DDBP_info->Ne_tot, DDBP_info->npelem, k);

    if (basiscomm_index == -1)
        basiscomm_index = basis_index_to_basiscomm_index(DDBP_info, k, n);

    int rank_kptcomm = indices_to_rank_kptcomm(
        DDBP_info, elemcomm_index, basiscomm_index, dmcomm_rank, kptcomm
    );

    return rank_kptcomm;
}


/**
 * @brief   Check if a coordinate lies in a cuboid region.
 */
int is_atom_in_region(const double atom_pos[3], const double vert[6]) {
    if (atom_pos[0] < vert[0]-TEMP_TOL || atom_pos[0] > vert[1]+TEMP_TOL) return 0;
    if (atom_pos[1] < vert[2]-TEMP_TOL || atom_pos[1] > vert[3]+TEMP_TOL) return 0;
    if (atom_pos[2] < vert[4]-TEMP_TOL || atom_pos[2] > vert[5]+TEMP_TOL) return 0;
    return 1;
}

/**
 * @brief  For a given atom and it's influence radius, check if it or its
 *         images affect the given region within the cell.
 */
int atom_images_in_region(
    const double atom_pos[3], const double rc[3], const double cell[3],
    const int BCs[3], const double vert[6], double **image_coords)
{
    //[image_xs,image_xe,image_ys,image_ye,image_zs,image_ze]
    int image[6] = {0,0,0,0,0,0};

    for (int d = 0; d < 3; d++) {
        if (BCs[d] == 0) {
            image[d*2] = -floor((rc[d] + atom_pos[d]) / cell[d] + TEMP_TOL);
            image[d*2+1] = floor((rc[d] + cell[d] - atom_pos[d]) / cell[d] + TEMP_TOL);
        }
    }

    // printf("image_bounds = [%d,%d,%d,%d,%d,%d]\n",
    //     image[0],image[1],image[2],image[3],image[4],image[5]);

    // check how many of it's images interacts with the local distributed domain
    int n_images = 0;
    for (int rr = image[4]; rr <= image[5]; rr++) {
        double z0_i = atom_pos[2] + cell[2] * rr; // z coord of image atom
        if ((z0_i < vert[4] - rc[2]) || (z0_i >= vert[5] + rc[2])) continue;
        for (int qq = image[2]; qq <= image[3]; qq++) {
            double y0_i = atom_pos[1] + cell[1] * qq; // y coord of image atom
            if ((y0_i < vert[2] - rc[1]) || (y0_i >= vert[3] + rc[1])) continue;
            for (int pp = image[0]; pp <= image[1]; pp++) {
                double x0_i = atom_pos[0] + cell[0] * pp; // x coord of image atom
                if ((x0_i < vert[0] - rc[0]) || (x0_i >= vert[1] + rc[0])) continue;
                n_images++;
            }
        }
    }

    double *coords = malloc(n_images * 3 * sizeof(double));
    assert(coords != NULL);
    // go over it again to save the coords
    int count = 0;
    for (int rr = image[4]; rr <= image[5]; rr++) {
        double z0_i = atom_pos[2] + cell[2] * rr; // z coord of image atom
        if ((z0_i < vert[4] - rc[2]) || (z0_i >= vert[5] + rc[2])) continue;
        for (int qq = image[2]; qq <= image[3]; qq++) {
            double y0_i = atom_pos[1] + cell[1] * qq; // y coord of image atom
            if ((y0_i < vert[2] - rc[1]) || (y0_i >= vert[3] + rc[1])) continue;
            for (int pp = image[0]; pp <= image[1]; pp++) {
                double x0_i = atom_pos[0] + cell[0] * pp; // x coord of image atom
                if ((x0_i < vert[0] - rc[0]) || (x0_i >= vert[1] + rc[0])) continue;
                coords[count*3]   = x0_i;
                coords[count*3+1] = y0_i;
                coords[count*3+2] = z0_i;
                count++;
            }
        }
    }

    // if (n_images == 0) {
    //     printf("\nThis atom has no influence on the extended element!\n");
    //     printf("n_images = %d\n",n_images);
    // } else {
    //     printf("\nThis atom has influence on the extended element!\n");
    //     printf("n_images = %d\n",n_images);
    //     printf("atom position: (%.2f,%.2f,%.2f)\n",atom_pos[0],atom_pos[1],atom_pos[2]);
    //     printf("vert = [%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]\n",
    //         vert[0],vert[1],vert[2],vert[3],vert[4],vert[5]);
    //     printf("cell = [%.2f,%.2f,%.2f]\n",cell[0],cell[1],cell[2]);
    // }
    *image_coords = coords; // Note: remember to free the memory outside this function!
    return n_images;
}

/**
 * @brief Initialize the dynamic array, allocate initial
 *        memory and set size.
 *
 */
void init_dyarray(dyArray *a)
{
    assert(a != NULL);
    size_t initsize = INIT_CAPACITY;
    a->array = malloc(initsize * sizeof(*a->array));
    assert(a->array != NULL);
    a->len = 0;
    a->capacity = initsize;
}

/**
 * @brief Realloc the dynamic array to the given new capacity.
 *
 *        Note that if the array is extended, all the previous data
 *        are still preserved. If the array is shrinked, all the
 *        previous data up to the new capacity is preserved.
 */
void realloc_dyarray(dyArray *a, size_t new_capacity)
{
    assert(a != NULL);
    value_type *new_arr = realloc(a->array, new_capacity * sizeof(*a->array));
    assert(new_arr != NULL);
    a->array = new_arr;
    a->capacity = new_capacity;
}

/**
 * @brief Double the capacity of the dynamic array.
 *
 *        Note that if the array is extended, all the previous data
 *        are still preserved. If the array is shrinked, all the
 *        previous data up to the new capacity is preserved.
 */
void dyarray_double_capacity(dyArray *a) {
    assert(a != NULL);
    size_t new_capacity = a->capacity ? a->capacity << 1 : INIT_CAPACITY;
    new_capacity = max(new_capacity, INIT_CAPACITY);
    realloc_dyarray(a, new_capacity);
}

/**
 * @brief Half the capacity of the dynamic array.
 *
 *        Note that if the array is extended, all the previous data
 *        are still preserved. If the array is shrinked, all the
 *        previous data up to the new capacity is preserved.
 */
void dyarray_half_capacity(dyArray *a) {
    assert(a != NULL);
    size_t new_capacity = a->capacity >> 1;
    new_capacity = max(new_capacity, INIT_CAPACITY);
    realloc_dyarray(a, new_capacity);
}

/**
 * @brief Append an element to the dynamic array.
 *
 */
void append_dyarray(dyArray *a, value_type element)
{
    assert(a != NULL);

    // double the size of memory allocated if it's len up
    if (a->len == a->capacity) {
        dyarray_double_capacity(a);
    }

    // append the element to array
    a->array[a->len] = element;
    a->len++;
}

/**
 * @brief Pop the last element from the dynamic array.
 *
 */
value_type pop_dyarray(dyArray *a)
{
    assert(a != NULL);
    if (a->len < 1) {
        printf("Error: pop_dyarray target is empty!\n");
        exit(1);
    }

    a->len--; // reduce len by 1

    if (4 * a->len < a->capacity) {
        dyarray_half_capacity(a);
    }

    return a->array[a->len];
}

/**
 * @brief Clear the dynamic array.
 *
 *        This function does not destroy the array, it simply
 *        resets the lenght of the dynamic array to 0, and resets
 *        the capacity.
 */
void clear_dyarray(dyArray *a) {
    assert(a != NULL);
    size_t initsize = INIT_CAPACITY;
    realloc_dyarray(a, initsize);
    a->len = 0;
}

/**
 * @brief Delete the dynamic array.
 *
 */
void delete_dyarray(dyArray *a)
{
    assert(a != NULL);
    free(a->array);
    a->array = NULL;
    a->len = a->capacity = 0;
}


/**
 * @brief Print the dynamic array.
 *
 */
void print_dyarray(const dyArray *a) {
    printf("([");
    for (int i = 0; i < a->len; i++) {
        if (i > 0) printf(" ");
        printf("%d", a->array[i]);
    }
    printf("], len = %ld, capacity = %ld)\n",a->len,a->capacity);
}

// if array is too long, only show the first 5 and last 5
void show_dyarray(const dyArray *a) {
    if (a->len <= 10) {
        print_dyarray(a);
        return;
    }

    printf("([");
    for (int i = 0; i < 5; i++) {
        if (i > 0) printf(" ");
        printf("%d", a->array[i]);
    }
    printf(" ...");
    for (int i = a->len-5; i < a->len; i++) {
        if (i > 0) printf(" ");
        printf("%d", a->array[i]);
    }
    printf("], len = %ld, capacity = %ld)\n",a->len,a->capacity);
}


/**
 * @brief Create a DDBP Array object, set up parameters and
 *        allocate memory.
 *
 * @param BCs Global Boundary Conditions of the array
 * @param Edims Global number of elements in each dir
 * @param nelem Number of elements assigned to the process.
 * @param elem_list List of elemets assigned.
 * @param ncol Number of columns of the array.
 * @param X The DDBP Array.
 */
void create_DDBP_Array(
    const int BCs[3], const int Edims[3],
    int nelem, DDBP_ELEM *elem_list, int ncol, DDBP_ARRAY *X)
{
    X->BCs[0] = BCs[0];
    X->BCs[1] = BCs[1];
    X->BCs[2] = BCs[2];
    X->Edims[0] = Edims[0];
    X->Edims[1] = Edims[1];
    X->Edims[2] = Edims[2];
    X->nelem = nelem;
    X->elem_list = elem_list;
    X->ncol = ncol;
    X->array = malloc(nelem * sizeof(*X->array));
    X->nrows = malloc(nelem * sizeof(*X->nrows));
    assert(X->array != NULL && X->nrows != NULL);
    X->haloX_info = malloc(nelem * sizeof(*X->haloX_info));
    assert(X->haloX_info != NULL);

    for (int k = 0; k < nelem; k++) {
        DDBP_ELEM *E_k = &elem_list[k];
        int nrow = E_k->nALB;
        X->nrows[k] = nrow;
        X->array[k] = calloc(ncol * nrow, sizeof(double));
        assert(X->array[k] != NULL);
    }
}


/**
 * @brief Initialize a DDBP Array.
 *
 *        Set up the parameters including the halo exchange objects
 *        and allocate memory for the array.
 *
 * @param DDBP_info DDBP_INFO object.
 * @param ncol Number of columns in the DDBP Array.
 * @param X DDBP Array to be initialized.
 * @param comm MPI communicator where X is distributed (for haloX).
 */
void init_DDBP_Array(
    DDBP_INFO *DDBP_info, int ncol, DDBP_ARRAY *X, MPI_Comm comm)
{
    int nelem = DDBP_info->n_elem_elemcomm;
    int BCs[3], Edims[3];
    BCs[0] = DDBP_info->BCx;
    BCs[1] = DDBP_info->BCy;
    BCs[2] = DDBP_info->BCz;
    Edims[0] = DDBP_info->Nex;
    Edims[1] = DDBP_info->Ney;
    Edims[2] = DDBP_info->Nez;
    create_DDBP_Array(BCs, Edims, nelem, DDBP_info->elem_list, ncol, X);
    setup_haloX_DDBP_Array(DDBP_info, X, comm);
}


/**
 * @brief Copy the halo exchange info.
 * 
 * @param n Number of haloX info to be copied.
 * @param haloX_src Source haloX info list.
 * @param haloX_dest Destination haloX info list.
 */
void copy_haloX_info(int n, const haloX_t *haloX_src, haloX_t *haloX_dest)
{
    for (int i = 0; i < n; i++) haloX_dest[i] = haloX_src[i];
}


/**
 * @brief Duplicate a DDBP Array template. Note that the new array is
 *        initialized to 0.
 *
 *        Create a new DDBP Array of the same size and distribution as
 *        the template. This routine sets up the parameters and the
 *        halo exchange info, and allocates memory, but the array values
 *        are initialized to 0's.
 * 
 * @param X Template DDBP Array.
 * @param Y New DDBP Array to be set up.
 */
void duplicate_DDBP_Array_template(const DDBP_ARRAY *X, DDBP_ARRAY *Y)
{
    // set up parameters and allocate memory
    create_DDBP_Array(X->BCs, X->Edims, X->nelem, X->elem_list, X->ncol, Y);
    // copy halo exchange info
    copy_haloX_info(X->nelem, X->haloX_info, Y->haloX_info);
}


/**
 * @brief Create a deep copy of DDBP Array, including the array values.
 * 
 * @param X Source DDBP Array.
 * @param Y New copy of DDBP Array.
 */
void deepcopy_DDBP_Array(const DDBP_ARRAY *X, DDBP_ARRAY *Y)
{
    // first create the array: set up params and haloX info, allocate
    // memory, array value is initialized to 0
    duplicate_DDBP_Array_template(X, Y);
    // copy the array values, Y = 1.0 * X + 0.0 * Y = X
    axpby_DDBP_Array(1.0, X, 0.0, Y);
}



/**
 * @brief Delete a DDBP Array object.
 *
 * @param X DDBP Array to be deleted.
 */
void delete_DDBP_Array(DDBP_ARRAY *X)
{
    int nelem = X->nelem;
    for (int k = 0; k < nelem; k++) {
        free(X->array[k]);
        // free(X->haloX_info[k]);
    }
    free(X->array);
    free(X->haloX_info);
    free(X->nrows);
}


/**
 * @brief Randomize a DDBP Array.
 * 
 * @param X DDBP Array.
 * @param comm Communicator where X is distributed, the random
 *             seeds are generated based on the rank and shift.
 */
void randomize_DDBP_Array(DDBP_ARRAY *X, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    int nelem = X->nelem;
    int ncol = X->ncol;
    for (int k = 0; k < nelem; k++) {
        int nrow = X->nrows[k];
        double rand_min = -0.5, rand_max = 0.5;
        int seed = 1 + (k+rank)*(k+rank+1)+k; // Cantor pairing function
        SetRandMat_seed(X->array[k], nrow, ncol, rand_min, rand_max, seed);
    }
}


/**
 * @brief Scale X, X = a * X.
 * 
 * @param a Scalar a.
 * @param X DDBP array X.
 */
void scale_DDBP_Array(double a, DDBP_ARRAY *X)
{
    // return if a == 1.0
    if (fabs(a - 1.0) < TEMP_TOL) return;

    int nelem = X->nelem;
    int ncol = X->ncol;
    for (int k = 0; k < nelem; k++) {
        int nrow = X->nrows[k];
        int len = nrow * ncol;
        // scale X->array[k] by a
        double *xk = X->array[k];
        for (int i = 0; i < len; i++) xk[i] *= a;
    }
}

/**
 * @brief Copy X into Y, Y = a * X.
 * 
 * @param a Scalar a.
 * @param X DDBP array X.
 * @param Y DDBP array Y.
 */
void copy_DDBP_Array(double a, const DDBP_ARRAY *X, DDBP_ARRAY *Y)
{
    char AisOne = fabs(a - 1.0) < TEMP_TOL ? 'y' : 'n';
    char AisZero = fabs(a) < TEMP_TOL ? 'y' : 'n';

    int nelem = X->nelem;
    int ncol = X->ncol;
    for (int k = 0; k < nelem; k++) {
        int nrow = X->nrows[k];
        int len = nrow * ncol;
        double *xk = X->array[k];
        double *yk = Y->array[k];
        if (AisZero == 'y') {
            for (int i = 0; i < len; i++) yk[i] = 0.0;
        } else if (AisOne == 'y') {
            for (int i = 0; i < len; i++) yk[i] = xk[i];
        } else {
            for (int i = 0; i < len; i++) yk[i] = xk[i] * a;
        }
    }
}


/**
 * @brief Calculate Y = a * X + Y.
 *
 *        "axpy" stands for "a X plus Y".
 *
 * @param a Scalar a.
 * @param X DDBP array X.
 * @param Y DDBP array Y.
 */
void axpy_DDBP_Array(double a, const DDBP_ARRAY *X, DDBP_ARRAY *Y)
{
    // return if a == 0.0
    if (fabs(a) < TEMP_TOL) return;
    char AisOne = fabs(a - 1.0) < TEMP_TOL ? 'y' : 'n'; // a == 1.0
    int nelem = X->nelem;
    int ncol = X->ncol;
    for (int k = 0; k < nelem; k++) {
        int nrow = X->nrows[k];
        int len = nrow * ncol;
        // scale X->array[k] by a
        double *yk = Y->array[k];
        double *xk = X->array[k];
        if (AisOne == 'y') {
            for (int i = 0; i < len; i++) yk[i] += xk[i];
        } else {
            for (int i = 0; i < len; i++) yk[i] += xk[i] * a;
        }
    }
}



/**
 * @brief Calculate Y = a * X + b * Y.
 *
 *        "axpby" stands for "a X plus b Y".
 *
 * @param a Scalar a.
 * @param X DDBP array X.
 * @param b Scalar b.
 * @param Y DDBP array Y.
 */
void axpby_DDBP_Array(
    double a, const DDBP_ARRAY *X, double b, DDBP_ARRAY *Y)
{
    char BisZero = fabs(b) < TEMP_TOL ? 'y' : 'n'; // b == 0.0
    if (BisZero == 'y') {
        // Y = a * X
        copy_DDBP_Array(a, X, Y);
    } else {
        // Y = b * Y
        scale_DDBP_Array(b, Y);
        // Y = a * X + Y
        axpy_DDBP_Array(a, X, Y);
    }
}


/**
 * @brief Calculate Z = a * X + b * Y + c * Z.
 *
 *        "axpbypcz" stands for "a X plus b Y plus c Z".
 *
 * @param a Scalar a.
 * @param X DDBP array X.
 * @param b Scalar b.
 * @param Y DDBP array Y.
 * @param c Scalar c.
 * @param Z DDBP array Z.
 */
void axpbypcz_DDBP_Array(
    double a, const DDBP_ARRAY *X, double b, const DDBP_ARRAY *Y,
    double c, DDBP_ARRAY *Z)
{
    char CisOne = fabs(c - 1.0) < TEMP_TOL ? 'y' : 'n';
    char CisZero = fabs(c) < TEMP_TOL ? 'y' : 'n';
    int nelem = X->nelem;
    int ncol = X->ncol;
    for (int k = 0; k < nelem; k++) {
        int nrow = X->nrows[k];
        int len = nrow * ncol;
        double *xk = X->array[k];
        double *yk = Y->array[k];
        double *zk = Z->array[k];
        if (CisZero == 'y') {
            for (int i = 0; i < len; i++)
                zk[i] = a * xk[i] + b * yk[i];
        } else if (CisOne == 'y') {
            for (int i = 0; i < len; i++)
                zk[i] += a * xk[i] + b * yk[i];
        } else {
            for (int i = 0; i < len; i++)
                zk[i] = a * xk[i] + b * yk[i] + c * zk[i];
        }
    }
}



/**
 * @brief Calculate dot products between X and Y.
 *
 *        If there are more than one columns, it finds all the
 *        dot products between the columns:
 *          results[n] = X(:,n)' * Y(:,n).
 *
 * @param X DDBP Array X.
 * @param Y DDBP Array Y.
 * @param ncol Number of columns in X and Y.
 * @param results Results array.
 * @param comm Communicator where X and Y are distributed.
 */
void DotProduct_DDBP_Array(
    const DDBP_ARRAY *X, const DDBP_ARRAY *Y,
    int ncol, double *results, MPI_Comm comm)
{
    assert(ncol == X->ncol && ncol == Y->ncol);

    for (int n = 0; n < ncol; n++)
        results[n] = 0.0;

    int nelem = X->nelem;
    for (int k = 0; k < nelem; k++) {
        int nrow = X->nrows[k];
        double *xk = X->array[k];
        double *yk = Y->array[k];
        for (int n = 0; n < ncol; n++) {
            double *xk_n = xk + n * nrow;
            double *yk_n = yk + n * nrow;
            double loc_res_n = 0.0;
            for (int i = 0; i < nrow; i++) {
                loc_res_n += xk_n[i] * yk_n[i];
            }
            results[n] += loc_res_n;
        }
    }

    int size;
    MPI_Comm_size(comm, &size);
    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, results, ncol, MPI_DOUBLE, MPI_SUM, comm);
    }
}


/**
 * @brief Find optimal number of processes to use for performing
 *        a parallel matrix operation using scalapack routines.
 * 
 * @param m Size of the matrix, number of rows.
 * @param n Size of the matrix, number of columns.
 * @param typ Operation type. Options: "pdgemm".
 * @return int Maximum number of processes to be used.
 */
int best_max_nproc(int m, int n, char *typ) {
    int max_nproc = 1;
    if (strcmpi(typ, "pdgemm") == 0) {
        int navg = round(sqrt(m * n));
        if (navg < 200) max_nproc = 4;
        else if (navg <= 1280) max_nproc = navg / 20;
        else if (navg <= 2500) max_nproc = 64;
        else if (navg <= 8000) max_nproc = 256;
        else max_nproc = navg / 12;
        return max_nproc;
    }
    return max_nproc;
}


/**
 * @brief Calculate Hermitian products between X and Y, we assume X and Y
 *        are of the same size and distribution.
 *
 *        M = alpha * X^T * Y + beta * M.
 *
 * @param n Global size of the resulting matrix M.
 * @param alpha Scalar alpha.
 * @param X DDBP Array X.
 * @param descX Descriptor for X in each element.
 * @param Y DDBP Array Y (can be the same as X).
 * @param descY Descriptor for Y in each element. Should be same as descX.
 * @param beta Scalar beta.
 * @param M Resulting matrix (distributed).
 * @param descM Descriptor for M.
 * @param rowcomm Row communicator where X and Y are distributed (elemcomm).
 * @param colcomm Column communicator where X and Y are distributed (bandcomm).
 */
void Hermitian_Multiply_DDBP_Array(
    int n, double alpha, const DDBP_ARRAY *X, int *descX, const DDBP_ARRAY *Y,
    int *descY, double beta, double *M, int *descM, MPI_Comm rowcomm, MPI_Comm colcomm)
{
    if (colcomm == MPI_COMM_NULL || rowcomm == MPI_COMM_NULL) return;
    assert(X->ncol == Y->ncol);
    assert(X->nelem == Y->nelem);
    assert(fabs(beta) < TEMP_TOL); // TODO: if beta /= 0, the answer is wrong!
                                   // TODO: because the original M value is allreduced too!

    int nprow, npcol;
    MPI_Comm_size(rowcomm, &nprow);
    MPI_Comm_size(colcomm, &npcol);

    // TODO: another way is to contatenate all the elements together and do pdgemm once
    int nelem = X->nelem;
    int count = 0;
    for (int k = 0; k < nelem; k++) {
        int nrow = X->nrows[k];
        double *xk = X->array[k];
        double *yk = Y->array[k];
        double beta_k = (count == 0) ? beta : 1.0;
        count++;
        // M = xk^T * yk + beta_k * M
        pdgemm_subcomm(
            "T", "N", n, n, nrow, 1.0, xk, descX, yk, descY, beta_k, M, descM,
            rowcomm, best_max_nproc(nrow, n, "pdgemm")
        );
    }

    if (npcol > 1 && descM[1] >= 0) {
        int m_loc, n_loc;
        #if defined(USE_MKL) || defined(USE_SCALAPACK)
        int nprow_ictxt, npcol_ictxt, myrow, mycol;
        Cblacs_gridinfo(descM[1], &nprow_ictxt, &npcol_ictxt, &myrow, &mycol);
        int m_A = descM[2];
        int n_A = descM[3];
        int mb_A = descM[4];
        int nb_A = descM[5];
        int rsrc_A = descM[6];
        int csrc_A = descM[7];
        m_loc = numroc_(&m_A, &mb_A, &myrow, &rsrc_A, &nprow_ictxt);
        n_loc = numroc_(&n_A, &nb_A, &mycol, &csrc_A, &npcol_ictxt);
        #else
        m_loc = n_loc = 1; // TODO: find the right value without ScaLAPACK routines
        assert(0);
        #endif
        // int rank;
        // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // if (m_loc != descM[8]) {
            // printf("rank = %d, (nprow,npcol) = (%d,%d), (myrow,mycol) = (%d,%d), m_loc = %d, n_loc = %d, descM[8] = %d\n",
            //     rank,nprow_ictxt,npcol_ictxt, myrow,mycol, m_loc, n_loc, descM[8]);
            // print_array(descM, 9, sizeof(int));
            // MPI_Barrier(colcomm);
            // exit(9);
        // }
        // m_loc = n_loc = descM[8];
        // m_loc = max(m_loc,1);
        // n_loc = max(n_loc,1);
        // if (m_loc * n_loc >= 0)
        MPI_Allreduce(MPI_IN_PLACE, M, m_loc*n_loc, MPI_DOUBLE, MPI_SUM, colcomm);
    }
}


/**
 * @brief Calculate matrix multiplication of a DDBP Array and a matrix. We
 *        assume each rowcomm has a full distributed Q matrix, thus we do
 *        the multiplication in each rowcomm independently.
 *
 *          Y = alpha * X * Q + beta * Y.
 *
 * @param n Global size of the resulting matrix Q.
 * @param alpha Scalar alpha.
 * @param X DDBP Array X.
 * @param descX Descriptor for X in each element.
 * @param Q Resulting matrix (distributed).
 * @param descQ Descriptor for Q.
 * @param beta Scalar beta.
 * @param Y DDBP Array Y (can be the same as X).
 * @param descY Descriptor for Y in each element. Should be same as descX.
 * @param rowcomm Row communicator where X and Y are distributed (elemcomm).
 */
void DDBP_Array_Matrix_Multiply(
    int n, double alpha, const DDBP_ARRAY *X, int *descX, const double *Q,
    int *descQ, double beta, DDBP_ARRAY *Y, int *descY, MPI_Comm rowcomm)
{
    if (rowcomm == MPI_COMM_NULL) return;
    assert(X->ncol == Y->ncol);
    assert(X->nelem == Y->nelem);

    int nprow;
    MPI_Comm_size(rowcomm, &nprow);

    // TODO: another way is to contatenate all the elements together and do pdgemm once
    int nelem = X->nelem;
    for (int k = 0; k < nelem; k++) {
        int nrow = X->nrows[k];
        double *xk = X->array[k];
        double *yk = Y->array[k];
        // yk = xk * Q + beta_k * yk
        pdgemm_subcomm(
            "N", "N", nrow, n, n, alpha, xk, descX, Q, descQ, beta, yk, descY,
            rowcomm, best_max_nproc(nrow, n, "pdgemm")
        );
    }
}



/**
 * @brief Calculate a function on the FD grid based on the coefficients of
 *        the DDBP basis.
 *                 yk = vk * xk.
 * 
 * @param vk DDBP basis function in an element.
 * @param descvk Descriptor for vk.
 * @param xk Coefficients of DDBP basis for a function.
 * @param descxk Descriptor for xk.
 * @param yk Function on the FD grid in an element (output).
 * @param descyk Descriptor for yk.
 * @param rowcomm Element communicator containing the element.
 */
void DDBP_Element_Basis_Coeff_Multiply(
    const double *vk, int *descvk, const double *xk, int *descxk,
    double *yk, int *descyk, MPI_Comm rowcomm)
{
    if (rowcomm == MPI_COMM_NULL) return;

    int m_vk = descvk[2];
    int n_vk = descvk[3];
    int m_xk = descxk[2];
    int n_xk = descxk[3];
    int m_yk = descyk[2];
    int n_yk = descyk[3];
    assert(n_vk == m_xk);
    assert(m_vk == m_yk);
    assert(n_xk == n_yk);

    int m = m_yk, n = n_yk, k = m_xk;
    int n_eff = (int) fabs(round(cbrt(m * n * k)));
    if (n_eff == 0) return;

    // yk = vk * xk
    pdgemm_subcomm(
        "N", "N", m, n, k, 1.0, vk, descvk, xk, descxk, 0.0, yk, descyk,
        rowcomm, best_max_nproc(n_eff, n_eff, "pdgemm")
    );
}



/**
 * @brief Calculate 2-norm of X.
 *
 *        If there are more than one columns, it finds the norms
 *        for all the columns:
 *          results[n] = ||X(:,n)||_2.
 *
 * @param X DDBP Array X.
 * @param ncol Number of columns in X.
 * @param results Results array.
 * @param comm Communicator where X is distributed.
 */
void Norm_DDBP_Array(
    const DDBP_ARRAY *X, int ncol, double *results, MPI_Comm comm)
{
    DotProduct_DDBP_Array(X, X, ncol, results, comm);
    for (int n = 0; n < ncol; n++) {
        results[n] = sqrt(results[n]);
    }
}


/**
 * @brief Print a DDBP Array object.
 *
 * @param X DDBP Array to be printed.
 */
void print_DDBP_Array(DDBP_ARRAY *X)
{
    printf("DDBP_ARRAY->BCs = [%d,%d,%d]\n",X->BCs[0],X->BCs[1],X->BCs[2]);
    printf("DDBP_ARRAY->Edims = [%d,%d,%d]\n",X->Edims[0],X->Edims[1],X->Edims[2]);
    printf("DDBP_ARRAY->nelem = %d\n",X->nelem);
    printf("DDBP_ARRAY->ncol  = %d\n",X->ncol);
    printf("DDBP_ARRAY->nrows  = ");
    print_array(X->nrows, X->nelem, sizeof(*X->nrows));
    for (int k = 0; k < X->nelem; k++) {
        DDBP_ELEM *E_k = &X->elem_list[k];
        print_haloX(E_k, &X->haloX_info[k], MPI_COMM_WORLD);
    }
    printf("\n");
}


int double_check_arrays(const double *a, const double *b, const int n)
{
    double tol = 1e-8;
    double err = 0.0;
    for (int i = 0; i < n; i++) {
        err = max(err, fabs(a[i] - b[i]));
    }
    if (err >= tol)
        printf("In function double_check_result: err = %.3e\n",err);
    return (int) (err >= tol); // 1 - error, 0 - success
}


int double_check_int_arrays(const int *a, const int *b, const int n)
{
    int err = 0;
    for (int i = 0; i < n; i++) {
        err = max(err, abs(a[i] - b[i]));
    }
    if (err > 0)
        printf("In function double_check_result: err = %d\n",err);
    return (int) (err > 0); // 1 - error, 0 - success
}


int double_check_DDBP_arrays(
    const DDBP_ARRAY *X, const DDBP_ARRAY *Y, const int is, const int ncol)
{
    if (X->BCs[0] != Y->BCs[0]) return 1;
    if (X->BCs[1] != Y->BCs[1]) return 1;
    if (X->BCs[2] != Y->BCs[2]) return 1;
    if (X->Edims[0] != Y->Edims[0]) return 1;
    if (X->Edims[1] != Y->Edims[1]) return 1;
    if (X->Edims[2] != Y->Edims[2]) return 1;
    if (X->nelem != Y->nelem) return 1;
    if (X->ncol != Y->ncol) return 1;
    int nelem = X->nelem;
    for (int k = 0; k < nelem; k++) {
        if (X->nrows[k] != Y->nrows[k]) return 1;
        int nrow = X->nrows[k];
        int arr_err = double_check_arrays(
            X->array[k]+is*nrow, Y->array[k]+is*nrow, nrow * ncol);
        if (arr_err) return 1;
    }
    return 0;
}



/**
 * @brief Restrict any function defined on the extended element
 *        to the element.
 *
 */
void restrict_to_element(const DDBP_ELEM *E_k, const double *x_ex, double *x)
{
    const int x_i_spos = E_k->DMVert[0];
    const int y_i_spos = E_k->DMVert[2];
    const int z_i_spos = E_k->DMVert[4];
    const int x_o_epos = E_k->nx_d - 1;
    const int y_o_epos = E_k->ny_d - 1;
    const int z_o_epos = E_k->nz_d - 1;
    const int stride_y_i = E_k->nx_ex_d;
    const int stride_z_i = E_k->nx_ex_d * E_k->ny_ex_d;
    const int stride_y_o = E_k->nx_d;
    const int stride_z_o = E_k->nx_d * E_k->ny_d;

    restrict_to_subgrid(
        x_ex, x, stride_y_o, stride_y_i, stride_z_o, stride_z_i,
        0, x_o_epos, 0, y_o_epos, 0, z_o_epos,
        x_i_spos, y_i_spos, z_i_spos
    );
}



// /**
//  * @brief Extend any function defined on the element to the extended element and
//  *        leave the extended part untouched.
//  *
//  */
// void extend_to_extended_element(const DDBP_ELEM *E_k, const double *x, double *x_ex)
// {
//     const int x_i_spos = 0;
//     const int y_i_spos = 0;
//     const int z_i_spos = 0;
//     const int x_o_spos = E_k->DMVert[0];
//     const int y_o_spos = E_k->DMVert[2];
//     const int z_o_spos = E_k->DMVert[4];
//     const int x_o_epos = x_o_spos + E_k->nx_d - 1;
//     const int y_o_epos = y_o_spos + E_k->ny_d - 1;
//     const int z_o_epos = z_o_spos + E_k->nz_d - 1;
//     const int stride_y_o = E_k->nx_ex_d;
//     const int stride_z_o = E_k->nx_ex_d * E_k->ny_ex_d;
//     const int stride_y_i = E_k->nx_d;
//     const int stride_z_i = E_k->nx_d * E_k->ny_d;

//     restrict_to_subgrid(
//         x, x_ex, stride_y_o, stride_y_i, stride_z_o, stride_z_i,
//         x_o_spos, x_o_epos, y_o_spos, y_o_epos, z_o_spos, z_o_epos,
//         x_i_spos, y_i_spos, z_i_spos
//     );
// }



// /**
//  * @brief Extend any function defined on the current domain by extending
//  *        the grid by {nshiftx,nshifty,nshiftz} grid points in all directions.
//  *        In other words,
//  *          x_ex(nshiftx:nshiftx+nx-1,nshifty:nshifty+ny-1,nshiftz:nshiftz+nz-1)
//  *        = x
//  *
//  */
// void extend_to_extended_domain(
//     const double *x, double *x_ex,
//     int nshiftx, int nshifty, int nshiftz,
//     int nx, int ny, int nz,
//     int nx_ex, int ny_ex, int nz_ex)
// {
//     const int x_i_spos = 0;
//     const int y_i_spos = 0;
//     const int z_i_spos = 0;
//     const int x_o_spos = nshiftx;
//     const int y_o_spos = nshifty;
//     const int z_o_spos = nshiftz;
//     const int x_o_epos = x_o_spos + nx - 1;
//     const int y_o_epos = y_o_spos + ny - 1;
//     const int z_o_epos = z_o_spos + nz - 1;
//     const int stride_y_o = nx_ex;
//     const int stride_z_o = nx_ex * ny_ex;
//     const int stride_y_i = nx;
//     const int stride_z_i = nx * ny;

//     restrict_to_subgrid(
//         x, x_ex, stride_y_o, stride_y_i, stride_z_o, stride_z_i,
//         x_o_spos, x_o_epos, y_o_spos, y_o_epos, z_o_spos, z_o_epos,
//         x_i_spos, y_i_spos, z_i_spos
//     );
// }



// /**
//  * @brief Restrict any function defined on the current domain by cutting off
//  *        the grid by {nshiftx,nshifty,nshiftz} grid points in all directions.
//  *        In other words,
//  *        x = x_ex(nshiftx:nshiftx+nx-1,nshifty:nshifty+ny-1,nshiftz:nshiftz+nz-1).
//  *
//  *        This is a reverse opration to the extend_to_extended_domain() routine.
//  */
// void restrict_to_domain(
//     const double *x_ex, double *x,
//     int nshiftx, int nshifty, int nshiftz,
//     int nx, int ny, int nz,
//     int nx_ex, int ny_ex, int nz_ex
// )
// {
//     const int x_i_spos = nshiftx;
//     const int y_i_spos = nshifty;
//     const int z_i_spos = nshiftz;
//     const int x_o_spos = 0;
//     const int y_o_spos = 0;
//     const int z_o_spos = 0;
//     const int x_o_epos = x_o_spos + nx - 1;
//     const int y_o_epos = y_o_spos + ny - 1;
//     const int z_o_epos = z_o_spos + nz - 1;
//     const int stride_y_i = nx_ex;
//     const int stride_z_i = nx_ex * ny_ex;
//     const int stride_y_o = nx;
//     const int stride_z_o = nx * ny;

//     restrict_to_subgrid(
//         x_ex, x, stride_y_o, stride_y_i, stride_z_o, stride_z_i,
//         x_o_spos, x_o_epos, y_o_spos, y_o_epos, z_o_spos, z_o_epos,
//         x_i_spos, y_i_spos, z_i_spos
//     );
// }



/**
 * @brief Extract any subgrid of a function defined on the current grid and assign
 *        it to a subgrid in another array.
 *        In other words,
 *        x_out(is:ie,js:je,ks:ke) = x(ips:ipe,jps:jpe,kps:kpe).
 *
 *        This is to mimic the Matlab way of extracting sub-matrices.
 */
void extract_subgrid(
    const double *x, int npx, int npy, int npz,
    int ips, int ipe, int jps, int jpe, int kps, int kpe,
    double *x_out, int nx, int ny, int nz,
    int is, int ie, int js, int je, int ks, int ke
)
{
    if ((ie-is) != (ipe-ips) || (je-js) != (jpe-jps) || (ke-ks) != (kpe-kps))
    {
        printf("\n\nSubscripted assignment dimension mismatch.\n\n");
        exit(EXIT_FAILURE);
    }

    if (!(0 <= ips && ipe < npx) ||
        !(0 <= jps && jpe < npy) ||
        !(0 <= kps && kpe < npz))
    {
        printf("\n\nInput index exceeds array dimensions.\n\n");
        exit(EXIT_FAILURE);
    }

    if (!(0 <= is && is <= ie && ie < nx) ||
        !(0 <= js && js <= je && je < ny) ||
        !(0 <= ks && ks <= ke && ke < nz))
    {
        printf("\n\nOutput index exceeds array dimensions.\n\n");
        exit(EXIT_FAILURE);
    }

    restrict_to_subgrid(
        x, x_out, nx, npx, nx*ny, npx*npy,
        is, ie, js, je, ks, ke, ips, jps, kps
    );
}


/**
 * @brief Extract any subgrid of a function defined on the current grid and add
 *        it to a subgrid in another array.
 *        In other words,
 *        x_out(is:ie,js:je,ks:ke) += alpha * x(ips:ipe,jps:jpe,kps:kpe).
 *
 *        This is to mimic the Matlab way of extracting sub-matrices, but note
 *        this ADDS the value instead of overwriting it in the output array.
 */
void sum_subgrid(
    double alpha, const double *x, int npx, int npy, int npz,
    int ips, int ipe, int jps, int jpe, int kps, int kpe,
    double *x_out, int nx, int ny, int nz,
    int is, int ie, int js, int je, int ks, int ke
)
{
    if ((ie-is) != (ipe-ips) || (je-js) != (jpe-jps) || (ke-ks) != (kpe-kps))
    {
        printf("\n\nSubscripted assignment dimension mismatch.\n\n");
        exit(EXIT_FAILURE);
    }

    if (!(0 <= ips && ipe < npx) ||
        !(0 <= jps && jpe < npy) ||
        !(0 <= kps && kpe < npz))
    {
        printf("\n\nInput index exceeds array dimensions.\n\n");
        exit(EXIT_FAILURE);
    }

    if (!(0 <= is && is <= ie && ie < nx) ||
        !(0 <= js && js <= je && je < ny) ||
        !(0 <= ks && ks <= ke && ke < nz))
    {
        printf("\n\nOutput index exceeds array dimensions.\n\n");
        exit(EXIT_FAILURE);
    }

    const int stride_y_i = npx;
    const int stride_y_o = nx;
    const int stride_z_i = npx * npy;
    const int stride_z_o = nx * ny;
    const int shift_ip = ips - is;
    const int shift_jp = jps - js;
    const int shift_kp = kps - ks;
    for (int k = ks; k <= ke; k++) {
        int kp = k + shift_kp;
        for (int j = js; j <= je; j++) {
            int jp = j + shift_jp;
            int offset = k * stride_z_o + j * stride_y_o;
            int offset_i = kp * stride_z_i + jp * stride_y_i;
            for (int i = is; i <= ie; i++) {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_i  = offset_i + ip;
                x_out[idx] += alpha * x[idx_i];
            }
        }
    }
}



/**
 * @brief Perform in-place matrix transpose.
 *
 * @param ordering 'R' or 'r' for row-major. 'C' or 'c' for column-major.
 * @param A Matrix.
 * @param m Number of rows.
 * @param n Number of columns.
 */
void inplace_matrix_traspose(const char ordering, double *A, int m, int n)
{
    // TODO: implement a version without MKL!
    size_t rows = (size_t) m;
    size_t cols = (size_t) n;
    const double alpha = 1.0;

    // ! mkl_dimatcopy has large overhead!
    // mkl_dimatcopy(ordering, 'T', rows, cols, alpha, A, rows, cols);

    size_t mat_size = m*n*sizeof(double);
    double *B = malloc(mat_size);
    assert(B != NULL);

    #if defined(USE_MKL)
    mkl_domatcopy(ordering, 'T', rows, cols, alpha, A, rows, B, cols);
    memcpy(A, B, mat_size);
    #else
    // TODO: implement matrix transpose without MKL
    assert(0);
    #endif

    free(B);
}


// this function is for debugging purpose only
void print_Element(const DDBP_ELEM *E_k)
{
    const int k = E_k->index;

    printf("E[%d]->index       = %d\n", k, E_k->index      ); // element global index
    printf("E[%d]->coords      = (%d %d %d)\n", k, E_k->coords[0],E_k->coords[1],E_k->coords[2]); // element global coordinates
    printf("E[%d]->EBC[3]      = [%d %d %d]\n", k, E_k->EBCx,E_k->EBCy,E_k->EBCz); // element BCs
    printf("E[%d]->nALB        = %d\n", k, E_k->nALB       ); // total number of adaptive local basis (ALB) functions in this element
    printf("E[%d]->n_atom      = %d\n", k, E_k->n_atom     ); // number of atoms in the element
    printf("E[%d]->n_atom_ex   = %d\n", k, E_k->n_atom_ex  ); // number of atoms in the extended element

    printf("E[%d]->atom_list   = ", k);
    print_dyarray(&E_k->atom_list);
    printf("E[%d]->atom_list_ex= ", k);
    print_dyarray(&E_k->atom_list_ex);

    printf("E[%d]->n_element_nbhd = %d\n", k, E_k->n_element_nbhd); // number of neighbor elements
    // DDBP_ELEM *element_nbhd[]; // list of neighbor elements
    printf("E[%d]->element_nbhd_list = [", k); // number of neighbor elements
    for (int i = 0; i < E_k->n_element_nbhd; i++) {
        if (i > 0) printf(" ");
        printf("%d", E_k->element_nbhd_list[i]);
    }
    printf("]\n");
    // element domain/grid
    printf("E[%d]->xs          = %f\n", k, E_k->xs); // start coordinate of the element
    printf("E[%d]->xe          = %f\n", k, E_k->xe); // end coordinate of the element
    printf("E[%d]->ys          = %f\n", k, E_k->ys); // start coordinate of the element
    printf("E[%d]->ye          = %f\n", k, E_k->ye); // end coordinate of the element
    printf("E[%d]->zs          = %f\n", k, E_k->zs); // start coordinate of the element
    printf("E[%d]->ze          = %f\n", k, E_k->ze); // end coordinate of the element
    printf("E[%d]->xs_sg       = %f\n", k, E_k->xs_sg); // start coordinate of the element
    printf("E[%d]->xe_sg       = %f\n", k, E_k->xe_sg); // end coordinate of the element
    printf("E[%d]->ys_sg       = %f\n", k, E_k->ys_sg); // start coordinate of the element
    printf("E[%d]->ye_sg       = %f\n", k, E_k->ye_sg); // end coordinate of the element
    printf("E[%d]->zs_sg       = %f\n", k, E_k->zs_sg); // start coordinate of the element
    printf("E[%d]->ze_sg       = %f\n", k, E_k->ze_sg); // end coordinate of the element
    printf("E[%d]->is          = %d\n", k, E_k->is); // start index of the element in x dir
    printf("E[%d]->ie          = %d\n", k, E_k->ie); // end index of the element in x dir
    printf("E[%d]->js          = %d\n", k, E_k->js); // start index of the element in y dir
    printf("E[%d]->je          = %d\n", k, E_k->je); // end index of the element in y dir
    printf("E[%d]->ks          = %d\n", k, E_k->ks); // start index of the element in z dir
    printf("E[%d]->ke          = %d\n", k, E_k->ke); // end index of the element in z dir
    printf("E[%d]->nx          = %d\n", k, E_k->nx); // number of grid points in the element in x dir
    printf("E[%d]->ny          = %d\n", k, E_k->ny); // number of grid points in the element in y dir
    printf("E[%d]->nz          = %d\n", k, E_k->nz); // number of grid points in the element in z dir
    // extended element/grid
    printf("E[%d]->buffer_x    = %f\n", k, E_k->buffer_x); // buffer size in x dir
    printf("E[%d]->buffer_y    = %f\n", k, E_k->buffer_y); // buffer size in y dir
    printf("E[%d]->buffer_z    = %f\n", k, E_k->buffer_z); // buffer size in z dir
    printf("E[%d]->xs_ex       = %f\n", k, E_k->xs_ex); // start coordinate of the extended element
    printf("E[%d]->xe_ex       = %f\n", k, E_k->xe_ex); // end coordinate of the extended element
    printf("E[%d]->ys_ex       = %f\n", k, E_k->ys_ex); // start coordinate of the extended element
    printf("E[%d]->ye_ex       = %f\n", k, E_k->ye_ex); // end coordinate of the extended element
    printf("E[%d]->zs_ex       = %f\n", k, E_k->zs_ex); // start coordinate of the extended element
    printf("E[%d]->ze_ex       = %f\n", k, E_k->ze_ex); // end coordinate of the extended element
    printf("E[%d]->xs_ex_sg    = %f\n", k, E_k->xs_ex_sg); // start coordinate of the extended element
    printf("E[%d]->xe_ex_sg    = %f\n", k, E_k->xe_ex_sg); // end coordinate of the extended element
    printf("E[%d]->ys_ex_sg    = %f\n", k, E_k->ys_ex_sg); // start coordinate of the extended element
    printf("E[%d]->ye_ex_sg    = %f\n", k, E_k->ye_ex_sg); // end coordinate of the extended element
    printf("E[%d]->zs_ex_sg    = %f\n", k, E_k->zs_ex_sg); // start coordinate of the extended element
    printf("E[%d]->ze_ex_sg    = %f\n", k, E_k->ze_ex_sg); // end coordinate of the extended element
    printf("E[%d]->is_ex       = %d\n", k, E_k->is_ex); // start index of the extended element in x dir
    printf("E[%d]->ie_ex       = %d\n", k, E_k->ie_ex); // end index of the extended element in x dir, Caution: map back to domain for periodic BC
    printf("E[%d]->js_ex       = %d\n", k, E_k->js_ex); // start index of the extended element in y dir
    printf("E[%d]->je_ex       = %d\n", k, E_k->je_ex); // end index of the extended element in y dir, Caution: map back to domain for periodic BC
    printf("E[%d]->ks_ex       = %d\n", k, E_k->ks_ex); // start index of the extended element in z dir
    printf("E[%d]->ke_ex       = %d\n", k, E_k->ke_ex); // end index of the extended element in z dir, Caution: map back to domain for periodic BC
    printf("E[%d]->nx_ex       = %d\n", k, E_k->nx_ex); // number of grid points in the extended element in x dir
    printf("E[%d]->ny_ex       = %d\n", k, E_k->ny_ex); // number of grid points in the extended element in y dir
    printf("E[%d]->nz_ex       = %d\n", k, E_k->nz_ex); // number of grid points in the extended element in z dir
    printf("E[%d]->DMVert_ex = [", k); // number of neighbor elements
    for (int i = 0; i < 6; i++) {
        if (i > 0) printf(" ");
        printf("%d", E_k->DMVert_ex[i]);
    }
    printf("]\n");
    printf("E[%d]->nx_ex_d     = %d\n", k, E_k->nx_ex_d); // number of distributed grid points in the extended element in x dir
    printf("E[%d]->ny_ex_d     = %d\n", k, E_k->ny_ex_d); // number of distributed grid points in the extended element in y dir
    printf("E[%d]->nz_ex_d     = %d\n", k, E_k->nz_ex_d); // number of distributed grid points in the extended element in z dir
    printf("E[%d]->nd_ex_d     = %d\n", k, E_k->nd_ex_d); // number of distributed grid points in the extended element in z dir
    printf("E[%d]->DMVert = [", k); // number of neighbor elements
    for (int i = 0; i < 6; i++) {
        if (i > 0) printf(" ");
        printf("%d", E_k->DMVert[i]);
    }
    printf("]\n");
    printf("E[%d]->nx_d        = %d\n", k, E_k->nx_d); // number of distributed grid points in the extended element in x dir
    printf("E[%d]->ny_d        = %d\n", k, E_k->ny_d); // number of distributed grid points in the extended element in y dir
    printf("E[%d]->nz_d        = %d\n", k, E_k->nz_d); // number of distributed grid points in the extended element in z dir
    printf("E[%d]->nd_d        = %d\n", k, E_k->nd_d); // number of distributed grid points in the extended element in z dir
    // print element SPARC object
    SPARC_OBJ *ESPRC_k = E_k->ESPRC;
    char size_bytes[16];
    formatBytes(sizeof(*ESPRC_k),16,size_bytes);
    printf("size of E[%d]->ESPRC  = %s\n", k, size_bytes);
    printf("E[%d]->ESPRC->range_x = %f\n", k, ESPRC_k->range_x);
    printf("E[%d]->ESPRC->range_y = %f\n", k, ESPRC_k->range_y);
    printf("E[%d]->ESPRC->range_z = %f\n", k, ESPRC_k->range_z);
    printf("E[%d]->ESPRC->order   = %d\n", k, ESPRC_k->order);
    printf("E[%d]->ESPRC->BCx     = %d\n", k, ESPRC_k->BCx);
    printf("E[%d]->ESPRC->BCy     = %d\n", k, ESPRC_k->BCy);
    printf("E[%d]->ESPRC->BCz     = %d\n", k, ESPRC_k->BCz);
    printf("E[%d]->ESPRC->LatVec = [%.6g",k, ESPRC_k->LatVec[0]);
    for (int i = 1; i < 9; i++) {
        printf(" %.6g", ESPRC_k->LatVec[i]);
        if (i == 2 || i == 5) printf(";");
    }
    printf("]\n");
    printf("E[%d]->ESPRC->LatUVec = [%.6g",k, ESPRC_k->LatUVec[0]);
    for (int i = 1; i < 9; i++) {
        printf(" %.6g", ESPRC_k->LatUVec[i]);
        if (i == 2 || i == 5) printf(";");
    }
    printf("]\n");
}




void print_array(const void *array, int n, size_t type_size) {
    printf("[");
    for (int i  = 0; i < n; i++) {
        if (i) printf(", ");
        if (type_size == sizeof(int)) {
            printf("%d", *((const int*) array+i));
        } else if (type_size == sizeof(double)) {
            printf("%.3f", *((const double*) array+i));
        }
    }
    printf("]\n");
}


void show_array(double *array, int n) {
    if (n <= 10) {
        printf("[");
        for (int i = 0; i < n; i++) {
            if (i > 0) printf(" ");
            printf("%7.3f", array[i]);
        }
        printf("], len = %d)\n",n);
        return;
    }

    printf("([");
    for (int i = 0; i < 5; i++) {
        if (i > 0) printf(" ");
        printf("%7.3f", array[i]);
    }
    printf(" ...");
    for (int i = n-5; i < n; i++) {
        if (i > 0) printf(" ");
        printf("%7.3f", array[i]);
    }
    printf("], len = %d)\n",n);
}


// show matrix, row major
// *if in col maj, this shows the transpose of the matrix
void show_mat(double *array, int m, int n)
{
    printf("[\n");
    for (int i = 0; i < m; i++) {
        show_array(array + n * i, n);
    }
    printf("], mat_size = (%d,%d)\n",m,n);
}


void print_haloX(DDBP_ELEM *E_k, haloX_t *haloX, MPI_Comm kptcomm)
{
    int my_rank;
    MPI_Comm_rank(kptcomm, &my_rank);

    printf("Printing haloX for element %d\n", E_k->index);
    printf("my_rank = %d\n", my_rank);

    printf("neighbor_indices = ");
    print_array(haloX->neighbor_indices, 6, sizeof(int));
    printf("neighbor_ranks = ");
    print_array(haloX->neighbor_ranks, 6, sizeof(int)); // ranks (kptcomm) of the neighbor processes to transfer data with
    printf("sendcounts = ");
    print_array(haloX->sendcounts, 6, sizeof(int)); // the number of elements to send to neighbor i
    printf("recvcounts = ");
    print_array(haloX->recvcounts, 6, sizeof(int)); // the number of elements to receive from neighbor i
    printf("sdispls = ");
    print_array(haloX->sdispls, 6, sizeof(int)); // the displacement (offset from sbuf) from which to send
    printf("rdispls = ");
    print_array(haloX->rdispls, 6, sizeof(int)); // the displacement (offset from rbuf) to which data from
    printf("stags = ");
    print_array(haloX->stags, 6, sizeof(int)); // send tags
    printf("rtags = ");
    print_array(haloX->rtags, 6, sizeof(int)); // recv tags
    // MPI_Datatype sendtype;
    // MPI_Datatype recvtype;
    // MPI_Request requests[12]; // first 6 for sending, last 6 for receiving
}


// this function is for debugging purpose only
void print_DDBP_info(DDBP_INFO *DDBP_info)
{
    printf("DDBP_info->Nex   = %d\n", DDBP_info->Nex);
    printf("DDBP_info->Ney   = %d\n", DDBP_info->Ney);
    printf("DDBP_info->Nez   = %d\n", DDBP_info->Nez);
    printf("DDBP_info->Ne_tot    = %d\n", DDBP_info->Ne_tot);
    printf("DDBP_info->buffer_x  = %f\n", DDBP_info->buffer_x);
    printf("DDBP_info->buffer_y  = %f\n", DDBP_info->buffer_y);
    printf("DDBP_info->buffer_z  = %f\n", DDBP_info->buffer_z);
    printf("DDBP_info->BCx  = %d\n", DDBP_info->BCx);
    printf("DDBP_info->BCy  = %d\n", DDBP_info->BCy);
    printf("DDBP_info->BCz  = %d\n", DDBP_info->BCz);
    printf("DDBP_info->EBCx = %d\n", DDBP_info->EBCx);
    printf("DDBP_info->EBCy = %d\n", DDBP_info->EBCy);
    printf("DDBP_info->EBCz = %d\n", DDBP_info->EBCz);
    printf("DDBP_info->nALB_atom  = %d\n", DDBP_info->nALB_atom);
    printf("DDBP_info->nALB_tot   = %d\n", DDBP_info->nALB_tot);
    printf("DDBP_info->n_atom = %d\n", DDBP_info->n_atom);

    // int *atom_types;
    printf("DDBP_info->atom_types = [");
    for (int i = 0; i < DDBP_info->n_atom; i++) {
        if (i > 0) printf(" ");
        printf("%d", DDBP_info->atom_types[i]);
    }
    printf("]\n");

    // double *rcs_x; // the size of the rc box in x dir
    printf("DDBP_info->rcs_x = [");
    for (int i = 0; i < DDBP_info->n_atom; i++) {
        if (i > 0) printf(" ");
        printf("%.2f", DDBP_info->rcs_x[i]);
    }
    printf("]\n");

    // double *rcs_x; // the size of the rc box in x dir
    printf("DDBP_info->rcs_y = [");
    for (int i = 0; i < DDBP_info->n_atom; i++) {
        if (i > 0) printf(" ");
        printf("%.2f", DDBP_info->rcs_y[i]);
    }
    printf("]\n");

    // double *rcs_z; // the size of the rc box in x dir
    printf("DDBP_info->rcs_z = [");
    for (int i = 0; i < DDBP_info->n_atom; i++) {
        if (i > 0) printf(" ");
        printf("%.2f", DDBP_info->rcs_z[i]);
    }
    printf("]\n");

    printf("DDBP_info->Lex   = %f\n", DDBP_info->Lex); // element side in x dir
    printf("DDBP_info->Ley   = %f\n", DDBP_info->Ley); // element side in y dir
    printf("DDBP_info->Lez   = %f\n", DDBP_info->Lez); // element side in z dir
    // MPI_Comm elemcomm; // communicator for the parallelization of DDBP elements
    printf("DDBP_info->npelem           = %d\n", DDBP_info->npelem); // number of elemcomms
    printf("DDBP_info->npbasis          = %d\n", DDBP_info->npbasis); // number of basiscomms
    printf("DDBP_info->npdm             = %d\n", DDBP_info->npdm); // number of basiscomms
    printf("DDBP_info->dmcomm_dims      = [%d,%d,%d]\n",
        DDBP_info->dmcomm_dims[0],DDBP_info->dmcomm_dims[1],DDBP_info->dmcomm_dims[2]); // number of basiscomms

    printf("DDBP_info->elemcomm_index   = %d\n", DDBP_info->elemcomm_index  ); // index of the current elemcomm
    printf("DDBP_info->elem_start_index = %d\n", DDBP_info->elem_start_index); // start index of the assigned elements
    printf("DDBP_info->elem_end_index   = %d\n", DDBP_info->elem_end_index  ); // end index of the assigned elements
    printf("DDBP_info->n_elem_elemcomm  = %d\n", DDBP_info->n_elem_elemcomm ); // total number of elements assigned to the current elemcomm

    // MPI_Comm basiscomm; // communicator for the parallelization of DDBP elements

    printf("DDBP_info->basiscomm_index   = %d\n", DDBP_info->basiscomm_index  ); // index of the current basiscomm
    printf("DDBP_info->basis_start_index = %d\n", DDBP_info->basis_start_index); // start index of the assigned basis functions
    printf("DDBP_info->basis_end_index   = %d\n", DDBP_info->basis_end_index  ); // end index of the assigned basis functions
    printf("DDBP_info->n_basis_basiscomm = %d\n", DDBP_info->n_basis_basiscomm); // total number of basis functions assigned to the current basiscomms
}


extern void usleep();
void break_point(MPI_Comm comm, int index)
{
    int nproc;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    for (int i = 0; i < nproc; i++) {
    MPI_Barrier(comm);
    if (i == rank) {
        printf("rank = %2d, breaking point %d\n",rank, index);
    }
        // usleep(100000);
        MPI_Barrier(comm);
    }
    MPI_Barrier(comm);
    usleep(50000);
    MPI_Barrier(comm);
}

void Ibreak_point(int index)
{
    int nproc;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    printf("rank = %2d, Ibreaking point %d\n",rank, index);
    usleep(50000);
}

