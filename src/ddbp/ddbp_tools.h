/**
 * @file    ddbp_tools.h
 * @brief   This file contains the function declarations for the Discrete
 *          Discontinuous Basis Projection (DDBP) method.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef _DDBP_TOOLS_H
#define _DDBP_TOOLS_H

#include "isddft.h"



/**
 * @brief   Find the index of a DDBP element based on its Cartesian coordinates.
 *                      (ii_E, jj_E, kk_E) -> index.
 *
 * @param dims   The dimensions of the elements (number of elements in all dir's).
 * @param coords The coordinates of the element.
 * @param index  The index of the the element.
 */
void DDBP_Cart_Index(const int dims[3], const int coords[3], int *index);



/**
 * @brief   Find the Cartesian coordinates of a DDBP element based on its index.
 *                      index -> (ii_E, jj_E, kk_E).
 *
 * @param dims   The dimensions of the elements (number of elements in all dir's).
 * @param index  The index of the the element.
 * @param coords The coordinates of the element.
 */
void DDBP_Index_Cart(const int dims[3], const int index, int coords[3]);


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
);


/**
 * @brief   Calculates start node of an element owned by  
 *          the process (in one direction).
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param bc    Boundary condition in the given direction. 0 - PBC, 1 - DBC.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int element_decompose_nstart(const int n, const int p, const int bc, const int rank);


/**
 * @brief   Calculates end node of an element owned by  
 *          the process (in one direction).
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param bc    Boundary condition in the given direction. 0 - PBC, 1 - DBC.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int element_decompose_nend(const int n, const int p, const int bc, const int rank);


/**
 * @brief   Calculates numbero of nodes of an element owned by  
 *          the process (in one direction).
 *
 * @param n     Number of nodes in the given direction of the global domain.
 * @param p     Total number of processes in the given direction of the process topology.
 * @param bc    Boundary condition in the given direction. 0 - PBC, 1 - DBC.
 * @param rank  Rank of the process in possession of a distributed domain.
 */
int element_decompose(const int n, const int p, const int bc, const int rank);


/**
 * @brief   Calculates which process owns the provided node of an
 *          element (in one direction).
 */
int element_decompose_rank(
    const int n, const int p, const int bc, const int node_indx);


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
 *                        figure out.)
 * @param dmcomm_rank     Rank of process in the dmcomm.
 */
int DDBP_basis_owner(
    DDBP_INFO *DDBP_info, int k, int n, int basiscomm_index,
    int dmcomm_rank, MPI_Comm kptcomm
);


/**
 * @brief   Check if a coordinate lies in a cuboid region.
 */
int is_atom_in_region(const double atom_pos[3], const double vert[6]);


/**
 * @brief  For a given atom and it's influence radius, check if it or its
 *         images affect the given region within the cell.
 */
int atom_images_in_region(
    const double atom_pos[3], const double rc[3], const double cell[3],
    const int BCs[3], const double vert[6], double **image_coords);


/**
 * @brief Initialize the dynamic array, allocate initial
 *        memory and set size.
 *
 * @param initsize  Initial size of memory to be allocated.
 */
void init_dyarray(dyArray *a);


/**
 * @brief Append an element to the dynamic array.
 *
 */
void append_dyarray(dyArray *a, value_type element);


/**
 * @brief Pop the last element from the dynamic array.
 *
 */
value_type pop_dyarray(dyArray *a);


/**
 * @brief Clear the dynamic array.
 *
 *        This function does not destroy the array, it simply
 *        resets the lenght of the dynamic array to 0, and resets
 *        the capacity.
 */
void clear_dyarray(dyArray *a);


/**
 * @brief Delete the dynamic array.
 *
 */
void delete_dyarray(dyArray *a);


//* for debugging purpose *//
void print_dyarray(const dyArray *a);
void show_dyarray(const dyArray *a);


/**
 * @brief Create a DDBP Array object
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
    int nelem, DDBP_ELEM *elem_list, int ncol, DDBP_ARRAY *X);


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
    DDBP_INFO *DDBP_info, int ncol, DDBP_ARRAY *X, MPI_Comm comm);



/**
 * @brief Duplicate a DDBP Array template.
 *
 *        Create a new DDBP of the same size and same distribution as
 *        the template. This routine sets up the parameters and the halo
 *        exchange info, and allocates memory.
 * 
 * @param X Template DDBP Array.
 * @param Y New DDBP Array to be set up.
 */
void duplicate_DDBP_Array_template(const DDBP_ARRAY *X, DDBP_ARRAY *Y);


/**
 * @brief Create a deep copy of DDBP Array, including the array values.
 * 
 * @param X Source DDBP Array.
 * @param Y New copy of DDBP Array.
 */
void deepcopy_DDBP_Array(const DDBP_ARRAY *X, DDBP_ARRAY *Y);



/**
 * @brief Randomize a DDBP Array.
 * 
 * @param X DDBP Array.
 * @param comm Communicator where X is distributed, the random
 *             seeds are generated based on the rank and shift.
 */
void randomize_DDBP_Array(DDBP_ARRAY *X, MPI_Comm comm);


/**
 * @brief Delete a DDBP Array object.
 * 
 * @param X DDBP Array to be deleted.
 */
void delete_DDBP_Array(DDBP_ARRAY *X);


/**
 * @brief Scale X, X = a * X.
 * 
 * @param a Scalar a.
 * @param X DDBP array X.
 */
void scale_DDBP_Array(double a, DDBP_ARRAY *X);


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
    double a, const DDBP_ARRAY *X, double b, DDBP_ARRAY *Y);


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
    double c, DDBP_ARRAY *Z);


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
    int ncol, double *results, MPI_Comm comm);


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
    const DDBP_ARRAY *X, int ncol, double *results, MPI_Comm comm);


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
    int *descY, double beta, double *M, int *descM, MPI_Comm rowcomm, MPI_Comm colcomm);


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
    int *descQ, double beta, DDBP_ARRAY *Y, int *descY, MPI_Comm rowcomm);


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
    double *yk, int *descyk, MPI_Comm rowcomm);


int double_check_arrays(const double *a, const double *b, const int n);
int double_check_int_arrays(const int *a, const int *b, const int n);
int double_check_DDBP_arrays(
    const DDBP_ARRAY *X, const DDBP_ARRAY *Y, const int is, const int ncol);

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
);


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
);


/**
 * @brief Restrict any function defined on the extended element
 *        to the element.
 *
 */
void restrict_to_element(const DDBP_ELEM *E_k, const double *x_ex, double *x);


// /**
//  * @brief Extend any function defined on the element to the extended element and
//  *        leave the extended part untouched.
//  *
//  */
// void extend_to_extended_element(const DDBP_ELEM *E_k, const double *x, double *x_ex);


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
//     int nx_ex, int ny_ex, int nz_ex);


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
//     int nx_ex, int ny_ex, int nz_ex);


/**
 * @brief Perform in-place matrix transpose.
 * 
 * @param ordering 'R' or 'r' for row-major. 'C' or 'c' for column-major.
 * @param A Matrix.
 * @param m Number of rows.
 * @param n Number of columns.
 */
void inplace_matrix_traspose(const char ordering, double *A, int m, int n);


/**
 * @brief Find optimal number of processes to use for performing
 *        a parallel matrix operation using scalapack routines.
 * 
 * @param m Size of the matrix, number of rows.
 * @param n Size of the matrix, number of columns.
 * @param typ Operation type. Options: "pdgemm".
 * @return int Maximum number of processes to be used.
 */
int best_max_nproc(int m, int n, char *typ);


// this function is for debugging purpose only
void print_Element(const DDBP_ELEM *E_k);

void print_array(const void *array, int n, size_t type_size);

// this function is for debugging purpose only
void print_DDBP_info(DDBP_INFO *DDBP_info);

// show matrix, row major
// *if in col maj, this shows the transpose of the matrix
void show_mat(double *array, int m, int n);

void print_haloX(DDBP_ELEM *E_k, haloX_t *haloX, MPI_Comm kptcomm);


// blocking break point, with Barrier
void break_point(MPI_Comm comm, int index);
// non-blocking break point, w/o Barrier
void Ibreak_point(int index);

#endif // _DDBP_TOOLS_H

