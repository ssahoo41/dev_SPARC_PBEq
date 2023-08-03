/**
 * @file    ddbp_basis.h
 * @brief   This file contains the function declarations for the Discrete 
 *          Discontinuous Basis Projection (DDBP) basis generation.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef DDBP_BASIS_H
#define DDBP_BASIS_H 

#include "isddft.h"


/**
 * @brief  Set up nonlocal projectors for all DDBP (extended) elements.
 *
 *         This has to be done every time the atom positions are updated.
 */
void setup_ddbp_element_Vnl(SPARC_OBJ *pSPARC);


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
    void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm kptcomm, int proc_active);


/**
 * @brief  Set up local effective potentials for all DDBP (extended) elements.
 *
 *         This has to be done every time density is updated (in each SCF).
 */
void setup_ddbp_element_Veff(SPARC_OBJ *pSPARC);


/**
 * @brief   Find DDBP basis in the extended element of E_k using CheFSI.
 */
void find_extended_element_basis_CheFSI(DDBP_INFO *DDBP_info, DDBP_ELEM *E_k,
    double *x0, int count, int kpt, int spn_i);


/**
 * @brief   Restrict DDBP basis from the extended element to the element.
 */
void restrict_basis_to_element(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, int kpt, int spn_i
);


/**
 * @brief   Orthogonalize the restricted DDBP basis within the element.
 *
 *          Warning: This function assumes no domain paral for basis calculation.
 */
void orth_restricted_basis(
    DDBP_INFO *DDBP_info, DDBP_ELEM *E_k, int kpt, int spn_i
);


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
    int isInitSCF, int kpt, int spn_i);


/**
 * @brief   Calculate DDBP basis for all elements.
 */
void Calculate_DDBP_basis(SPARC_OBJ *pSPARC, int count, int kpt, int spn_i);


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
);


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
    DDBP_INFO *DDBP_info, int nkpt, int isInitSCF, int kpt, int spn_i);


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
    DDBP_INFO *DDBP_info, DDBP_ARRAY *X, int isInitSCF, int kpt, int spn_i);


#endif // DDBP_BASIS_H
