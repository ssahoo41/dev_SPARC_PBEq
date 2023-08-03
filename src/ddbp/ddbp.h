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

#ifndef _DDBP_H
#define _DDBP_H

#include "isddft.h"
#include "ddbp_tools.h"
#include "ddbp_paral.h"
#include "ddbp_basis.h"
#include "ddbp_hamiltonian.h"
#include "ddbp_nloc.h"
#include "ddbp_matvec.h"
#include "ddbp_eigensolver.h"


/**
 * @brief   Initialize DDBP.
 */
void init_DDBP(SPARC_OBJ *pSPARC);


/**
 * @brief   Create the DDBP_info structure and store the relevant parameters.
 */
void Create_DDBP_info(SPARC_OBJ *pSPARC);


/**
 * @brief   Free the DDBP_info structure.
 */
void free_DDBP_info(SPARC_OBJ *pSPARC);


/**
 * @brief   Free the DDBP related variables that has to be reset for SCF.
 */
void free_DDBP_scfvar(SPARC_OBJ *pSPARC);


/**
 * @brief   Assign atoms to the elements and extended elements.
 */
void ddbp_assign_atoms(
    const double *atom_pos, const int n_atom, const double cell[3],
    const int BCs[3], DDBP_INFO *DDBP_info);


/**
 * @brief  Set up nonlocal projectors for all DDBP (extended) elements.
 *
 *         This has to be done every time the atom positions are updated.
 */
void setup_ddbp_element_Vnl(SPARC_OBJ *pSPARC);


/**
 * @brief Calculate DDBP Basis and the DDBP Hamiltonian to prepare for CheFSI_DDBP.
 *
 *        This routine does the following things to prepare for
 *        CheFSI on the DDBP Hamiltonian:
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
    SPARC_OBJ *pSPARC, int count, int kpt, int spn_i);


/**
 * @brief   CheFSI with DDBP method.
 */
void CheFSI_DDBP(SPARC_OBJ *pSPARC, double lambda_cutoff, double *x0, int count, int k, int spn_i);


#endif // _DDBP_H

