/**
 * @file    ddbp_hamiltonian.h
 * @brief   This file contains the function declarations for the Discrete
 *          Discontinuous Basis Projection (DDBP) Hamiltonian routines.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef _DDBP_HAMILTONIAN_H
#define _DDBP_HAMILTONIAN_H

#include "isddft.h"

/**
 * @brief   Calculate DDBP Hamiltonian.
 *
 *          The DDBP Hailtonian is defined by
 *                        H_ddbp := V^T * H * V,
 *          where H := -1/2 D^2 + Veff + Vnl is the global Hamiltonian, V is the
 *          DDBP basis (a block-diagonal matrix). Since the Hamiltonian H is
 *          sparse (neglecting Vnl), the resulting H_ddbp turns out to be a
 *          block-sparse matrix. Therefore, we only calculate the non-zero blocks
 *          of H_ddbp.
 */
void Calculate_DDBP_Hamiltonian(SPARC_OBJ *pSPARC, int count, int kpt, int spn_i);


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
void Update_Veff_part_of_DDBP_Hamiltonian(SPARC_OBJ *pSPARC, int count, int kpt, int spn_i);


#endif // _DDBP_HAMILTONIAN_H
