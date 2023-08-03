/**
 * @file    ddbp_nloc.h
 * @brief   This file contains the function declarations for the nonlocal
 *          routines for the Discrete Discontinuous Basis Projection
 *          (DDBP) method.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2021 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef _DDBP_NLOC_H
#define _DDBP_NLOC_H

#include "isddft.h"


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
    int proc_active);


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
    int *localPsd, int *IP_displ, PSD_OBJ *psd, double dV);


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
    int ldX, double beta, double *ChiX, int ldChiX, int ncol);


/**
 * @brief   Calculate the nonlocal projectors for the DDBP Hamiltonian.
 *
 *          The nonlocal part of DDBP Hailtonian is defined by
 *                        Vnl_ddbp := V^T * Vnl * V,
 *          where Vnl = sum_Jlm gamma_Jl |X_Jlm> <X_Jlm| is the
 *          nonlocal pseudopotential operator.
 *            Substituting the Vnl expression into Vnl_ddbp, we
 *          obtain
 *            Vnl_ddbp = sum_Jlm gamma_Jl V^T|X_Jlm> <X_Jlm|V,
 *                     = sum_Jlm gamma_Jl |X'_Jlm> <X'_Jlm|,
 *          where |X'_Jlm> = V^T |X_Jlm> is the nonlocal
 *          projectors expressed in the DDBP basis.
 */
void Calculate_nloc_projectors_DDBP_Hamiltonian(
    SPARC_OBJ *pSPARC, int count, int kpt, int spn_i
);

#endif // _DDBP_NLOC_H
