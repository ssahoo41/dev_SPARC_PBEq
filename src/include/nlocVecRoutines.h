/**
 * @file    nlocVecRoutines.h
 * @brief   This file contains function declarations for nonlocal components.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Abhiraj Sharma <asharma424@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *          Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */
 
#ifndef NLOCVECROUTINES_H
#define NLOCVECROUTINES_H 

#include "isddft.h"


/**
 * @brief   Find the list of all atoms that influence the processor 
 *          domain in psi-domain. Allows input of different BCs and
 *          cell dimensions for defining the image atoms.
 * @param pSPARC 
 * @param cell_orig Origin of the shifted cell, over which images are generated.
 * @param cell Dimensions of the shifted cell, over which images are generated.
 * @param BCs Boundary conditions of the shifted cell.
 * @param Atom_Influence_nloc (Output) Object containing info of atoms that has nloc
 *                            influence on the local process domain.
 * @param DMVertices Domain vertices.
 * @param comm Communicator over which the domain is distributed.
 */
void GetInfluencingAtoms_nloc_kernel(
    SPARC_OBJ *pSPARC, double cell_orig[3], double cell[3], int BCs[3],
    ATOM_NLOC_INFLUENCE_OBJ **Atom_Influence_nloc, int *DMVertices, MPI_Comm comm);


/**
 * @brief   Find the list of all atoms that influence the processor 
 *          domain in psi-domain.
 */
void GetInfluencingAtoms_nloc(SPARC_OBJ *pSPARC, ATOM_NLOC_INFLUENCE_OBJ **Atom_Influence_nloc, 
                              int *DMVertices, MPI_Comm comm);


/**
 * @brief   Calculate nonlocal projectors. 
 */
void CalculateNonlocalProjectors(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ **nlocProj,  
     ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, int *DMVertices, MPI_Comm comm);

void CalculateNonlocalProjectors_kpt(SPARC_OBJ *pSPARC, NLOC_PROJ_OBJ **nlocProj, 
        ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, int *DMVertices, MPI_Comm comm);
/**
 * @brief   Calculate indices for storing nonlocal inner product in an array. 
 *
 *          We will store the inner product < Chi_Jlm, x_n > in a continuous array "alpha",
 *          the dimensions are in the order: <lm>, n, J. Here we find out the sizes of the 
 *          inner product corresponding to atom J, and the total number of inner products
 *          corresponding to each vector x_n.
 */
void CalculateNonlocalInnerProductIndex(SPARC_OBJ *pSPARC);


/**
 * @brief   Calculate Vnl times vectors in a matrix-free way.
 */
void Vnl_vec_mult(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                  NLOC_PROJ_OBJ *nlocProj, int ncol, double *x, double *Hx, MPI_Comm comm);



/**
 * @brief   Calculate Vnl times vectors in a matrix-free way with Bloch factor
 */
void Vnl_vec_mult_kpt(const SPARC_OBJ *pSPARC, int DMnd, ATOM_NLOC_INFLUENCE_OBJ *Atom_Influence_nloc, 
                      NLOC_PROJ_OBJ *nlocProj, int ncol, double _Complex *x, double _Complex *Hx, int kpt, MPI_Comm comm);
#endif // NLOCVECROUTINES_H


