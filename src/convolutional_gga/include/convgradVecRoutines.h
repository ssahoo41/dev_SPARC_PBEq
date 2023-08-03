/**
 * @file    convgradVecRoutines.h
 * @brief   This file contains declaration of functions required for convolutional GGA.
 *
 * @author  Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
 *          Andrew J. Medford <ajm@gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

#include "MP_types.h"

/**
 * @brief   function to calculate gradient of electron density using 3D convolutions in
 *  the given direction. (A function similar to this is used for GGA_PBE in SPARC)
 *
 */
void Conv_gradient_vectors_dir(SPARC_OBJ *pSPARC, int *DMVertices, const int ncol, 
				const double *x, double *Dx, const int dir, MPI_Comm comm);


/**
 * @brief  function to set up the 3D stencil for GGA_CONV_PBE
 *
 */
void calculate_GGA_FD_Stencil(SPARC_OBJ *pSPARC, const int dir, const int stencilDimX, 
                            const int stencilDimY, const int stencilDimZ, double *stencil);

/**
 * @brief   function to calculate the convolutiosn using HSMP stencils by setting up the stencil and
 * 			then calling ConvolveAndAddResult function. This function needs MULTIPOLE_OBJ to be
 * 			initialized first.
 *
 */
void Conv_feat_vectors_dir(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp, int *DMVertices, const int ncol,
			const double *x, double *Dx, const char *n, MPI_Comm comm);


/**
 * @brief   Convolution function that can be used for GGA_CONV_PBE and GGA_CONV_PBE_MULTIPOLE. 
 * This function requires electron density (image) and stencil to do the convolution and the 
 * result of convolution is distributed in pSPARC->dmcomm_phi (phi-domain).
 *
 */
void ConvolveAndAddResult(SPARC_OBJ *pSPARC, const int stencilDimX, const int stencilDimY, const int stencilDimZ,
			const double *x, double *Dx, const double *stencil, MPI_Comm comm, int *DMVertices);

