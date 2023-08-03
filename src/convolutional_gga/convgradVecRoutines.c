/**
 * @file    convgradVecRoutines.c
 * @brief   This file contains declaration of functions required for convolutional GGA.
 *
 * @author  Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
 *          Andrew J. Medford <ajm@gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
#include <assert.h>
#include <time.h> 
#include <mpi.h>
#include <math.h>
#include <assert.h>
/* BLAS, LAPACK, LAPACKE routines */
#ifdef USE_MKL
    // #define MKL_Complex16 double complex
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#include "MCSHHelper.h"
#include "MCSH.h"
#include "gradVecRoutines.h"
#include "isddft.h"
#include "MP_types.h"
#include "convgradVecRoutines.h"

/**
 * @brief   function to calculate gradient of electron density using 3D convolutions in
 *  the given direction. (A function similar to this is used for GGA_PBE in SPARC)
 *
 */
void Conv_gradient_vectors_dir(SPARC_OBJ *pSPARC, int *DMVertices,
				const int ncol, const double *x, double *Dx, const int dir, MPI_Comm comm){
	int worldRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

	int stencilDimX = pSPARC->order + 1;
	int stencilDimY = pSPARC->order + 1;
	int stencilDimZ = pSPARC->order + 1;
	int Nd = pSPARC->Nd;
	int DMnd = pSPARC->Nd_d;
	int stencilSize = stencilDimX * stencilDimY * stencilDimZ;
	double *stencil = calloc(stencilSize, sizeof(double));
	
	double start_stencil_t, end_stencil_t;
	
	start_stencil_t = MPI_Wtime();
	calculate_GGA_FD_Stencil(pSPARC, dir, stencilDimX, stencilDimY, stencilDimZ, stencil);
	end_stencil_t = MPI_Wtime();

    	for (int i = 0; i < ncol; i++)
			ConvolveAndAddResult(pSPARC, stencilDimX, stencilDimY, stencilDimZ, x+i*(unsigned)Nd, Dx+i*(unsigned)DMnd, stencil, comm, DMVertices);
}

/**
 * @brief  function to set up the 3D stencil for GGA_CONV_PBE
 *
 */
void calculate_GGA_FD_Stencil(SPARC_OBJ *pSPARC, const int dir, const int stencilDimX,
								const int stencilDimY, const int stencilDimZ, double *stencil)
{
	int FDn = pSPARC->order/2;
	
	int stencilCenterX = (stencilDimX - 1) / 2;
	int stencilCenterY = (stencilDimY - 1) / 2;
	int stencilCenterZ = (stencilDimZ - 1) / 2;

	double *D1_stencil_coeffs_dim;
	D1_stencil_coeffs_dim = (double *)malloc((2*FDn + 1) * sizeof(double));
	
	D1_stencil_coeffs_dim[FDn] = 0.0;
	int p;
	switch(dir) {
		case 0:
			for (p = 1; p <= FDn; p++){
				D1_stencil_coeffs_dim[FDn - p] = -1 * (pSPARC->D1_stencil_coeffs_x[p]);
				D1_stencil_coeffs_dim[p + FDn] = pSPARC->D1_stencil_coeffs_x[p];
			}
			break;
		case 1:
			for (p = 1; p <= FDn; p++){
				D1_stencil_coeffs_dim[FDn - p] = -1 * (pSPARC->D1_stencil_coeffs_y[p]);
				D1_stencil_coeffs_dim[p + FDn] = pSPARC->D1_stencil_coeffs_y[p];
			}
			break;
		case 2:
			for (p = 1; p<= FDn; p++){
				D1_stencil_coeffs_dim[FDn - p] = -1 * (pSPARC->D1_stencil_coeffs_z[p]);
				D1_stencil_coeffs_dim[p + FDn] = pSPARC->D1_stencil_coeffs_z[p];
			}
			break;
		default: printf("gradient dir must be either 0, 1 or 2!\n");
                 break;	
	}
		
	switch(dir) {
		case 0:
			{
				int i;
				int index_x = 0; //x-direction stencil
				for (i = 0; i < stencilDimX; i++){
					index_x = stencilDimX * stencilDimY * stencilCenterZ + stencilDimX * stencilCenterY + i;
					stencil[index_x] = D1_stencil_coeffs_dim[i];
				}
			}
			break;

		case 1:
			{
				int j;
				int index_y = 0; //y-direction stencil
				for (j = 0; j < stencilDimY; j++){
					index_y = stencilDimX * stencilDimY * stencilCenterZ + stencilDimX * j + stencilCenterX;
					stencil[index_y] = D1_stencil_coeffs_dim[j];
				}
			}
			break;

		case 2:
			{
				int k;
				int index_z = 0; //z-direction stencil
				for (k = 0; k < stencilDimZ; k++){
					index_z = stencilDimX * stencilDimY * k + stencilDimX * stencilCenterY + stencilCenterX;
					stencil[index_z] = D1_stencil_coeffs_dim[k];
				}
			}
			
			break;
		default: printf("gradient dir must be either 0, 1 or 2!\n");
    		break;	
	}
}

/**
 * @brief   function to calculate the convolution using HSMP stencils by setting up the stencil and
 * 			then calling ConvolveAndAddResult function. This function needs MULTIPOLE_OBJ to be
 * 			initialized first.
 *
 */
void Conv_feat_vectors_dir(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp, int *DMVertices, const int ncol,
			const double *x, double *Dx, const char *n, MPI_Comm comm){
	
	int Nd = pSPARC->Nd;
	int DMnd = pSPARC->Nd_d;
	
	int worldRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

	double normalizedU[9];
	normalizeU(mp->U, normalizedU);

	double start_stencil_t, end_stencil_t;
	
	int stencilDimX, stencilDimY, stencilDimZ;
	// GetDimensionsPlane(mp, 0.5, normalizedU, &stencilDimX, &stencilDimY, &stencilDimZ);
	GetDimensionsPlane(mp, normalizedU, &stencilDimX, &stencilDimY, &stencilDimZ);

	double *stencil = calloc(stencilDimX * stencilDimY * stencilDimZ, sizeof(double));

	start_stencil_t = MPI_Wtime();
	if (strcmp(n, "000") == 0){
		calculateStencil(stencilDimX, stencilDimY, stencilDimZ, mp->hx, mp->hy, mp->hz, mp->MCSHMaxR, 0, n, 1, 0, normalizedU, mp->accuracy, stencil);
	}
	else{
		calculateStencil(stencilDimX, stencilDimY, stencilDimZ, mp->hx, mp->hy, mp->hz, mp->MCSHMaxR, 1, n, 1, 0, normalizedU, mp->accuracy, stencil);
	}
	
	end_stencil_t = MPI_Wtime();
	
	for (int i = 0; i < ncol; i++)
		ConvolveAndAddResult(pSPARC, stencilDimX, stencilDimY, stencilDimZ, x+i*(unsigned)Nd, Dx+i*(unsigned)DMnd, stencil, comm, DMVertices);
}

/**
 * @brief   Convolution function that can be used for GGA_CONV_PBE and GGA_CONV_PBE_MULTIPOLE. 
 * This function requires electron density (image) and stencil to do the convolution and the 
 * result of convolution is distributed in pSPARC->dmcomm_phi (phi-domain).
 *
 */
void ConvolveAndAddResult(SPARC_OBJ *pSPARC, const int stencilDimX, const int stencilDimY, const int stencilDimZ,
			const double *x, double *Dx, const double *stencil, MPI_Comm comm, int *DMVertices)
{
	int worldRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

	int imageDimX = pSPARC->Nx;
	int imageDimY = pSPARC->Ny;
	int imageDimZ = pSPARC->Nz;
	double start_convolve_t, end_convolve_t;
	start_convolve_t = MPI_Wtime();
		
    // this takes DMVerts (coordinates but image dim is imageDimX * imageDimY * imageDimZ)
	convolve6(x, stencil, imageDimX, imageDimY, imageDimZ, stencilDimX, stencilDimY, stencilDimZ, Dx, DMVertices);
		
	// free(convolveResult);
	end_convolve_t  = MPI_Wtime();
	if (worldRank == 0){
		printf("Total time for convolution: \t convolve: %f \n", end_convolve_t - start_convolve_t);
	}
}

