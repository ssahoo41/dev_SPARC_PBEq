/**
* @file MCSH.h
* @brief This file contains the declaration of MCSH functions.
*
* @author Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
*		Andrew J. Medford <ajm@gatech.edu>
*
* Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
*/

#include "MP_types.h"

/**
* @brief  function to evaluate the MCSH polynomials
*/
void evaluateSeriesPolynomial(const double *x, const double *y, const double *z, const double *parameters, const int *polyTypes, const int numTerm, const int arraySize, double *result);

/**
* @brief  function to define the parameters for different orders and groups of MCSH functions
*/
void MaxwellCartesianSphericalHarmonics(const double *x, const double *y, const double *z, const int l, const char *n, const double rCutoff, double *result, const int size);

/**
* @brief  function to define the parameters for orders of Legendre polynomials
*/
void LegendrePolynomial(const double *x, const double *y, const double *z, const int polynomialOrder, const double rCutoff, double *result, const int size);

/**
* @brief  function to calculate the components (groups) for a given angular and radial order.
*
*/
void calcFeature( const int type, const double *image, MULTIPOLE_OBJ *mp, const double rCutoff, const int l, char **n_list,
		const int radialFunctionType, const int radialFunctionOrder, const double *U, double coeff, 
		double *componentGroup, int DMVerts[6]);

/**
* @brief  function to calculate the final descriptor for a given angular and radial order.
* 		
*        This function calls the calcFeature function number of times depending on the number of groups
*        for a given angular and radial order.
*/
void prepareMCSHFeatureAndSave(const double *image, MULTIPOLE_OBJ *mp, const double rCutoff, const int l, 
				const int radialFunctionType, const int radialFunctionOrder, const double *U,
				int DMVerts[6], MPI_Comm comm, double *featureVectorGlobal);

/**
* @brief  function to calculate score for a given angular order for parallelization of 
* 		  descriptor calculation.	    
*/
double scoreTask(const double rCutoff, const int l);

/**
* @brief  Calculate the stencil using a combination of angular and radial functions
*		 for descriptor calculation.
*
* 		This function calls the MaxwellCartesianSphericalHarmonics and LegendrePolynomial functions
* 		to calculate the stencil for convolution.
*/
void calculateStencil(const int stencilDimX, const int stencilDimY, const int stencilDimZ, const double hx, const double hy, const double hz, 
					  const double rCutoff, const int l, const char *n, const int radialFunctionType, const int radialFunctionOrder, 
					  const double *U, const int accuracy, double *stencil);


/**
* @brief  function to calculate the convolution of the stencil with the image.
* 		This function calls the calculateStencil function to calculate the stencil and then convolves with the image.	    
*/
void calcStencilAndConvolveAndAddResult(const double *image, MULTIPOLE_OBJ *mp, const double rCutoff, const int l, const char *n,
					const int radialFunctionType, const int radialFunctionOrder, const double *U, double *convolveResult, int DMVerts[6]);

