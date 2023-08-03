
/**
* @file MCSH.c
* @brief This file contains the declaration of MCSH functions.
*
* @author Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
*		Andrew J. Medford <ajm@gatech.edu>
*
* Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
*/
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
#include <assert.h>
#include <time.h> 
#include <mpi.h>
# include <math.h>
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
#include "MP_types.h"
#include "MCSHTools.h"

/**
* @brief  function to evaluate the MCSH polynomials
*/
void evaluateSeriesPolynomial(const double *x, const double *y, const double *z, const double *parameters, const int *polyTypes, const int numTerm, const int arraySize, double *result)
{	
	// parameters: [coeff_term1, x_pow_term1, y_pow_term1, z_pow_term1,
	// 			 coeff_term2, x_pow_term2, y_pow_term2, z_pow_term2,
	// 			 coeff_term3, x_pow_term3, y_pow_term3, z_pow_term3...]
	// polyType: [type_term1, type_term2, ...]
	// type: 0 - scalar, 1 - linear x, 2 - linear y, 3 - linear z, 4 - poly x, 5 - poly y, 6 poly - z, 7 - poly xyz
	int i;
	for (i = 0; i < arraySize; i++)
	{
		result[i] = 0;
	}
	for (i = 0; i < numTerm; i++)
	{
		if (polyTypes[i] == 0) // constant
		{
			addScalarVector(result, parameters[i * 4], result, arraySize);
		}
		else if (polyTypes[i] == 1) // linear x
		{
			multiplyScalarVector_add(x, parameters[i * 4], result, arraySize);
		}
		else if (polyTypes[i] == 2) // linear y
		{
			multiplyScalarVector_add(y, parameters[i * 4], result, arraySize);
		}
		else if (polyTypes[i] == 3) // linear z
		{
			multiplyScalarVector_add(z, parameters[i * 4], result, arraySize);
		}
		else if (polyTypes[i] == 4) // poly x
		{
			polyArray_add(x, parameters[i * 4 + 1], parameters[i * 4], result, arraySize);
		}
		else if (polyTypes[i] == 5) // poly y
		{
			polyArray_add(y, parameters[i * 4 + 2], parameters[i * 4], result, arraySize);
		}
		else if (polyTypes[i] == 6) // poly z
		{
			polyArray_add(z, parameters[i * 4 + 3], parameters[i * 4], result, arraySize);
		}
		else if (polyTypes[i] == 7) // poly xyz
		{
			polyXYZArray_add(x, y, z, parameters[i * 4 + 1], parameters[i * 4 + 2], parameters[i * 4 + 3], parameters[i * 4], result, arraySize);
		}
	}
}

/**
* @brief  function to define the parameters for different orders and groups of MCSH functions
*/
void MaxwellCartesianSphericalHarmonics(const double *x, const double *y, const double *z, const int l, const char *n, const double rCutoff, double *result, const int size)
{
	double *r = calloc( size, sizeof(double));
	double *x_hat = calloc( size, sizeof(double));
	double *y_hat = calloc( size, sizeof(double));
	double *z_hat = calloc( size, sizeof(double));

	getRArray(x, y, z, r, size);
	divideVector(x, r, x_hat, size);
	divideVector(y, r, y_hat, size);
	divideVector(z, r, z_hat, size);

	int rank = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &rank);

	if (rank == 0){
	printf("Rcut = %f \n", rCutoff);
	}

	int i;
	switch (l) 
	{
		case 0:
			if (strcmp(n, "000") == 0) {
				for ( i = 0; i < size; i++){
					result[i] = 1.0;
				}
			} 
			else{
				printf("\nWARNING: n is not valid %s \n", n);
			}
			break;

		case 1:
			if (strcmp(n, "100") == 0) {
				for ( i = 0; i < size; i++){
					result[i] = x[i];
				}
			} 
			else if (strcmp(n, "010") == 0){
				for ( i = 0; i < size; i++){
					result[i] = y[i];
				}
			}
			else if (strcmp(n, "001") == 0){
				for ( i = 0; i < size; i++){
					result[i] = z[i];
				}
			}
			else{
				printf("\nWARNING: n is not valid %s \n", n);
			}
			break;

		case 2:
			if (strcmp(n, "200") == 0) {
				// result = 3.0 * x * x - 1.0;
				int numTerm = 2;
				double parametersVal[] = { 3.0, 2, 0, 0, 
										  -1.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {4, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);

			} 
			else if (strcmp(n, "020") == 0){
				// result = 3.0 * y * y - 1.0;
				int numTerm = 2;
				double parametersVal[] = { 3.0, 0, 2, 0, 
										  -1.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "002") == 0){
				// result = 3.0 * z * z - 1.0;
				int numTerm = 2;
				double parametersVal[] = { 3.0, 0, 0, 2, 
										  -1.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "110") == 0){
				// result = 3.0 * x * y;
				polyXYZArray(x, y, z, 1, 1, 0, 3.0, result, size);
			}
			else if (strcmp(n, "101") == 0){
				// result = 3.0 * x * z;
				polyXYZArray(x, y, z, 1, 0, 1, 3.0, result, size);
			}
			else if (strcmp(n, "011") == 0){
				// result = 3.0 * y * z;
				polyXYZArray(x, y, z, 0, 1, 1, 3.0, result, size);
			}
			else{
				printf("\nWARNING: n is not valid %s \n", n);
			}
			break;

		case 3:
			if (strcmp(n, "300") == 0) {
				// result = 15.0 * x * x * x - 9.0 * x;
				int numTerm = 2;
				double parametersVal[] = { 15.0, 3, 0, 0, 
										   -9.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {4, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			} 
			else if (strcmp(n, "030") == 0){
				// result = 15.0 * y * y * y - 9.0 * y;
				int numTerm = 2;
				double parametersVal[] = { 15.0, 0, 3, 0, 
										   -9.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {5, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "003") == 0){
				// result = 15.0 * z * z * z - 9.0 * z;
				int numTerm = 2;
				double parametersVal[] = { 15.0, 0, 0, 3, 
										   -9.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {6, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "210") == 0){
				// result = 15.0 * x * x * y - 3.0 * y;
				int numTerm = 2;
				double parametersVal[] = { 15.0, 2, 1, 0, 
										   -3.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "120") == 0){
				// result = 15.0 * x * y * y - 3.0 * x;
				int numTerm = 2;
				double parametersVal[] = { 15.0, 1, 2, 0, 
										   -3.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);;
			}
			else if (strcmp(n, "201") == 0){
				// result = 15.0 * x * x * z - 3.0 * z;
				int numTerm = 2;
				double parametersVal[] = { 15.0, 2, 0, 1, 
										   -3.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "102") == 0){
				// result = 15.0 * x * z * z - 3.0 * x;
				int numTerm = 2;
				double parametersVal[] = { 15.0, 1, 0, 2, 
										   -3.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "021") == 0){
				// result = 15.0 * y * y * z - 3.0 * z;
				int numTerm = 2;
				double parametersVal[] = { 15.0, 0, 2, 1, 
							   -3.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);

			}
			else if (strcmp(n, "012") == 0){
				// result = 15.0 * y * z * z - 3.0 * y;
				int numTerm = 2;
				double parametersVal[] = { 15.0, 0, 1, 2, 
										   -3.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "111") == 0){
				// result = 15.0 * x * y * z;
				polyXYZArray(x, y, z, 1, 1, 1, 15.0, result, size);
			}
			else{
				printf("\nWARNING: n is not valid %s \n", n);
			}
			break;

		case 4:
			if (strcmp(n, "400") == 0) {
				// result = 105.0 * x * x * x * x - 90.0 * x * x + 9.0;
				int numTerm = 3;
				double parametersVal[] = { 105.0, 4, 0, 0, 
							   			   -90.0, 2, 0, 0,
							                9.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {4, 4, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			} 
			else if (strcmp(n, "040") == 0){
				// result = 105.0 * y * y * y * y - 90.0 * y * y + 9.0;
				int numTerm = 3;
				double parametersVal[] = { 105.0, 0, 4, 0, 
										   -90.0, 0, 2, 0,
											 9.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {5, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "004") == 0){
				// result = 105.0 * z * z * z * z - 90.0 * z * z + 9.0;
				int numTerm = 3;
				double parametersVal[] = { 105.0, 0, 0, 4, 
										   -90.0, 0, 0, 2,
											 9.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {6, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "310") == 0){
				// result = 105.0 * x * x * x * y - 45.0 * x * y;
				int numTerm = 2;
				double parametersVal[] = { 105.0, 3, 1, 0, 
							               -45.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "130") == 0){
				// result = 105.0 * x * y * y * y - 45.0 * x * y;
				int numTerm = 2;
				double parametersVal[] = { 105.0, 1, 3, 0, 
										   -45.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "301") == 0){
				// result = 105.0 * x * x * x * z - 45.0 * x * z;
				int numTerm = 2;
				double parametersVal[] = { 105.0, 3, 0, 1, 
										   -45.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "103") == 0){
				// result = 105.0 * x * z * z * z - 45.0 * x * z;
				int numTerm = 2;
				double parametersVal[] = { 105.0, 1, 0, 3, 
										   -45.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "031") == 0){
				// result = 105.0 * y * y * y * z - 45.0 * y * z;
				int numTerm = 2;
				double parametersVal[] = { 105.0, 0, 3, 1, 
										   -45.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "013") == 0){
				// result = 105.0 * y * z * z * z - 45.0 * y * z;
				int numTerm = 2;
				double parametersVal[] = { 105.0, 0, 1, 3, 
										   -45.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "220") == 0){
				// result = 105.0 * x * x * y * y - 15.0 * x * x - 15.0 * y * y + 3.0;
				int numTerm = 4;
				double parametersVal[] = { 105.0, 2, 2, 0, 
							   			   -15.0, 2, 0, 0,
							     		   -15.0, 0, 2, 0,
							                3.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "202") == 0){
				// result = 105.0 * x * x * z * z - 15.0 * x * x - 15.0 * z * z + 3.0;
				int numTerm = 4;
				double parametersVal[] = { 105.0, 2, 0, 2, 
							   			   -15.0, 2, 0, 0,
							               -15.0, 0, 0, 2,
							                3.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "022") == 0){
				// result = 105.0 * y * y * z * z - 15.0 * y * y - 15.0 * z * z + 3.0;
				int numTerm = 4;
				double parametersVal[] = { 105.0, 0, 2, 2, 
										   -15.0, 0, 2, 0,
										   -15.0, 0, 0, 2,
										     3.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "211") == 0){
				// result = 105.0 * x * x * y * z - 15.0 * y * z;
				int numTerm = 2;
				double parametersVal[] = { 105.0, 2, 1, 1, 
										   -15.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "121") == 0){
				// result = 105.0 * x * y * y * z - 15.0 * x * z;
				int numTerm = 2;
				double parametersVal[] = { 105.0, 1, 2, 1, 
										   -15.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "112") == 0){
				// result = 105.0 * x * y * z * z - 15.0 * x * y;
				int numTerm = 2;
				double parametersVal[] = { 105.0, 1, 1, 2, 
										   -15.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else{
				printf("\nWARNING: n is not valid %s \n", n);
			}
			break;

		case 5:
			if (strcmp(n, "500") == 0) {
				// 945.0 * x**5 -1050.0 * x**3 + 225.0 * x
				int numTerm = 3;
				double parametersVal[] = {   945.0, 5, 0, 0, 
										   -1050.0, 3, 0, 0,
										     225.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {4, 4, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			} 
			else if (strcmp(n, "050") == 0){
				// 945.0 * y**5 -1050.0 * y**3 + 225.0 * y
				int numTerm = 3;
				double parametersVal[] = {   945.0, 0, 5, 0, 
										   -1050.0, 0, 3, 0,
										     225.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {5, 5, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "005") == 0){
				// 945.0 * z**5 -1050.0 * z**3 + 225.0 * z
				int numTerm = 3;
				double parametersVal[] = {   945.0, 0, 0, 5, 
										   -1050.0, 0, 0, 3,
										     225.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {6, 6, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "410") == 0){
				// 945.0 * x**4 * y-630.0 * x**2 * y+ 45.0 * y
				int numTerm = 3;
				double parametersVal[] = {   945.0, 4, 1, 0, 
										    -630.0, 2, 1, 0,
										      45.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "140") == 0){
				// 945.0 * x* y**4 -630.0 * x* y**2 + 45.0 * x
				int numTerm = 3;
				double parametersVal[] = {   945.0, 1, 4, 0, 
										    -630.0, 1, 2, 0,
										      45.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "401") == 0){
				// 945.0 * x**4 * z-630.0 * x**2 * z+ 45.0 * z
				int numTerm = 3;
				double parametersVal[] = {   945.0, 4, 0, 1, 
										    -630.0, 2, 0, 1,
										      45.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "104") == 0){
				// 945.0 * x* z**4 -630.0 * x* z**2 + 45.0 * x
				int numTerm = 3;
				double parametersVal[] = {   945.0, 1, 0, 4, 
										    -630.0, 1, 0, 2,
										      45.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "041") == 0){
				// 945.0 * y**4 * z-630.0 * y**2 * z+ 45.0 * z
				int numTerm = 3;
				double parametersVal[] = {   945.0, 0, 4, 1, 
										    -630.0, 0, 2, 1,
										      45.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "014") == 0){
				// 945.0 * y* z**4 -630.0 * y* z**2 + 45.0 * y
				int numTerm = 3;
				double parametersVal[] = {   945.0, 0, 1, 4, 
										    -630.0, 0, 1, 2,
										      45.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "320") == 0){
				// 945.0 * x**3 * y**2 -105.0 * x**3 -315.0 * x* y**2 + 45.0 * x
				int numTerm = 4;
				double parametersVal[] = {   945.0, 3, 2, 0, 
										    -105.0, 3, 0, 0,
										    -315.0, 1, 2, 0,
										      45.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "230") == 0){
				// 945.0 * x**2 * y**3 -315.0 * x**2 * y-105.0 * y**3 + 45.0 * y
				int numTerm = 4;
				double parametersVal[] = {   945.0, 2, 3, 0, 
										    -105.0, 0, 3, 0,
										    -315.0, 2, 1, 0,
										      45.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "302") == 0){
				// 945.0 * x**3 * z**2 -105.0 * x**3 -315.0 * x* z**2 + 45.0 * x
				int numTerm = 4;
				double parametersVal[] = {   945.0, 3, 0, 2, 
										    -105.0, 3, 0, 0,
										    -315.0, 1, 0, 2,
										      45.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "203") == 0){
				// 945.0 * x**2 * z**3 -315.0 * x**2 * z-105.0 * z**3 + 45.0 * z
				int numTerm = 4;
				double parametersVal[] = {   945.0, 2, 0, 3, 
										    -105.0, 0, 0, 3,
										    -315.0, 2, 0, 1,
										      45.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "032") == 0){
				// 945.0 * y**3 * z**2 -105.0 * y**3 -315.0 * y* z**2 + 45.0 * y
				int numTerm = 4;
				double parametersVal[] = {   945.0, 0, 3, 2, 
										    -105.0, 0, 3, 0,
										    -315.0, 0, 1, 2,
										      45.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "023") == 0){
				// 945.0 * y**2 * z**3 -315.0 * y**2 * z-105.0 * z**3 + 45.0 * z
				int numTerm = 4;    
				double parametersVal[] = {   945.0, 0, 2, 3, 
										    -105.0, 0, 0, 3,
										    -315.0, 0, 2, 1,
										      45.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "311") == 0){
				// 945.0 * x**3 * y* z-315.0 * x* y* z
				int numTerm = 2;
				double parametersVal[] = {   945.0, 3, 1, 1,
										    -315.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "131") == 0){
				// 945.0 * x* y**3 * z-315.0 * x* y* z
				int numTerm = 2;
				double parametersVal[] = {   945.0, 1, 3, 1,
										    -315.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "113") == 0){
				// 945.0 * x* y* z**3 -315.0 * x* y* z
				int numTerm = 2;
				double parametersVal[] = {   945.0, 1, 1, 3,
										    -315.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "221") == 0){
				// 945.0 * x**2 * y**2 * z-105.0 * x**2 * z-105.0 * y**2 * z+ 15.0 * z
				int numTerm = 4;
				double parametersVal[] = {   945.0, 2, 2, 1,
										    -105.0, 2, 0, 1,
										    -105.0, 0, 2, 1,
										      15.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "212") == 0){
				// 945.0 * x**2 * y* z**2 -105.0 * x**2 * y-105.0 * y* z**2 + 15.0 * y
				int numTerm = 4;
				double parametersVal[] = {   945.0, 2, 1, 2,
										    -105.0, 2, 1, 0,
										    -105.0, 0, 1, 2,
										      15.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "122") == 0){
				// 945.0 * x* y**2 * z**2 -105.0 * x* y**2 -105.0 * x* z**2 + 15.0 * x
				int numTerm = 4;
				double parametersVal[] = {   945.0, 1, 2, 2,
										    -105.0, 1, 2, 0,
										    -105.0, 1, 0, 2,
										      15.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else{
				printf("\nWARNING: n is not valid %s \n", n);
			}

			break;


		case 6:
			if (strcmp(n, "600") == 0) {
				// 10395.0 * x**6 -14175.0 * x**4 + 4725.0 * x**2 -225.0
				int numTerm = 4;
				double parametersVal[] = {   10395.0, 6, 0, 0,
										    -14175.0, 4, 0, 0,
										      4725.0, 2, 0, 0,
										      -225.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {4, 4, 4, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			} 
			else if (strcmp(n, "060") == 0){
				// 10395.0 * y**6 -14175.0 * y**4 + 4725.0 * y**2 -225.0
				int numTerm = 4;
				double parametersVal[] = {   10395.0, 0, 6, 0,
										    -14175.0, 0, 4, 0,
										      4725.0, 0, 2, 0,
										      -225.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {5, 5, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "006") == 0){
				// 10395.0 * z**6 -14175.0 * z**4 + 4725.0 * z**2 -225.0 
				int numTerm = 4;
				double parametersVal[] = {   10395.0, 0, 0, 6,
										    -14175.0, 0, 0, 4,
										      4725.0, 0, 0, 2,
										      -225.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {6, 6, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "510") == 0){
				// 10395.0 * x**5 * y-9450.0 * x**3 * y+ 1575.0 * x* y
				int numTerm = 3;
				double parametersVal[] = {   10395.0, 5, 1, 0,
										     -9450.0, 3, 1, 0,
										      1575.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "150") == 0){
				// 10395.0 * x* y**5 -9450.0 * x* y**3 + 1575.0 * x* y
				int numTerm = 3;
				double parametersVal[] = {   10395.0, 1, 5, 0,
										     -9450.0, 1, 3, 0,
										      1575.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "501") == 0){
				// 10395.0 * x**5 * z-9450.0 * x**3 * z+ 1575.0 * x* z
				int numTerm = 3;
				double parametersVal[] = {   10395.0, 5, 0, 1,
										     -9450.0, 3, 0, 1,
										      1575.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "105") == 0){
				// 10395.0 * x* z**5 -9450.0 * x* z**3 + 1575.0 * x* z
				int numTerm = 3;
				double parametersVal[] = {   10395.0, 1, 0, 5,
										     -9450.0, 1, 0, 3,
										      1575.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "051") == 0){
				// 10395.0 * y**5 * z-9450.0 * y**3 * z+ 1575.0 * y* z
				int numTerm = 3;
				double parametersVal[] = {   10395.0, 0, 5, 1,
										     -9450.0, 0, 3, 1,
										      1575.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "015") == 0){
				// 10395.0 * y* z**5 -9450.0 * y* z**3 + 1575.0 * y* z
				int numTerm = 3;
				double parametersVal[] = {   10395.0, 0, 1, 5,
										     -9450.0, 0, 1, 3,
										      1575.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "420") == 0){
				// 10395.0 * x**4 * y**2 -945.0 * x**4 -5670.0 * x**2 * y**2 + 630.0 * x**2 + 315.0 * y**2 -45.0 
				int numTerm = 6;
				double parametersVal[] = {  10395.0, 4, 2, 0,
										     -945.0, 4, 0, 0,
										    -5670.0, 2, 2, 0,
										      630.0, 2, 0, 0,
										      315.0, 0, 2, 0,
										      -45.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 4, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "240") == 0){
				// 10395.0 * x**2 * y**4 -5670.0 * x**2 * y**2 + 315.0 * x**2 -945.0 * y**4 + 630.0 * y**2 -45.0 
				int numTerm = 6;
				double parametersVal[] = {  10395.0, 2, 4, 0,
										     -945.0, 0, 4, 0,
										    -5670.0, 2, 2, 0,
										      630.0, 0, 2, 0,
										      315.0, 2, 0, 0,
										      -45.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 5, 4, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "402") == 0){
				// 10395.0 * x**4 * z**2 -945.0 * x**4 -5670.0 * x**2 * z**2 + 630.0 * x**2 + 315.0 * z**2 -45.0 
				int numTerm = 6;
				double parametersVal[] = {  10395.0, 4, 0, 2,
										     -945.0, 4, 0, 0,
										    -5670.0, 2, 0, 2,
										      630.0, 2, 0, 0,
										      315.0, 0, 0, 2,
										      -45.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 4, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "204") == 0){
				// 10395.0 * x**2 * z**4 -5670.0 * x**2 * z**2 + 315.0 * x**2 -945.0 * z**4 + 630.0 * z**2 -45.0 
				int numTerm = 6;
				double parametersVal[] = {  10395.0, 2, 0, 4,
										     -945.0, 0, 0, 4,
										    -5670.0, 2, 0, 2,
										      630.0, 0, 0, 2,
										      315.0, 2, 0, 0,
										      -45.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 6, 4, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "042") == 0){
				// 10395.0 * y**4 * z**2 -945.0 * y**4 -5670.0 * y**2 * z**2 + 630.0 * y**2 + 315.0 * z**2 -45.0 
				int numTerm = 6;
				double parametersVal[] = {  10395.0, 0, 4, 2,
										     -945.0, 0, 4, 0,
										    -5670.0, 0, 2, 2,
										      630.0, 0, 2, 0,
										      315.0, 0, 0, 2,
										      -45.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 5, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "024") == 0){
				// 10395.0 * y**2 * z**4 -5670.0 * y**2 * z**2 + 315.0 * y**2 -945.0 * z**4 + 630.0 * z**2 -45.0 
				int numTerm = 6;
				double parametersVal[] = {  10395.0, 0, 2, 4,
										     -945.0, 0, 0, 4,
										    -5670.0, 0, 2, 2,
										      630.0, 0, 0, 2,
										      315.0, 0, 2, 0,
										      -45.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 6, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "411") == 0){
				// 10395.0 * x**4 * y* z-5670.0 * x**2 * y* z+ 315.0 * y* z
				int numTerm = 3;
				double parametersVal[] = {  10395.0, 4, 1, 1,
										    -5670.0, 2, 1, 1,
										      315.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "141") == 0){
				// 10395.0 * x* y**4 * z-5670.0 * x* y**2 * z+ 315.0 * x* z
				int numTerm = 3;
				double parametersVal[] = {  10395.0, 1, 4, 1,
										    -5670.0, 1, 2, 1,
										      315.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "114") == 0){
				// 10395.0 * x* y* z**4 -5670.0 * x* y* z**2 + 315.0 * x* y
				int numTerm = 3;
				double parametersVal[] = {  10395.0, 1, 1, 4,
										    -5670.0, 1, 1, 2,
										      315.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "330") == 0){
				// 10395.0 * x**3 * y**3 -2835.0 * x**3 * y-2835.0 * x* y**3 + 945.0 * x* y
				int numTerm = 4;
				double parametersVal[] = {  10395.0, 3, 3, 0,
										    -2835.0, 3, 1, 0,
										    -2835.0, 1, 3, 0,
										      945.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "303") == 0){
				// 10395.0 * x**3 * z**3 -2835.0 * x**3 * z-2835.0 * x* z**3 + 945.0 * x* z
				int numTerm = 4;
				double parametersVal[] = {  10395.0, 3, 0, 3,
										    -2835.0, 3, 0, 1,
										    -2835.0, 1, 0, 3,
										      945.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "033") == 0){
				// 10395.0 * y**3 * z**3 -2835.0 * y**3 * z-2835.0 * y* z**3 + 945.0 * y* z
				int numTerm = 4;
				double parametersVal[] = {  10395.0, 0, 3, 3,
										    -2835.0, 0, 3, 1,
										    -2835.0, 0, 1, 3,
										      945.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "321") == 0){
				// 10395.0 * x**3 * y**2 * z-945.0 * x**3 * z-2835.0 * x* y**2 * z+ 315.0 * x* z
				int numTerm = 4;
				double parametersVal[] = {  10395.0, 3, 2, 1,
										     -945.0, 3, 0, 1,
										    -2835.0, 1, 2, 1,
										      315.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "231") == 0){
				// 10395.0 * x**2 * y**3 * z-2835.0 * x**2 * y* z-945.0 * y**3 * z+ 315.0 * y* z
				int numTerm = 4;
				double parametersVal[] = {  10395.0, 2, 3, 1,
										     -945.0, 0, 3, 1,
										    -2835.0, 2, 1, 1,
										      315.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "312") == 0){
				// 10395.0 * x**3 * y* z**2 -945.0 * x**3 * y-2835.0 * x* y* z**2 + 315.0 * x* y
				int numTerm = 4;
				double parametersVal[] = {  10395.0, 3, 1, 2,
										     -945.0, 3, 1, 0,
										    -2835.0, 1, 1, 2,
										      315.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "213") == 0){
				// 10395.0 * x**2 * y* z**3 -2835.0 * x**2 * y* z-945.0 * y* z**3 + 315.0 * y* z
				int numTerm = 4;
				double parametersVal[] = {  10395.0, 2, 1, 3,
										     -945.0, 0, 1, 3,
										    -2835.0, 2, 1, 1,
										      315.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "132") == 0){
				// 10395.0 * x* y**3 * z**2 -945.0 * x* y**3 -2835.0 * x* y* z**2 + 315.0 * x* y
				int numTerm = 4;
				double parametersVal[] = {  10395.0, 1, 3, 2,
										     -945.0, 1, 3, 0,
										    -2835.0, 1, 1, 2,
										      315.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "123") == 0){
				// 10395.0 * x* y**2 * z**3 -2835.0 * x* y**2 * z-945.0 * x* z**3 + 315.0 * x* z
				int numTerm = 4;
				double parametersVal[] = {  10395.0, 1, 2, 3,
										     -945.0, 1, 0, 3,
										    -2835.0, 1, 2, 1,
										      315.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "222") == 0){
				// 10395.0 * x**2 * y**2 * z**2 -945.0 * x**2 * y**2 -945.0 * x**2 * z**2 + 105.0 * x**2 -945.0 * y**2 * z**2 + 105.0 * y**2 + 105.0 * z**2 -15.0 
				int numTerm = 8;
				double parametersVal[] = {  10395.0, 2, 2, 2,
										     -945.0, 2, 2, 0,
										     -945.0, 2, 0, 2,
										     -945.0, 0, 2, 2,
										      105.0, 2, 0, 0,
										      105.0, 0, 2, 0,
										      105.0, 0, 0, 2,
										      -15.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 4, 5, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else{
				printf("\nWARNING: n is not valid %s \n", n);
			}

			break;

		case 7:
			if (strcmp(n, "700") == 0){
				// 135135.0 * x**7 -218295.0 * x**5 + 99225.0 * x**3 -11025.0 * x 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 7, 0, 0,
										   -218295.0, 5, 0, 0,
										     99225.0, 3, 0, 0,
										    -11025.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {4, 4, 4, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "070") == 0){
				// 135135.0 * y**7 -218295.0 * y**5 + 99225.0 * y**3 -11025.0 * y 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 0, 7, 0,
										   -218295.0, 0, 5, 0,
										     99225.0, 0, 3, 0,
										    -11025.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {5, 5, 5, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "007") == 0){
				// 135135.0 * z**7 -218295.0 * z**5 + 99225.0 * z**3 -11025.0 * z
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 0, 0, 7,
										   -218295.0, 0, 0, 5,
										     99225.0, 0, 0, 3,
										    -11025.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {6, 6, 6, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}	
			else if (strcmp(n, "610") == 0){
				// 135135.0 * x**6 * y -155925.0 * x**4 * y + 42525.0 * x**2 * y -1575.0 * y 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 6, 1, 0,
										   -155925.0, 4, 1, 0,
										     42525.0, 2, 1, 0,
										     -1575.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "160") == 0){
				// 135135.0 * x * y**6 -155925.0 * x * y**4 + 42525.0 * x * y**2 -1575.0 * x 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 1, 6, 0,
										   -155925.0, 1, 4, 0,
										     42525.0, 1, 2, 0,
										     -1575.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "601") == 0){
				// 135135.0 * x**6 * z -155925.0 * x**4 * z + 42525.0 * x**2 * z -1575.0 * z 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 6, 0, 1,
										   -155925.0, 4, 0, 1,
										     42525.0, 2, 0, 1,
										     -1575.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "106") == 0){
				// 135135.0 * x * z**6 -155925.0 * x * z**4 + 42525.0 * x * z**2 -1575.0 * x 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 1, 0, 6,
										   -155925.0, 1, 0, 4,
										     42525.0, 1, 0, 2,
										     -1575.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);

			}
			else if (strcmp(n, "061") == 0){
				// 135135.0 * y**6 * z -155925.0 * y**4 * z + 42525.0 * y**2 * z -1575.0 * z 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 0, 6, 1,
										   -155925.0, 0, 4, 1,
										     42525.0, 0, 2, 1,
										     -1575.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "016") == 0){
				// 135135.0 * y * z**6 -155925.0 * y * z**4 + 42525.0 * y * z**2 -1575.0 * y 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 0, 1, 6,
										   -155925.0, 0, 1, 4,
										     42525.0, 0, 1, 2,
										     -1575.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);

			}
			else if (strcmp(n, "520") == 0){
				// 135135.0 * x**5 * y**2 -10395.0 * x**5 -103950.0 * x**3 * y**2 + 9450.0 * x**3 + 14175.0 * x * y**2 -1575.0 * x 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 5, 2, 0,
										    -10395.0, 5, 0, 0,
										   -103950.0, 3, 2, 0,
										      9450.0, 3, 0, 0,
										     14175.0, 1, 2, 0,
										 	 -1575.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 4, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "250") == 0){
				// 135135.0 * x**2 * y**5 -103950.0 * x**2 * y**3 + 14175.0 * x**2 * y -10395.0 * y**5 + 9450.0 * y**3 -1575.0 * y 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 2, 5, 0,
										    -10395.0, 0, 5, 0,
										   -103950.0, 2, 3, 0,
										      9450.0, 0, 3, 0,
										     14175.0, 2, 1, 0,
										 	 -1575.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 5, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "502") == 0){
				// 135135.0 * x**5 * z**2 -10395.0 * x**5 -103950.0 * x**3 * z**2 + 9450.0 * x**3 + 14175.0 * x * z**2 -1575.0 * x 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 5, 0, 2,
										    -10395.0, 5, 0, 0,
										   -103950.0, 3, 0, 2,
										      9450.0, 3, 0, 0,
										     14175.0, 1, 0, 2,
										 	 -1575.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 4, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "205") == 0){
				// 135135.0 * x**2 * z**5 -103950.0 * x**2 * z**3 + 14175.0 * x**2 * z -10395.0 * z**5 + 9450.0 * z**3 -1575.0 * z 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 2, 0, 5,
										    -10395.0, 0, 0, 5,
										   -103950.0, 2, 0, 3,
										      9450.0, 0, 0, 3,
										     14175.0, 2, 0, 1,
										 	 -1575.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 6, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "052") == 0){
				// 135135.0 * y**5 * z**2 -10395.0 * y**5 -103950.0 * y**3 * z**2 + 9450.0 * y**3 + 14175.0 * y * z**2 -1575.0 * y 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 0, 5, 2,
										    -10395.0, 0, 5, 0,
										   -103950.0, 0, 3, 2,
										      9450.0, 0, 3, 0,
										     14175.0, 0, 1, 2,
										 	 -1575.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 5, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "025") == 0){
				// 135135.0 * y**2 * z**5 -103950.0 * y**2 * z**3 + 14175.0 * y**2 * z -10395.0 * z**5 + 9450.0 * z**3 -1575.0 * z 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 0, 2, 5,
										    -10395.0, 0, 0, 5,
										   -103950.0, 0, 2, 3,
										      9450.0, 0, 0, 3,
										     14175.0, 0, 2, 1,
										 	 -1575.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 6, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "511") == 0){
				// 135135.0 * x**5 * y * z -103950.0 * x**3 * y * z + 14175.0 * x * y * z 
				int numTerm = 3;
				double parametersVal[] = {  135135.0, 5, 1, 1,
										   -103950.0, 3, 1, 1,
										     14175.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "151") == 0){
				// 135135.0 * x * y**5 * z -103950.0 * x * y**3 * z + 14175.0 * x * y * z 
				int numTerm = 3;
				double parametersVal[] = {  135135.0, 1, 5, 1,
										   -103950.0, 1, 3, 1,
										     14175.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "115") == 0){
				// 135135.0 * x * y * z**5 -103950.0 * x * y * z**3 + 14175.0 * x * y * z 
				int numTerm = 3;
				double parametersVal[] = {  135135.0, 1, 1, 5,
										   -103950.0, 1, 1, 3,
										     14175.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "430") == 0){
				// 135135.0 * x**4 * y**3 -31185.0 * x**4 * y -62370.0 * x**2 * y**3 + 17010.0 * x**2 * y + 2835.0 * y**3 -945.0 * y 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 4, 3, 0,
										    -31185.0, 4, 1, 0,
										    -62370.0, 2, 3, 0,
										     17010.0, 2, 1, 0,
										      2835.0, 0, 3, 0,
										 	  -945.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 5, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "340") == 0){
				// 135135.0 * x**3 * y**4 -62370.0 * x**3 * y**2 + 2835.0 * x**3 -31185.0 * x * y**4 + 17010.0 * x * y**2 -945.0 * x 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 3, 4, 0,
										    -31185.0, 1, 4, 0,
										    -62370.0, 3, 2, 0,
										     17010.0, 1, 2, 0,
										      2835.0, 3, 0, 0,
										 	  -945.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 4, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "403") == 0){
				// 135135.0 * x**4 * z**3 -31185.0 * x**4 * z -62370.0 * x**2 * z**3 + 17010.0 * x**2 * z + 2835.0 * z**3 -945.0 * z 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 4, 0, 3,
										    -31185.0, 4, 0, 1,
										    -62370.0, 2, 0, 3,
										     17010.0, 2, 0, 1,
										      2835.0, 0, 0, 3,
										 	  -945.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 6, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "304") == 0){
				// 135135.0 * x**3 * z**4 -62370.0 * x**3 * z**2 + 2835.0 * x**3 -31185.0 * x * z**4 + 17010.0 * x * z**2 -945.0 * x 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 3, 4, 0,
										    -31185.0, 1, 4, 0,
										    -62370.0, 3, 2, 0,
										     17010.0, 1, 2, 0,
										      2835.0, 3, 0, 0,
										 	  -945.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 4, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "043") == 0){
				// 135135.0 * y**4 * z**3 -31185.0 * y**4 * z -62370.0 * y**2 * z**3 + 17010.0 * y**2 * z + 2835.0 * z**3 -945.0 * z 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 0, 4, 3,
										    -31185.0, 0, 4, 1,
										    -62370.0, 0, 2, 3,
										     17010.0, 0, 2, 1,
										      2835.0, 0, 0, 3,
										 	  -945.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 6, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "034") == 0){
				// 135135.0 * y**3 * z**4 -62370.0 * y**3 * z**2 + 2835.0 * y**3 -31185.0 * y * z**4 + 17010.0 * y * z**2 -945.0 * y 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 0, 3, 4,
										    -31185.0, 0, 1, 4,
										    -62370.0, 0, 3, 2,
										     17010.0, 0, 1, 2,
										      2835.0, 0, 3, 0,
										 	  -945.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 5, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "421") == 0){
				// 135135.0 * x**4 * y**2 * z -10395.0 * x**4 * z -62370.0 * x**2 * y**2 * z + 5670.0 * x**2 * z + 2835.0 * y**2 * z -315.0 * z 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 4, 2, 1,
										    -10395.0, 4, 0, 1,
										    -62370.0, 2, 2, 1,
										      5670.0, 2, 0, 1,
										      2835.0, 0, 2, 1,
										 	  -315.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "241") == 0){
				// 135135.0 * x**2 * y**4 * z -62370.0 * x**2 * y**2 * z + 2835.0 * x**2 * z -10395.0 * y**4 * z + 5670.0 * y**2 * z -315.0 * z 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 2, 4, 1,
										    -10395.0, 0, 4, 1,
										    -62370.0, 2, 2, 1,
										      5670.0, 0, 2, 1,
										      2835.0, 2, 0, 1,
										 	  -315.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "412") == 0){
				// 135135.0 * x**4 * y * z**2 -10395.0 * x**4 * y -62370.0 * x**2 * y * z**2 + 5670.0 * x**2 * y + 2835.0 * y * z**2 -315.0 * y 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 4, 1, 2,
										    -10395.0, 4, 1, 0,
										    -62370.0, 2, 1, 2,
										      5670.0, 2, 1, 0,
										      2835.0, 0, 1, 2,
										 	  -315.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "214") == 0){
				// 135135.0 * x**2 * y * z**4 -62370.0 * x**2 * y * z**2 + 2835.0 * x**2 * y -10395.0 * y * z**4 + 5670.0 * y * z**2 -315.0 * y 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 2, 1, 4,
										    -10395.0, 0, 1, 4,
										    -62370.0, 2, 1, 2,
										      5670.0, 0, 1, 2,
										      2835.0, 2, 1, 0,
										 	  -315.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "142") == 0){
				// 135135.0 * x * y**4 * z**2 -10395.0 * x * y**4 -62370.0 * x * y**2 * z**2 + 5670.0 * x * y**2 + 2835.0 * x * z**2 -315.0 * x 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 1, 4, 2,
										    -10395.0, 1, 4, 0,
										    -62370.0, 1, 2, 2,
										      5670.0, 1, 2, 0,
										      2835.0, 1, 0, 2,
										 	  -315.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "124") == 0){
				// 135135.0 * x * y**2 * z**4 -62370.0 * x * y**2 * z**2 + 2835.0 * x * y**2 -10395.0 * x * z**4 + 5670.0 * x * z**2 -315.0 * x 
				int numTerm = 6;
				double parametersVal[] = {  135135.0, 1, 2, 4,
										    -10395.0, 1, 0, 4,
										    -62370.0, 1, 2, 2,
										      5670.0, 1, 0, 2,
										      2835.0, 1, 2, 0,
										 	  -315.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "331") == 0){
				// 135135.0 * x**3 * y**3 * z -31185.0 * x**3 * y * z -31185.0 * x * y**3 * z + 8505.0 * x * y * z 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 3, 3, 1,
										    -31185.0, 3, 1, 1,
										    -31185.0, 1, 3, 1,
										      8505.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "313") == 0){
				// 135135.0 * x**3 * y * z**3 -31185.0 * x**3 * y * z -31185.0 * x * y * z**3 + 8505.0 * x * y * z 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 3, 1, 3,
										    -31185.0, 3, 1, 1,
										    -31185.0, 1, 1, 3,
										      8505.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "133") == 0){
				// 135135.0 * x * y**3 * z**3 -31185.0 * x * y**3 * z -31185.0 * x * y * z**3 + 8505.0 * x * y * z 
				int numTerm = 4;
				double parametersVal[] = {  135135.0, 1, 3, 3,
										    -31185.0, 1, 3, 1,
										    -31185.0, 1, 1, 3,
										      8505.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "322") == 0){
				// 135135.0 * x**3 * y**2 * z**2 -10395.0 * x**3 * y**2 -10395.0 * x**3 * z**2 + 945.0 * x**3 -31185.0 * x * y**2 * z**2 + 2835.0 * x * y**2 + 2835.0 * x * z**2 -315.0 * x 
				int numTerm = 8;
				double parametersVal[] = {  135135.0, 3, 2, 2,
										    -10395.0, 3, 0, 2,
										    -10395.0, 3, 2, 0,
										       945.0, 3, 0, 0,
										    -31185.0, 1, 2, 2,
										 	  2835.0, 1, 0, 2,
										      2835.0, 1, 2, 0,
										 	  -315.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 4, 7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "232") == 0){
				// 135135.0 * x**2 * y**3 * z**2 -10395.0 * x**2 * y**3 -31185.0 * x**2 * y * z**2 + 2835.0 * x**2 * y -10395.0 * y**3 * z**2 + 945.0 * y**3 + 2835.0 * y * z**2 -315.0 * y 
				int numTerm = 8;
				double parametersVal[] = {  135135.0, 2, 3, 2,
										    -10395.0, 0, 3, 2,
										    -10395.0, 2, 3, 0,
										       945.0, 0, 3, 0,
										    -31185.0, 2, 1, 2,
										 	  2835.0, 0, 1, 2,
										      2835.0, 2, 1, 0,
										 	  -315.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 5, 7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "223") == 0){
				// 135135.0 * x**2 * y**2 * z**3 -31185.0 * x**2 * y**2 * z -10395.0 * x**2 * z**3 + 2835.0 * x**2 * z -10395.0 * y**2 * z**3 + 2835.0 * y**2 * z + 945.0 * z**3 -315.0 * z 
				int numTerm = 8;
				double parametersVal[] = {  135135.0, 2, 2, 3,
										    -10395.0, 2, 0, 3,
										    -10395.0, 0, 2, 3,
										       945.0, 0, 0, 3,
										    -31185.0, 2, 2, 1,
										 	  2835.0, 2, 0, 1,
										      2835.0, 0, 2, 1,
										 	  -315.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 6, 7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else{
				printf("\nWARNING: n is not valid %s \n", n);
			}
			break;

		case 8:
			if (strcmp(n, "800") == 0){
				// 2027025.0 * x**8 -3783780.0 * x**6 + 2182950.0 * x**4 -396900.0 * x**2 + 11025.0
				int numTerm = 5;
				double parametersVal[] = {  2027025.0, 8, 0, 0,
										   -3783780.0, 6, 0, 0,
										    2182950.0, 4, 0, 0,
										    -396900.0, 2, 0, 0,
										      11025.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {4, 4, 4, 4, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "080") == 0){
				// 2027025.0 * y**8 -3783780.0 * y**6 + 2182950.0 * y**4 -396900.0 * y**2 + 11025.0
				int numTerm = 5;
				double parametersVal[] = {  2027025.0, 0, 8, 0,
										   -3783780.0, 0, 6, 0,
										    2182950.0, 0, 4, 0,
										    -396900.0, 0, 2, 0,
										      11025.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {5, 5, 5, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "008") == 0){
				// 2027025.0 * z**8 -3783780.0 * z**6 + 2182950.0 * z**4 -396900.0 * z**2 + 11025.0
				int numTerm = 5;
				double parametersVal[] = {  2027025.0, 0, 0, 8,
										   -3783780.0, 0, 0, 6,
										    2182950.0, 0, 0, 4,
										    -396900.0, 0, 0, 2,
										      11025.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {6, 6, 6, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "710") == 0){
				// 2027025.0 * x**7 * y -2837835.0 * x**5 * y + 1091475.0 * x**3 * y -99225.0 * x * y
				int numTerm = 4;
				double parametersVal[] = {  2027025.0, 7, 1, 0,
										   -2837835.0, 5, 1, 0,
										    1091475.0, 3, 1, 0,
										     -99225.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "170") == 0){
				// 2027025.0 * x * y**7 -2837835.0 * x * y**5 + 1091475.0 * x * y**3 -99225.0 * x * y
				int numTerm = 4;
				double parametersVal[] = {  2027025.0, 1, 7, 0,
										   -2837835.0, 1, 5, 0,
										    1091475.0, 1, 3, 0,
										     -99225.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "701") == 0){
				// 2027025.0 * x**7 * z -2837835.0 * x**5 * z + 1091475.0 * x**3 * z -99225.0 * x * z 
				int numTerm = 4;
				double parametersVal[] = {  2027025.0, 7, 0, 1,
										   -2837835.0, 5, 0, 1,
										    1091475.0, 3, 0, 1,
										     -99225.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "107") == 0){
				// 2027025.0 * x * z**7 -2837835.0 * x * z**5 + 1091475.0 * x * z**3 -99225.0 * x * z
				int numTerm = 4;
				double parametersVal[] = {  2027025.0, 1, 0, 7,
										   -2837835.0, 1, 0, 5,
										    1091475.0, 1, 0, 3,
										     -99225.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "071") == 0){
				// 2027025.0 * y**7 * z -2837835.0 * y**5 * z + 1091475.0 * y**3 * z -99225.0 * y * z
				int numTerm = 4;
				double parametersVal[] = {  2027025.0, 0, 7, 1,
										   -2837835.0, 0, 5, 1,
										    1091475.0, 0, 3, 1,
										     -99225.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "017") == 0){
				// 2027025.0 * y * z**7 -2837835.0 * y * z**5 + 1091475.0 * y * z**3 -99225.0 * y * z
				int numTerm = 4;
				double parametersVal[] = {  2027025.0, 0, 1, 7,
										   -2837835.0, 0, 1, 5,
										    1091475.0, 0, 1, 3,
										     -99225.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "620") == 0){
				// 2027025.0 * x**6 * y**2 -135135.0 * x**6 -2027025.0 * x**4 * y**2 + 155925.0 * x**4 + 467775.0 * x**2 * y**2 -42525.0 * x**2 -14175.0 * y**2 + 1575.0
				int numTerm = 8;
				double parametersVal[] = {  2027025.0, 6, 2, 0,
										    -135135.0, 6, 0, 0,
										   -2027025.0, 4, 2, 0,
										     155925.0, 4, 0, 0,
										     467775.0, 2, 2, 0,
										     -42525.0, 2, 0, 0,
										     -14175.0, 0, 2, 0,
										       1575.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 4, 7, 4, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "260") == 0){
				// 2027025.0 * x**2 * y**6 -2027025.0 * x**2 * y**4 + 467775.0 * x**2 * y**2 -14175.0 * x**2 -135135.0 * y**6 + 155925.0 * y**4 -42525.0 * y**2 + 1575.0 
				int numTerm = 8;
				double parametersVal[] = {  2027025.0, 2, 6, 0,
										    -135135.0, 0, 6, 0,
										   -2027025.0, 2, 4, 0,
										     155925.0, 0, 4, 0,
										     467775.0, 2, 2, 0,
										     -42525.0, 0, 2, 0,
										     -14175.0, 2, 0, 0,
										       1575.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 5, 7, 5, 4, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "602") == 0){
				// 2027025.0 * x**6 * z**2 -135135.0 * x**6 -2027025.0 * x**4 * z**2 + 155925.0 * x**4 + 467775.0 * x**2 * z**2 -42525.0 * x**2 -14175.0 * z**2 + 1575.0
				int numTerm = 8;
				double parametersVal[] = {  2027025.0, 6, 0, 2,
										    -135135.0, 6, 0, 0,
										   -2027025.0, 4, 0, 2,
										     155925.0, 4, 0, 0,
										     467775.0, 2, 0, 2,
										     -42525.0, 2, 0, 0,
										     -14175.0, 0, 0, 2,
										       1575.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 4, 7, 4, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "206") == 0){
				// 2027025.0 * x**2 * z**6 -2027025.0 * x**2 * z**4 + 467775.0 * x**2 * z**2 -14175.0 * x**2 -135135.0 * z**6 + 155925.0 * z**4 -42525.0 * z**2 + 1575.0 
				int numTerm = 8;
				double parametersVal[] = {  2027025.0, 2, 0, 6,
										    -135135.0, 0, 0, 6,
										   -2027025.0, 2, 0, 4,
										     155925.0, 0, 0, 4,
										     467775.0, 2, 0, 2,
										     -42525.0, 0, 0, 2,
										     -14175.0, 2, 0, 0,
										       1575.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 6, 7, 6, 4, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "062") == 0){
				// 2027025.0 * y**6 * z**2 -135135.0 * y**6 -2027025.0 * y**4 * z**2 + 155925.0 * y**4 + 467775.0 * y**2 * z**2 -42525.0 * y**2 -14175.0 * z**2 + 1575.0
				int numTerm = 8;
				double parametersVal[] = {  2027025.0, 0, 6, 2,
										    -135135.0, 0, 6, 0,
										   -2027025.0, 0, 4, 2,
										     155925.0, 0, 4, 0,
										     467775.0, 0, 2, 2,
										     -42525.0, 0, 2, 0,
										     -14175.0, 0, 0, 2,
										       1575.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 5, 7, 5, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "026") == 0){
				// 2027025.0 * y**2 * z**6 -2027025.0 * y**2 * z**4 + 467775.0 * y**2 * z**2 -14175.0 * y**2 -135135.0 * z**6 + 155925.0 * z**4 -42525.0 * z**2 + 1575.0
				int numTerm = 8;
				double parametersVal[] = {  2027025.0, 0, 2, 6,
										    -135135.0, 0, 0, 6,
										   -2027025.0, 0, 2, 4,
										     155925.0, 0, 0, 4,
										     467775.0, 0, 2, 2,
										     -42525.0, 0, 0, 2,
										     -14175.0, 0, 2, 0,
										       1575.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 6, 7, 6, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "611") == 0){
				// 2027025.0 * x**6 * y * z -2027025.0 * x**4 * y * z + 467775.0 * x**2 * y * z -14175.0 * y * z 
				int numTerm = 4;
				double parametersVal[] = {  2027025.0, 6, 1, 1,
										   -2027025.0, 4, 1, 1,
										     467775.0, 2, 1, 1,
										     -14175.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "161") == 0){
				// 2027025.0 * x * y**6 * z -2027025.0 * x * y**4 * z + 467775.0 * x * y**2 * z -14175.0 * x * z
				int numTerm = 4;
				double parametersVal[] = {  2027025.0, 1, 6, 1,
										   -2027025.0, 1, 4, 1,
										     467775.0, 1, 2, 1,
										     -14175.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "116") == 0){
				// 2027025.0 * x * y * z**6 -2027025.0 * x * y * z**4 + 467775.0 * x * y * z**2 -14175.0 * x * y
				int numTerm = 4;
				double parametersVal[] = {  2027025.0, 1, 1, 6,
										   -2027025.0, 1, 1, 4,
										     467775.0, 1, 1, 2,
										     -14175.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "530") == 0){
				// 2027025.0 * x**5 * y**3 -405405.0 * x**5 * y -1351350.0 * x**3 * y**3 + 311850.0 * x**3 * y + 155925.0 * x * y**3 -42525.0 * x * y
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 5, 3, 0,
										    -405405.0, 5, 1, 0,
										   -1351350.0, 3, 3, 0,
										     311850.0, 3, 1, 0,
										     155925.0, 1, 3, 0,
										     -42525.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "350") == 0){
				// 2027025.0 * x**3 * y**5 -1351350.0 * x**3 * y**3 + 155925.0 * x**3 * y -405405.0 * x * y**5 + 311850.0 * x * y**3 -42525.0 * x * y 
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 3, 5, 0,
										    -405405.0, 1, 5, 0,
										   -1351350.0, 3, 3, 0,
										     311850.0, 1, 3, 0,
										     155925.0, 3, 1, 0,
										     -42525.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "503") == 0){
				// 2027025.0 * x**5 * z**3 -405405.0 * x**5 * z -1351350.0 * x**3 * z**3 + 311850.0 * x**3 * z + 155925.0 * x * z**3 -42525.0 * x * z 
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 5, 0, 3,
										    -405405.0, 5, 0, 1,
										   -1351350.0, 3, 0, 3,
										     311850.0, 3, 0, 1,
										     155925.0, 1, 0, 3,
										     -42525.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "305") == 0){
				// 2027025.0 * x**3 * z**5 -1351350.0 * x**3 * z**3 + 155925.0 * x**3 * z -405405.0 * x * z**5 + 311850.0 * x * z**3 -42525.0 * x * z 
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 3, 0, 5,
										    -405405.0, 1, 0, 5,
										   -1351350.0, 3, 0, 3,
										     311850.0, 1, 0, 3,
										     155925.0, 3, 0, 1,
										     -42525.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "053") == 0){
				// 2027025.0 * y**5 * z**3 -405405.0 * y**5 * z -1351350.0 * y**3 * z**3 + 311850.0 * y**3 * z + 155925.0 * y * z**3 -42525.0 * y * z
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 0, 5, 3,
										    -405405.0, 0, 5, 1,
										   -1351350.0, 0, 3, 3,
										     311850.0, 0, 3, 1,
										     155925.0, 0, 1, 3,
										     -42525.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "035") == 0){
				// 2027025.0 * y**3 * z**5 -1351350.0 * y**3 * z**3 + 155925.0 * y**3 * z -405405.0 * y * z**5 + 311850.0 * y * z**3 -42525.0 * y * z 
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 0, 3, 5,
										    -405405.0, 0, 1, 5,
										   -1351350.0, 0, 3, 3,
										     311850.0, 0, 1, 3,
										     155925.0, 0, 3, 1,
										     -42525.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "521") == 0){
				// 2027025.0 * x**5 * y**2 * z -135135.0 * x**5 * z -1351350.0 * x**3 * y**2 * z + 103950.0 * x**3 * z + 155925.0 * x * y**2 * z -14175.0 * x * z
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 5, 2, 1,
										    -135135.0, 5, 0, 1,
										   -1351350.0, 3, 2, 1,
										     103950.0, 3, 0, 1,
										     155925.0, 1, 2, 1,
										     -14175.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "251") == 0){
				// 2027025.0 * x**2 * y**5 * z -1351350.0 * x**2 * y**3 * z + 155925.0 * x**2 * y * z -135135.0 * y**5 * z + 103950.0 * y**3 * z -14175.0 * y * z 
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 2, 5, 1,
										    -135135.0, 0, 5, 1,
										   -1351350.0, 2, 3, 1,
										     103950.0, 0, 3, 1,
										     155925.0, 2, 1, 1,
										     -14175.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "512") == 0){
				// 2027025.0 * x**5 * y * z**2 -135135.0 * x**5 * y -1351350.0 * x**3 * y * z**2 + 103950.0 * x**3 * y + 155925.0 * x * y * z**2 -14175.0 * x * y 
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 5, 1, 2,
										    -135135.0, 5, 1, 0,
										   -1351350.0, 3, 1, 2,
										     103950.0, 3, 1, 0,
										     155925.0, 1, 1, 2,
										     -14175.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "215") == 0){
				// 2027025.0 * x**2 * y * z**5 -1351350.0 * x**2 * y * z**3 + 155925.0 * x**2 * y * z -135135.0 * y * z**5 + 103950.0 * y * z**3 -14175.0 * y * z
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 2, 1, 5,
										    -135135.0, 0, 1, 5,
										   -1351350.0, 2, 1, 3,
										     103950.0, 0, 1, 3,
										     155925.0, 2, 1, 1,
										     -14175.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "152") == 0){
				// 2027025.0 * x * y**5 * z**2 -135135.0 * x * y**5 -1351350.0 * x * y**3 * z**2 + 103950.0 * x * y**3 + 155925.0 * x * y * z**2 -14175.0 * x * y
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 1, 5, 2,
										    -135135.0, 1, 5, 0,
										   -1351350.0, 1, 3, 2,
										     103950.0, 1, 3, 0,
										     155925.0, 1, 1, 2,
										     -14175.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "125") == 0){
				// 2027025.0 * x * y**2 * z**5 -1351350.0 * x * y**2 * z**3 + 155925.0 * x * y**2 * z -135135.0 * x * z**5 + 103950.0 * x * z**3 -14175.0 * x * z
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 1, 2, 5,
										    -135135.0, 1, 0, 5,
										   -1351350.0, 1, 2, 3,
										     103950.0, 1, 0, 3,
										     155925.0, 1, 2, 1,
										     -14175.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "440") == 0){
				// 2027025.0 * x**4 * y**4 -810810.0 * x**4 * y**2 + 31185.0 * x**4 -810810.0 * x**2 * y**4 + 374220.0 * x**2 * y**2 -17010.0 * x**2 + 31185.0 * y**4 -17010.0 * y**2 + 945.0
				int numTerm = 9;
				double parametersVal[] = {  2027025.0, 4, 4, 0,
										    -810810.0, 4, 2, 0,
										      31185.0, 4, 0, 0,
										    -810810.0, 2, 4, 0,
										     374220.0, 2, 2, 0,
										     -17010.0, 2, 0, 0,
										      31185.0, 0, 4, 0,
										     -17010.0, 0, 2, 0,
										        945.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 4, 7, 7, 4, 5, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "404") == 0){
				// 2027025.0 * x**4 * z**4 -810810.0 * x**4 * z**2 + 31185.0 * x**4 -810810.0 * x**2 * z**4 + 374220.0 * x**2 * z**2 -17010.0 * x**2 + 31185.0 * z**4 -17010.0 * z**2 + 945.0 
				int numTerm = 9;
				double parametersVal[] = {  2027025.0, 4, 0, 4,
										    -810810.0, 4, 0, 2,
										      31185.0, 4, 0, 0,
										    -810810.0, 2, 0, 4,
										     374220.0, 2, 0, 2,
										     -17010.0, 2, 0, 0,
										      31185.0, 0, 0, 4,
										     -17010.0, 0, 0, 2,
										        945.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 4, 7, 7, 4, 6, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "044") == 0){
				// 2027025.0 * y**4 * z**4 -810810.0 * y**4 * z**2 + 31185.0 * y**4 -810810.0 * y**2 * z**4 + 374220.0 * y**2 * z**2 -17010.0 * y**2 + 31185.0 * z**4 -17010.0 * z**2 + 945.0 
				int numTerm = 9;
				double parametersVal[] = {  2027025.0, 0, 4, 4,
										    -810810.0, 0, 4, 2,
										      31185.0, 0, 4, 0,
										    -810810.0, 0, 2, 4,
										     374220.0, 0, 2, 2,
										     -17010.0, 0, 2, 0,
										      31185.0, 0, 0, 4,
										     -17010.0, 0, 0, 2,
										        945.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 5, 7, 7, 5, 6, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "431") == 0){
				// 2027025.0 * x**4 * y**3 * z -405405.0 * x**4 * y * z -810810.0 * x**2 * y**3 * z + 187110.0 * x**2 * y * z + 31185.0 * y**3 * z -8505.0 * y * z 
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 4, 3, 1,
										    -405405.0, 4, 1, 1,
										    -810810.0, 2, 3, 1,
										     187110.0, 2, 1, 1,
										      31185.0, 0, 3, 1,
										      -8505.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "341") == 0){
				// 2027025.0 * x**3 * y**4 * z -810810.0 * x**3 * y**2 * z + 31185.0 * x**3 * z -405405.0 * x * y**4 * z + 187110.0 * x * y**2 * z -8505.0 * x * z
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 3, 4, 1,
										    -405405.0, 1, 4, 1,
										    -810810.0, 3, 2, 1,
										     187110.0, 1, 2, 1,
										      31185.0, 3, 0, 1,
										      -8505.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "413") == 0){
				// 2027025.0 * x**4 * y * z**3 -405405.0 * x**4 * y * z -810810.0 * x**2 * y * z**3 + 187110.0 * x**2 * y * z + 31185.0 * y * z**3 -8505.0 * y * z
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 4, 1, 3,
										    -405405.0, 4, 1, 1,
										    -810810.0, 2, 1, 3,
										     187110.0, 2, 1, 1,
										      31185.0, 0, 1, 3,
										      -8505.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "314") == 0){
				// 2027025.0 * x**3 * y * z**4 -810810.0 * x**3 * y * z**2 + 31185.0 * x**3 * y -405405.0 * x * y * z**4 + 187110.0 * x * y * z**2 -8505.0 * x * y 
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 3, 1, 4,
										    -405405.0, 1, 1, 4,
										    -810810.0, 3, 1, 2,
										     187110.0, 1, 1, 2,
										      31185.0, 3, 1, 0,
										      -8505.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "143") == 0){
				// 2027025.0 * x * y**4 * z**3 -405405.0 * x * y**4 * z -810810.0 * x * y**2 * z**3 + 187110.0 * x * y**2 * z + 31185.0 * x * z**3 -8505.0 * x * z
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 1, 4, 3,
										    -405405.0, 1, 4, 1,
										    -810810.0, 1, 2, 3,
										     187110.0, 1, 2, 1,
										      31185.0, 1, 0, 3,
										      -8505.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "134") == 0){
				// 2027025.0 * x * y**3 * z**4 -810810.0 * x * y**3 * z**2 + 31185.0 * x * y**3 -405405.0 * x * y * z**4 + 187110.0 * x * y * z**2 -8505.0 * x * y
				int numTerm = 6;
				double parametersVal[] = {  2027025.0, 1, 3, 4,
										    -405405.0, 1, 1, 4,
										    -810810.0, 1, 3, 2,
										     187110.0, 1, 1, 2,
										      31185.0, 1, 3, 0,
										      -8505.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "422") == 0){
				// 2027025.0 * x**4 * y**2 * z**2 -135135.0 * x**4 * y**2 -135135.0 * x**4 * z**2 + 10395.0 * x**4 -810810.0 * x**2 * y**2 * z**2 + 
				// 62370.0 * x**2 * y**2 + 62370.0 * x**2 * z**2 -5670.0 * x**2 + 31185.0 * y**2 * z**2 -2835.0 * y**2 -2835.0 * z**2 + 315.0 
				int numTerm = 12;
				double parametersVal[] = {  2027025.0, 4, 2, 2,
										    -135135.0, 4, 2, 0,
										    -135135.0, 4, 0, 2,
										      10395.0, 4, 0, 0,
										    -810810.0, 2, 2, 2,
										      62370.0, 2, 2, 0,
										      62370.0, 2, 0, 2,
										      -5670.0, 2, 0, 0,
										      31185.0, 0, 2, 2,
										      -2835.0, 0, 2, 0,
										      -2835.0, 0, 0, 2,
										        315.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 4, 7, 7, 7, 4, 7, 5, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "242") == 0){
				// 2027025.0 * x**2 * y**4 * z**2 -135135.0 * x**2 * y**4 -810810.0 * x**2 * y**2 * z**2 + 62370.0 * x**2 * y**2 + 31185.0 * x**2 * z**2 -2835.0 * x**2 -135135.0 * y**4 * z**2 + 10395.0 * y**4 + 62370.0 * y**2 * z**2 -5670.0 * y**2 -2835.0 * z**2 + 315.0
				int numTerm = 12;
				double parametersVal[] = {  2027025.0, 2, 4, 2,
										    -135135.0, 2, 4, 0,
										    -135135.0, 0, 4, 2,
										      10395.0, 0, 4, 0,
										    -810810.0, 2, 2, 2,
										      62370.0, 2, 2, 0,
										      62370.0, 0, 2, 2,
										      -5670.0, 0, 2, 0,
										      31185.0, 2, 0, 2,
										      -2835.0, 2, 0, 0,
										      -2835.0, 0, 0, 2,
										        315.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 5, 7, 7, 7, 5, 7, 4, 6, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "224") == 0){
				// 2027025.0 * x**2 * y**2 * z**4 -810810.0 * x**2 * y**2 * z**2 + 31185.0 * x**2 * y**2 -135135.0 * x**2 * z**4 + 62370.0 * x**2 * z**2 -2835.0 * x**2 -135135.0 * y**2 * z**4 + 62370.0 * y**2 * z**2 -2835.0 * y**2 + 10395.0 * z**4 -5670.0 * z**2 + 315.0
				int numTerm = 12;
				double parametersVal[] = {  2027025.0, 2, 2, 4,
										    -135135.0, 2, 0, 4,
										    -135135.0, 0, 2, 4,
										      10395.0, 0, 0, 4,
										    -810810.0, 2, 2, 2,
										      62370.0, 2, 0, 2,
										      62370.0, 0, 2, 2,
										      -5670.0, 0, 0, 2,
										      31185.0, 2, 2, 0,
										      -2835.0, 2, 0, 0,
										      -2835.0, 0, 2, 0,
										        315.0, 0, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 6, 7, 7, 7, 6, 7, 4, 5, 0};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "332") == 0){
				// 2027025.0 * x**3 * y**3 * z**2 -135135.0 * x**3 * y**3 -405405.0 * x**3 * y * z**2 + 31185.0 * x**3 * y -405405.0 * x * y**3 * z**2 + 
				//31185.0 * x * y**3 + 93555.0 * x * y * z**2 -8505.0 * x * y
				int numTerm = 8;
				double parametersVal[] = {  2027025.0, 3, 3, 2,
										    -135135.0, 3, 3, 0,
										    -405405.0, 3, 1, 2,
										      31185.0, 3, 1, 0,
										    -405405.0, 1, 3, 2,
										      31185.0, 1, 3, 0,
										      93555.0, 1, 1, 2,
										      -8505.0, 1, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "323") == 0){
				// 2027025.0 * x**3 * y**2 * z**3 -405405.0 * x**3 * y**2 * z -135135.0 * x**3 * z**3 + 31185.0 * x**3 * z -405405.0 * x * y**2 * z**3 + 93555.0 * x * y**2 * z + 31185.0 * x * z**3 -8505.0 * x * z
				int numTerm = 8;
				double parametersVal[] = {  2027025.0, 3, 2, 3,
										    -135135.0, 3, 0, 3,
										    -405405.0, 3, 2, 1,
										      31185.0, 3, 0, 1,
										    -405405.0, 1, 2, 3,
										      31185.0, 1, 0, 3,
										      93555.0, 1, 2, 1,
										      -8505.0, 1, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "233") == 0){
				// 2027025.0 * x**2 * y**3 * z**3 -405405.0 * x**2 * y**3 * z -405405.0 * x**2 * y * z**3 + 93555.0 * x**2 * y * z -135135.0 * y**3 * z**3 + 31185.0 * y**3 * z + 31185.0 * y * z**3 -8505.0 * y * z
				int numTerm = 8;
				double parametersVal[] = {  2027025.0, 2, 3, 3,
										    -135135.0, 0, 3, 3,
										    -405405.0, 2, 3, 1,
										      31185.0, 0, 3, 1,
										    -405405.0, 2, 1, 3,
										      31185.0, 0, 1, 3,
										      93555.0, 2, 1, 1,
										      -8505.0, 0, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else
			{
				printf("\nWARNING: n is not valid %s \n", n);
			}

			break;

		case 9:
			if (strcmp(n, "900") == 0){
				// 34459425.0 * x**9 -72972900.0 * x**7 + 51081030.0 * x**5 -13097700.0 * x**3 + 893025.0 * x 
				int numTerm = 5;
				double parametersVal[] = {  34459425.0, 9, 0, 0,
										   -72972900.0, 7, 0, 0,
										    51081030.0, 5, 0, 0,
										   -13097700.0, 3, 0, 0,
										      893025.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {4, 4, 4, 4, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "090") == 0){
				// 34459425.0 * y**9 -72972900.0 * y**7 + 51081030.0 * y**5 -13097700.0 * y**3 + 893025.0 * y 
				int numTerm = 5;
				double parametersVal[] = {  34459425.0, 0, 9, 0,
										   -72972900.0, 0, 7, 0,
										    51081030.0, 0, 5, 0,
										   -13097700.0, 0, 3, 0,
										      893025.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {5, 5, 5, 5, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);

			}
			else if (strcmp(n, "009") == 0){
				// 34459425.0 * z**9 -72972900.0 * z**7 + 51081030.0 * z**5 -13097700.0 * z**3 + 893025.0 * z 
				int numTerm = 5;
				double parametersVal[] = {  34459425.0, 0, 0, 9,
										   -72972900.0, 0, 0, 7,
										    51081030.0, 0, 0, 5,
										   -13097700.0, 0, 0, 3,
										      893025.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {6, 6, 6, 6, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);

			}
			else if (strcmp(n, "810") == 0){
				// 34459425.0 * x**8 * y -56756700.0 * x**6 * y + 28378350.0 * x**4 * y -4365900.0 * x**2 * y + 99225.0 * y
				int numTerm = 5;
				double parametersVal[] = {  34459425.0, 8, 1, 0,
										   -56756700.0, 6, 1, 0,
										    28378350.0, 4, 1, 0,
										    -4365900.0, 2, 1, 0,
										       99225.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "180") == 0){
				// 34459425.0 * x * y**8 -56756700.0 * x * y**6 + 28378350.0 * x * y**4 -4365900.0 * x * y**2 + 99225.0 * x 
				int numTerm = 5;
				double parametersVal[] = {  34459425.0, 1, 8, 0,
										   -56756700.0, 1, 6, 0,
										    28378350.0, 1, 4, 0,
										    -4365900.0, 1, 2, 0,
										       99225.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "801") == 0){
				// 34459425.0 * x**8 * z -56756700.0 * x**6 * z + 28378350.0 * x**4 * z -4365900.0 * x**2 * z + 99225.0 * z 
				int numTerm = 5;
				double parametersVal[] = {  34459425.0, 8, 0, 1,
										   -56756700.0, 6, 0, 1,
										    28378350.0, 4, 0, 1,
										    -4365900.0, 2, 0, 1,
										       99225.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "108") == 0){
				// 34459425.0 * x * z**8 -56756700.0 * x * z**6 + 28378350.0 * x * z**4 -4365900.0 * x * z**2 + 99225.0 * x 
				int numTerm = 5;
				double parametersVal[] = {  34459425.0, 1, 0, 8,
										   -56756700.0, 1, 0, 6,
										    28378350.0, 1, 0, 4,
										    -4365900.0, 1, 0, 2,
										       99225.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "081") == 0){
				// 34459425.0 * y**8 * z -56756700.0 * y**6 * z + 28378350.0 * y**4 * z -4365900.0 * y**2 * z + 99225.0 * z 
				int numTerm = 5;
				double parametersVal[] = {  34459425.0, 0, 8, 1,
										   -56756700.0, 0, 6, 1,
										    28378350.0, 0, 4, 1,
										    -4365900.0, 0, 2, 1,
										       99225.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "018") == 0){
				// 34459425.0 * y * z**8 -56756700.0 * y * z**6 + 28378350.0 * y * z**4 -4365900.0 * y * z**2 + 99225.0 * y 
				int numTerm = 5;
				double parametersVal[] = {  34459425.0, 0, 1, 8,
										   -56756700.0, 0, 1, 6,
										    28378350.0, 0, 1, 4,
										    -4365900.0, 0, 1, 2,
										       99225.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "720") == 0){
				// 34459425.0 * x**7 * y**2 -2027025.0 * x**7 -42567525.0 * x**5 * y**2 + 2837835.0 * x**5 + 14189175.0 * x**3 * y**2 -1091475.0 * x**3 -1091475.0 * x * y**2 + 99225.0 * x 
				int numTerm = 8;
				double parametersVal[] = {   34459425.0, 7, 2, 0,
										     -2027025.0, 7, 0, 0,
										    -42567525.0, 5, 2, 0,
										      2837835.0, 5, 0, 0,
										     14189175.0, 3, 2, 0,
										     -1091475.0, 3, 0, 0,
										     -1091475.0, 1, 2, 0,
										        99225.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 4, 7, 4, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "270") == 0){
				// 34459425.0 * x**2 * y**7 -42567525.0 * x**2 * y**5 + 14189175.0 * x**2 * y**3 -1091475.0 * x**2 * y -2027025.0 * y**7 + 2837835.0 * y**5 -1091475.0 * y**3 + 99225.0 * y 
				int numTerm = 8;
				double parametersVal[] = {   34459425.0, 2, 7, 0,
										     -2027025.0, 0, 7, 0,
										    -42567525.0, 2, 5, 0,
										      2837835.0, 0, 5, 0,
										     14189175.0, 2, 3, 0,
										     -1091475.0, 0, 3, 0,
										     -1091475.0, 2, 1, 0,
										        99225.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 5, 7, 5, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "702") == 0){
				// 34459425.0 * x**7 * z**2 -2027025.0 * x**7 -42567525.0 * x**5 * z**2 + 2837835.0 * x**5 + 14189175.0 * x**3 * z**2 -1091475.0 * x**3 -1091475.0 * x * z**2 + 99225.0 * x
				int numTerm = 8;
				double parametersVal[] = {   34459425.0, 7, 0, 2,
										     -2027025.0, 7, 0, 0,
										    -42567525.0, 5, 0, 2,
										      2837835.0, 5, 0, 0,
										     14189175.0, 3, 0, 2,
										     -1091475.0, 3, 0, 0,
										     -1091475.0, 1, 0, 2,
										        99225.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 4, 7, 4, 7, 4, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "207") == 0){
				// 34459425.0 * x**2 * z**7 -42567525.0 * x**2 * z**5 + 14189175.0 * x**2 * z**3 -1091475.0 * x**2 * z -2027025.0 * z**7 + 2837835.0 * z**5 -1091475.0 * z**3 + 99225.0 * z
				int numTerm = 8;
				double parametersVal[] = {   34459425.0, 2, 0, 7,
										     -2027025.0, 0, 0, 7,
										    -42567525.0, 2, 0, 5,
										      2837835.0, 0, 0, 5,
										     14189175.0, 2, 0, 3,
										     -1091475.0, 0, 0, 3,
										     -1091475.0, 2, 0, 1,
										        99225.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 6, 7, 6, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "072") == 0){
				// 34459425.0 * y**7 * z**2 -2027025.0 * y**7 -42567525.0 * y**5 * z**2 + 2837835.0 * y**5 + 14189175.0 * y**3 * z**2 -1091475.0 * y**3 -1091475.0 * y * z**2 + 99225.0 * y
				int numTerm = 8;
				double parametersVal[] = {   34459425.0, 0, 7, 2,
										     -2027025.0, 0, 7, 0,
										    -42567525.0, 0, 5, 2,
										      2837835.0, 0, 5, 0,
										     14189175.0, 0, 3, 2,
										     -1091475.0, 0, 3, 0,
										     -1091475.0, 0, 1, 2,
										        99225.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 5, 7, 5, 7, 5, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "027") == 0){
				// 34459425.0 * y**2 * z**7 -42567525.0 * y**2 * z**5 + 14189175.0 * y**2 * z**3 -1091475.0 * y**2 * z -2027025.0 * z**7 + 2837835.0 * z**5 -1091475.0 * z**3 + 99225.0 * z
				int numTerm = 8;
				double parametersVal[] = {   34459425.0, 0, 2, 7,
										     -2027025.0, 0, 0, 7,
										    -42567525.0, 0, 2, 5,
										      2837835.0, 0, 0, 5,
										     14189175.0, 0, 2, 3,
										     -1091475.0, 0, 0, 3,
										     -1091475.0, 0, 2, 1,
										        99225.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 6, 7, 6, 7, 6, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "711") == 0){
				// 34459425.0 * x**7 * y * z -42567525.0 * x**5 * y * z + 14189175.0 * x**3 * y * z -1091475.0 * x * y * z 
				int numTerm = 4;
				double parametersVal[] = {  34459425.0, 7, 1, 1,
										   -42567525.0, 5, 1, 1,
										    14189175.0, 3, 1, 1,
										    -1091475.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "171") == 0){
				// 34459425.0 * x * y**7 * z -42567525.0 * x * y**5 * z + 14189175.0 * x * y**3 * z -1091475.0 * x * y * z
				int numTerm = 4;
				double parametersVal[] = {  34459425.0, 1, 7, 1,
										   -42567525.0, 1, 5, 1,
										    14189175.0, 1, 3, 1,
										    -1091475.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "117") == 0){
				// 34459425.0 * x * y * z**7 -42567525.0 * x * y * z**5 + 14189175.0 * x * y * z**3 -1091475.0 * x * y * z
				int numTerm = 4;
				double parametersVal[] = {  34459425.0, 1, 1, 7,
										   -42567525.0, 1, 1, 5,
										    14189175.0, 1, 1, 3,
										    -1091475.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "630") == 0){
				// 34459425.0 * x**6 * y**3 -6081075.0 * x**6 * y -30405375.0 * x**4 * y**3 + 6081075.0 * x**4 * y + 6081075.0 * x**2 * y**3 -1403325.0 * x**2 * y -155925.0 * y**3 + 42525.0 * y 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 6, 3, 0,
										    -6081075.0, 6, 1, 0,
										   -30405375.0, 4, 3, 0,
										     6081075.0, 4, 1, 0,
										     6081075.0, 2, 3, 0,
										    -1403325.0, 2, 1, 0,
										     -155925.0, 0, 3, 0,
										       42525.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 5, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "360") == 0){
				// 34459425.0 * x**3 * y**6 -30405375.0 * x**3 * y**4 + 6081075.0 * x**3 * y**2 -155925.0 * x**3 -6081075.0 * x * y**6 + 6081075.0 * x * y**4 -1403325.0 * x * y**2 + 42525.0 * x
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 3, 6, 0,
										    -6081075.0, 1, 6, 0,
										   -30405375.0, 3, 4, 0,
										     6081075.0, 1, 4, 0,
										     6081075.0, 3, 2, 0,
										    -1403325.0, 1, 2, 0,
										     -155925.0, 3, 0, 0,
										       42525.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 4, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "603") == 0){
				// 34459425.0 * x**6 * z**3 -6081075.0 * x**6 * z -30405375.0 * x**4 * z**3 + 6081075.0 * x**4 * z + 6081075.0 * x**2 * z**3 -1403325.0 * x**2 * z -155925.0 * z**3 + 42525.0 * z 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 6, 0, 3,
										    -6081075.0, 6, 0, 1,
										   -30405375.0, 4, 0, 3,
										     6081075.0, 4, 0, 1,
										     6081075.0, 2, 0, 3,
										    -1403325.0, 2, 0, 1,
										     -155925.0, 0, 0, 3,
										       42525.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 6, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "306") == 0){
				// 34459425.0 * x**3 * z**6 -30405375.0 * x**3 * z**4 + 6081075.0 * x**3 * z**2 -155925.0 * x**3 -6081075.0 * x * z**6 + 6081075.0 * x * z**4 -1403325.0 * x * z**2 + 42525.0 * x 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 3, 0, 6,
										    -6081075.0, 1, 0, 6,
										   -30405375.0, 3, 0, 4,
										     6081075.0, 1, 0, 4,
										     6081075.0, 3, 0, 2,
										    -1403325.0, 1, 0, 2,
										     -155925.0, 3, 0, 0,
										       42525.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 4, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "063") == 0){
				// 34459425.0 * y**6 * z**3 -6081075.0 * y**6 * z -30405375.0 * y**4 * z**3 + 6081075.0 * y**4 * z + 6081075.0 * y**2 * z**3 -1403325.0 * y**2 * z -155925.0 * z**3 + 42525.0 * z 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 0, 6, 3,
										    -6081075.0, 0, 6, 1,
										   -30405375.0, 0, 4, 3,
										     6081075.0, 0, 4, 1,
										     6081075.0, 0, 2, 3,
										    -1403325.0, 0, 2, 1,
										     -155925.0, 0, 0, 3,
										       42525.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 6, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "036") == 0){
				// 34459425.0 * y**3 * z**6 -30405375.0 * y**3 * z**4 + 6081075.0 * y**3 * z**2 -155925.0 * y**3 -6081075.0 * y * z**6 + 6081075.0 * y * z**4 -1403325.0 * y * z**2 + 42525.0 * y 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 0, 3, 6,
										    -6081075.0, 0, 1, 6,
										   -30405375.0, 0, 3, 4,
										     6081075.0, 0, 1, 4,
										     6081075.0, 0, 3, 2,
										    -1403325.0, 0, 1, 2,
										     -155925.0, 0, 3, 0,
										       42525.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 5, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "621") == 0){
				// 34459425.0 * x**6 * y**2 * z -2027025.0 * x**6 * z -30405375.0 * x**4 * y**2 * z + 2027025.0 * x**4 * z + 6081075.0 * x**2 * y**2 * z -467775.0 * x**2 * z -155925.0 * y**2 * z + 14175.0 * z 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 6, 2, 1,
										    -2027025.0, 6, 0, 1,
										   -30405375.0, 4, 2, 1,
										     2027025.0, 4, 0, 1,
										     6081075.0, 2, 2, 1,
										     -467775.0, 2, 0, 1,
										     -155925.0, 0, 2, 1,
										       14175.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "261") == 0){
				// 34459425.0 * x**2 * y**6 * z -30405375.0 * x**2 * y**4 * z + 6081075.0 * x**2 * y**2 * z -155925.0 * x**2 * z -2027025.0 * y**6 * z + 2027025.0 * y**4 * z -467775.0 * y**2 * z + 14175.0 * z 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 2, 6, 1,
										    -2027025.0, 0, 6, 1,
										   -30405375.0, 2, 4, 1,
										     2027025.0, 0, 4, 1,
										     6081075.0, 2, 2, 1,
										     -467775.0, 0, 2, 1,
										     -155925.0, 2, 0, 1,
										       14175.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "612") == 0){
				// 34459425.0 * x**6 * y * z**2 -2027025.0 * x**6 * y -30405375.0 * x**4 * y * z**2 + 2027025.0 * x**4 * y + 6081075.0 * x**2 * y * z**2 -467775.0 * x**2 * y -155925.0 * y * z**2 + 14175.0 * y 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 6, 1, 2,
										    -2027025.0, 6, 1, 0,
										   -30405375.0, 4, 1, 2,
										     2027025.0, 4, 1, 0,
										     6081075.0, 2, 1, 2,
										     -467775.0, 2, 1, 0,
										     -155925.0, 0, 1, 2,
										       14175.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "216") == 0){
				// 34459425.0 * x**2 * y * z**6 -30405375.0 * x**2 * y * z**4 + 6081075.0 * x**2 * y * z**2 -155925.0 * x**2 * y -2027025.0 * y * z**6 + 2027025.0 * y * z**4 -467775.0 * y * z**2 + 14175.0 * y 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 2, 1, 6,
										    -2027025.0, 0, 1, 6,
										   -30405375.0, 2, 1, 4,
										     2027025.0, 0, 1, 4,
										     6081075.0, 2, 1, 2,
										     -467775.0, 0, 1, 2,
										     -155925.0, 2, 1, 0,
										       14175.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "162") == 0){
				// 34459425.0 * x * y**6 * z**2 -2027025.0 * x * y**6 -30405375.0 * x * y**4 * z**2 + 2027025.0 * x * y**4 + 6081075.0 * x * y**2 * z**2 -467775.0 * x * y**2 -155925.0 * x * z**2 + 14175.0 * x
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 1, 6, 2,
										    -2027025.0, 1, 6, 0,
										   -30405375.0, 1, 4, 2,
										     2027025.0, 1, 4, 0,
										     6081075.0, 1, 2, 2,
										     -467775.0, 1, 2, 0,
										     -155925.0, 1, 0, 2,
										       14175.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "126") == 0){
				// 34459425.0 * x * y**2 * z**6 -30405375.0 * x * y**2 * z**4 + 6081075.0 * x * y**2 * z**2 -155925.0 * x * y**2 -2027025.0 * x * z**6 + 2027025.0 * x * z**4 -467775.0 * x * z**2 + 14175.0 * x 
				int numTerm = 8;
				double parametersVal[] = {  34459425.0, 1, 2, 6,
										    -2027025.0, 1, 0, 6,
										   -30405375.0, 1, 2, 4,
										     2027025.0, 1, 0, 4,
										     6081075.0, 1, 2, 2,
										     -467775.0, 1, 0, 2,
										     -155925.0, 1, 2, 0,
										       14175.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "540") == 0){
				// 34459425.0 * x**5 * y**4 -12162150.0 * x**5 * y**2 + 405405.0 * x**5 -20270250.0 * x**3 * y**4 + 8108100.0 * x**3 * y**2 -311850.0 * x**3 + 2027025.0 * x * y**4 -935550.0 * x * y**2 + 42525.0 * x 
				int numTerm = 9;
				double parametersVal[] = {   34459425.0, 5, 4, 0,
										    -12162150.0, 5, 2, 0,
										       405405.0, 5, 0, 0,
										    -20270250.0, 3, 4, 0,
										      8108100.0, 3, 2, 0,
										      -311850.0, 3, 0, 0,
										      2027025.0, 1, 4, 0,
										      -935550.0, 1, 2, 0,
										        42525.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 4, 7, 7, 4, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "450") == 0){
				// 34459425.0 * x**4 * y**5 -20270250.0 * x**4 * y**3 + 2027025.0 * x**4 * y -12162150.0 * x**2 * y**5 + 8108100.0 * x**2 * y**3 -935550.0 * x**2 * y + 405405.0 * y**5 -311850.0 * y**3 + 42525.0 * y
				int numTerm = 9;
				double parametersVal[] = {   34459425.0, 4, 5, 0,
										    -12162150.0, 2, 5, 0,
										       405405.0, 0, 5, 0,
										    -20270250.0, 4, 3, 0,
										      8108100.0, 2, 3, 0,
										      -311850.0, 0, 3, 0,
										      2027025.0, 4, 1, 0,
										      -935550.0, 2, 1, 0,
										        42525.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 5, 7, 7, 5, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "504") == 0){
				// 34459425.0 * x**5 * z**4 -12162150.0 * x**5 * z**2 + 405405.0 * x**5 -20270250.0 * x**3 * z**4 + 8108100.0 * x**3 * z**2 -311850.0 * x**3 + 2027025.0 * x * z**4 -935550.0 * x * z**2 + 42525.0 * x 
				int numTerm = 9;
				double parametersVal[] = {   34459425.0, 5, 0, 4,
										    -12162150.0, 5, 0, 2,
										       405405.0, 5, 0, 0,
										    -20270250.0, 3, 0, 4,
										      8108100.0, 3, 0, 2,
										      -311850.0, 3, 0, 0,
										      2027025.0, 1, 0, 4,
										      -935550.0, 1, 0, 2,
										        42525.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 4, 7, 7, 4, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "405") == 0){
				// 34459425.0 * x**4 * z**5 -20270250.0 * x**4 * z**3 + 2027025.0 * x**4 * z -12162150.0 * x**2 * z**5 + 8108100.0 * x**2 * z**3 -935550.0 * x**2 * z + 405405.0 * z**5 -311850.0 * z**3 + 42525.0 * z 
				int numTerm = 9;
				double parametersVal[] = {   34459425.0, 4, 0, 5,
										    -12162150.0, 2, 0, 5,
										       405405.0, 0, 0, 5,
										    -20270250.0, 4, 0, 3,
										      8108100.0, 2, 0, 3,
										      -311850.0, 0, 0, 3,
										      2027025.0, 4, 0, 1,
										      -935550.0, 2, 0, 1,
										        42525.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 6, 7, 7, 6, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "054") == 0){
				// 34459425.0 * y**5 * z**4 -12162150.0 * y**5 * z**2 + 405405.0 * y**5 -20270250.0 * y**3 * z**4 + 8108100.0 * y**3 * z**2 -311850.0 * y**3 + 2027025.0 * y * z**4 -935550.0 * y * z**2 + 42525.0 * y 
				int numTerm = 9;
				double parametersVal[] = {   34459425.0, 0, 5, 4,
										    -12162150.0, 0, 5, 2,
										       405405.0, 0, 5, 0,
										    -20270250.0, 0, 3, 4,
										      8108100.0, 0, 3, 2,
										      -311850.0, 0, 3, 0,
										      2027025.0, 0, 1, 4,
										      -935550.0, 0, 1, 2,
										        42525.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 5, 7, 7, 5, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "045") == 0){
				// 34459425.0 * y**4 * z**5 -20270250.0 * y**4 * z**3 + 2027025.0 * y**4 * z -12162150.0 * y**2 * z**5 + 8108100.0 * y**2 * z**3 -935550.0 * y**2 * z + 405405.0 * z**5 -311850.0 * z**3 + 42525.0 * z 
				int numTerm = 9;
				double parametersVal[] = {   34459425.0, 0, 4, 5,
										    -12162150.0, 0, 2, 5,
										       405405.0, 0, 0, 5,
										    -20270250.0, 0, 4, 3,
										      8108100.0, 0, 2, 3,
										      -311850.0, 0, 0, 3,
										      2027025.0, 0, 4, 1,
										      -935550.0, 0, 2, 1,
										        42525.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 6, 7, 7, 6, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "531") == 0){
				// 34459425.0 * x**5 * y**3 * z -6081075.0 * x**5 * y * z -20270250.0 * x**3 * y**3 * z + 4054050.0 * x**3 * y * z + 2027025.0 * x * y**3 * z -467775.0 * x * y * z 
				int numTerm = 6;
				double parametersVal[] = {  34459425.0, 5, 3, 1,
										    -6081075.0, 5, 1, 1,
										   -20270250.0, 3, 3, 1,
										     4054050.0, 3, 1, 1,
										     2027025.0, 1, 3, 1,
										     -467775.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "351") == 0){
				// 34459425.0 * x**3 * y**5 * z -20270250.0 * x**3 * y**3 * z + 2027025.0 * x**3 * y * z -6081075.0 * x * y**5 * z + 4054050.0 * x * y**3 * z -467775.0 * x * y * z
				int numTerm = 6;
				double parametersVal[] = {  34459425.0, 3, 5, 1,
										    -6081075.0, 1, 5, 1,
										   -20270250.0, 3, 3, 1,
										     4054050.0, 1, 3, 1,
										     2027025.0, 3, 1, 1,
										     -467775.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "513") == 0){
				// 34459425.0 * x**5 * y * z**3 -6081075.0 * x**5 * y * z -20270250.0 * x**3 * y * z**3 + 4054050.0 * x**3 * y * z + 2027025.0 * x * y * z**3 -467775.0 * x * y * z
				int numTerm = 6;
				double parametersVal[] = {  34459425.0, 5, 1, 3,
										    -6081075.0, 5, 1, 1,
										   -20270250.0, 3, 1, 3,
										     4054050.0, 3, 1, 1,
										     2027025.0, 1, 1, 3,
										     -467775.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "315") == 0){
				// 34459425.0 * x**3 * y * z**5 -20270250.0 * x**3 * y * z**3 + 2027025.0 * x**3 * y * z -6081075.0 * x * y * z**5 + 4054050.0 * x * y * z**3 -467775.0 * x * y * z
				int numTerm = 6;
				double parametersVal[] = {  34459425.0, 3, 1, 5,
										    -6081075.0, 1, 1, 5,
										   -20270250.0, 3, 1, 3,
										     4054050.0, 1, 1, 3,
										     2027025.0, 3, 1, 1,
										     -467775.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "153") == 0){
				// 34459425.0 * x * y**5 * z**3 -6081075.0 * x * y**5 * z -20270250.0 * x * y**3 * z**3 + 4054050.0 * x * y**3 * z + 2027025.0 * x * y * z**3 -467775.0 * x * y * z
				int numTerm = 6;
				double parametersVal[] = {  34459425.0, 1, 5, 3,
										    -6081075.0, 1, 5, 1,
										   -20270250.0, 1, 3, 3,
										     4054050.0, 1, 3, 1,
										     2027025.0, 1, 1, 3,
										     -467775.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "135") == 0){
				// 34459425.0 * x * y**3 * z**5 -20270250.0 * x * y**3 * z**3 + 2027025.0 * x * y**3 * z -6081075.0 * x * y * z**5 + 4054050.0 * x * y * z**3 -467775.0 * x * y * z 
				int numTerm = 6;
				double parametersVal[] = {  34459425.0, 1, 3, 5,
										    -6081075.0, 1, 1, 5,
										   -20270250.0, 1, 3, 3,
										     4054050.0, 1, 1, 3,
										     2027025.0, 1, 3, 1,
										     -467775.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "522") == 0){
				// 34459425.0 * x**5 * y**2 * z**2 -2027025.0 * x**5 * y**2 -2027025.0 * x**5 * z**2 + 135135.0 * x**5 -20270250.0 * x**3 * y**2 * z**2 + 1351350.0 * x**3 * y**2 + 1351350.0 * x**3 * z**2 -103950.0 * x**3 + 2027025.0 * x * y**2 * z**2 -155925.0 * x * y**2 -155925.0 * x * z**2 + 14175.0 * x 
				int numTerm = 12;
				double parametersVal[] = {   34459425.0, 5, 2, 2,
										     -2027025.0, 5, 2, 0,
										     -2027025.0, 5, 0, 2,
										       135135.0, 5, 0, 0,
										    -20270250.0, 3, 2, 2,
										      1351350.0, 3, 2, 0,
										      1351350.0, 3, 0, 2,
										      -103950.0, 3, 0, 0,
										      2027025.0, 1, 2, 2,
										      -155925.0, 1, 2, 0,
										      -155925.0, 1, 0, 2,
										        14175.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 4, 7, 7, 7, 4, 7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "252") == 0){
				// 34459425.0 * x**2 * y**5 * z**2 -2027025.0 * x**2 * y**5 -20270250.0 * x**2 * y**3 * z**2 + 1351350.0 * x**2 * y**3 + 2027025.0 * x**2 * y * z**2 -155925.0 * x**2 * y -2027025.0 * y**5 * z**2 + 135135.0 * y**5 + 1351350.0 * y**3 * z**2 -103950.0 * y**3 -155925.0 * y * z**2 + 14175.0 * y 
				int numTerm = 12;
				double parametersVal[] = {   34459425.0, 2, 5, 2,
										     -2027025.0, 2, 5, 0,
										     -2027025.0, 0, 5, 2,
										       135135.0, 0, 5, 0,
										    -20270250.0, 2, 3, 2,
										      1351350.0, 2, 3, 0,
										      1351350.0, 0, 3, 2,
										      -103950.0, 0, 3, 0,
										      2027025.0, 2, 1, 2,
										      -155925.0, 2, 1, 0,
										      -155925.0, 0, 1, 2,
										        14175.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 5, 7, 7, 7, 5, 7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "225") == 0){
				// 34459425.0 * x**2 * y**2 * z**5 -20270250.0 * x**2 * y**2 * z**3 + 2027025.0 * x**2 * y**2 * z -2027025.0 * x**2 * z**5 + 1351350.0 * x**2 * z**3 -155925.0 * x**2 * z -2027025.0 * y**2 * z**5 + 1351350.0 * y**2 * z**3 -155925.0 * y**2 * z + 135135.0 * z**5 -103950.0 * z**3 + 14175.0 * z
				int numTerm = 12;
				double parametersVal[] = {   34459425.0, 2, 2, 5,
										     -2027025.0, 2, 0, 5,
										     -2027025.0, 0, 2, 5,
										       135135.0, 0, 0, 5,
										    -20270250.0, 2, 2, 3,
										      1351350.0, 2, 0, 3,
										      1351350.0, 0, 2, 3,
										      -103950.0, 0, 0, 3,
										      2027025.0, 2, 2, 1,
										      -155925.0, 2, 0, 1,
										      -155925.0, 0, 2, 1,
										        14175.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "441") == 0){
				// 34459425.0 * x**4 * y**4 * z -12162150.0 * x**4 * y**2 * z + 405405.0 * x**4 * z -12162150.0 * x**2 * y**4 * z + 
				//4864860.0 * x**2 * y**2 * z -187110.0 * x**2 * z + 405405.0 * y**4 * z -187110.0 * y**2 * z + 8505.0 * z 
				int numTerm = 9;
				double parametersVal[] = {  34459425.0, 4, 4, 1,
										   -12162150.0, 4, 2, 1,
										      405405.0, 4, 0, 1,
										   -12162150.0, 2, 4, 1,
										     4864860.0, 2, 2, 1,
										     -187110.0, 2, 0, 1,
										      405405.0, 0, 4, 1,
										     -187110.0, 0, 2, 1,
										        8505.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "414") == 0){
				// 34459425.0 * x**4 * y * z**4 -12162150.0 * x**4 * y * z**2 + 405405.0 * x**4 * y -12162150.0 * x**2 * y * z**4 + 4864860.0 * x**2 * y * z**2 -187110.0 * x**2 * y + 405405.0 * y * z**4 -187110.0 * y * z**2 + 8505.0 * y 
				int numTerm = 9;
				double parametersVal[] = {  34459425.0, 4, 1, 4,
										   -12162150.0, 4, 1, 2,
										      405405.0, 4, 1, 0,
										   -12162150.0, 2, 1, 4,
										     4864860.0, 2, 1, 2,
										     -187110.0, 2, 1, 0,
										      405405.0, 0, 1, 4,
										     -187110.0, 0, 1, 2,
										        8505.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "144") == 0){
				// 34459425.0 * x * y**4 * z**4 -12162150.0 * x * y**4 * z**2 + 405405.0 * x * y**4 -12162150.0 * x * y**2 * z**4 + 4864860.0 * x * y**2 * z**2 -187110.0 * x * y**2 + 405405.0 * x * z**4 -187110.0 * x * z**2 + 8505.0 * x
				int numTerm = 9;
				double parametersVal[] = {  34459425.0, 1, 4, 4,
										   -12162150.0, 1, 4, 2,
										      405405.0, 1, 4, 0,
										   -12162150.0, 1, 2, 4,
										     4864860.0, 1, 2, 2,
										     -187110.0, 1, 2, 0,
										      405405.0, 1, 0, 4,
										     -187110.0, 1, 0, 2,
										        8505.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "432") == 0){
				// 34459425.0 * x**4 * y**3 * z**2 -2027025.0 * x**4 * y**3 -6081075.0 * x**4 * y * z**2 + 405405.0 * x**4 * y 
			 	//-12162150.0 * x**2 * y**3 * z**2 + 810810.0 * x**2 * y**3 + 2432430.0 * x**2 * y * z**2 -187110.0 * x**2 * y + 
				//405405.0 * y**3 * z**2 -31185.0 * y**3 -93555.0 * y * z**2 + 8505.0 * y 
				int numTerm = 12;
				double parametersVal[] = {  34459425.0, 4, 3, 2,
										    -2027025.0, 4, 3, 0,
										    -6081075.0, 4, 1, 2,
										      405405.0, 4, 1, 0,
										   -12162150.0, 2, 3, 2,
										      810810.0, 2, 3, 0,
										     2432430.0, 2, 1, 2,
										     -187110.0, 2, 1, 0,
										      405405.0, 0, 3, 2,
										      -31185.0, 0, 3, 0,
										      -93555.0, 0, 1, 2,
										        8505.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "342") == 0){
				// 34459425.0 * x**3 * y**4 * z**2 -2027025.0 * x**3 * y**4 -12162150.0 * x**3 * y**2 * z**2 + 810810.0 * x**3 * y**2 + 405405.0 * x**3 * z**2 -31185.0 * x**3 -6081075.0 * x * y**4 * z**2 + 405405.0 * x * y**4 + 2432430.0 * x * y**2 * z**2 -187110.0 * x * y**2 -93555.0 * x * z**2 + 8505.0 * x
				int numTerm = 12;
				double parametersVal[] = {  34459425.0, 3, 4, 2,
										    -2027025.0, 3, 4, 0,
										    -6081075.0, 1, 4, 2,
										      405405.0, 1, 4, 0,
										   -12162150.0, 3, 2, 2,
										      810810.0, 3, 2, 0,
										     2432430.0, 1, 2, 2,
										     -187110.0, 1, 2, 0,
										      405405.0, 3, 0, 2,
										      -31185.0, 3, 0, 0,
										      -93555.0, 1, 0, 2,
										        8505.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "423") == 0){
				// 34459425.0 * x**4 * y**2 * z**3 -6081075.0 * x**4 * y**2 * z -2027025.0 * x**4 * z**3 + 405405.0 * x**4 * z -12162150.0 * x**2 * y**2 * z**3 + 2432430.0 * x**2 * y**2 * z + 810810.0 * x**2 * z**3 -187110.0 * x**2 * z + 405405.0 * y**2 * z**3 -93555.0 * y**2 * z -31185.0 * z**3 + 8505.0 * z
				int numTerm = 12;
				double parametersVal[] = {  34459425.0, 4, 2, 3,
										    -2027025.0, 4, 0, 3,
										    -6081075.0, 4, 2, 1,
										      405405.0, 4, 0, 1,
										   -12162150.0, 2, 2, 3,
										      810810.0, 2, 0, 3,
										     2432430.0, 2, 2, 1,
										     -187110.0, 2, 0, 1,
										      405405.0, 0, 2, 3,
										      -31185.0, 0, 0, 3,
										      -93555.0, 0, 2, 1,
										        8505.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "324") == 0){
				// 34459425.0 * x**3 * y**2 * z**4 -12162150.0 * x**3 * y**2 * z**2 + 405405.0 * x**3 * y**2 -2027025.0 * x**3 * z**4 + 810810.0 * x**3 * z**2 -31185.0 * x**3 -6081075.0 * x * y**2 * z**4 + 2432430.0 * x * y**2 * z**2 -93555.0 * x * y**2 + 405405.0 * x * z**4 -187110.0 * x * z**2 + 8505.0 * x
				int numTerm = 12;
				double parametersVal[] = {  34459425.0, 3, 2, 4,
										    -2027025.0, 3, 0, 4,
										    -6081075.0, 1, 2, 4,
										      405405.0, 1, 0, 4,
										   -12162150.0, 3, 2, 2,
										      810810.0, 3, 0, 2,
										     2432430.0, 1, 2, 2,
										     -187110.0, 1, 0, 2,
										      405405.0, 3, 2, 0,
										      -31185.0, 3, 0, 0,
										      -93555.0, 1, 2, 0,
										        8505.0, 1, 0, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 7, 1};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "243") == 0){
				// 34459425.0 * x**2 * y**4 * z**3 -6081075.0 * x**2 * y**4 * z -12162150.0 * x**2 * y**2 * z**3 + 2432430.0 * x**2 * y**2 * z + 405405.0 * x**2 * z**3 -93555.0 * x**2 * z -2027025.0 * y**4 * z**3 + 405405.0 * y**4 * z + 810810.0 * y**2 * z**3 -187110.0 * y**2 * z -31185.0 * z**3 + 8505.0 * z 
				int numTerm = 12;
				double parametersVal[] = {  34459425.0, 2, 4, 3,
										    -2027025.0, 0, 4, 3,
										    -6081075.0, 2, 4, 1,
										      405405.0, 0, 4, 1,
										   -12162150.0, 2, 2, 3,
										      810810.0, 0, 2, 3,
										     2432430.0, 2, 2, 1,
										     -187110.0, 0, 2, 1,
										      405405.0, 2, 0, 3,
										      -31185.0, 0, 0, 3,
										      -93555.0, 2, 0, 1,
										        8505.0, 0, 0, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 3};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "234") == 0){
				// 34459425.0 * x**2 * y**3 * z**4 -12162150.0 * x**2 * y**3 * z**2 + 405405.0 * x**2 * y**3 -6081075.0 * x**2 * y * z**4 + 2432430.0 * x**2 * y * z**2 -93555.0 * x**2 * y -2027025.0 * y**3 * z**4 + 810810.0 * y**3 * z**2 -31185.0 * y**3 + 405405.0 * y * z**4 -187110.0 * y * z**2 + 8505.0 * y
				int numTerm = 12;
				double parametersVal[] = {  34459425.0, 2, 3, 4,
										    -2027025.0, 0, 3, 4,
										    -6081075.0, 2, 1, 4,
										      405405.0, 0, 1, 4,
										   -12162150.0, 2, 3, 2,
										      810810.0, 0, 3, 2,
										     2432430.0, 2, 1, 2,
										     -187110.0, 0, 1, 2,
										      405405.0, 2, 3, 0,
										      -31185.0, 0, 3, 0,
										      -93555.0, 2, 1, 0,
										        8505.0, 0, 1, 0};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 7, 2};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else if (strcmp(n, "333") == 0){
				// 34459425.0 * x**3 * y**3 * z**3 -6081075.0 * x**3 * y**3 * z -6081075.0 * x**3 * y * z**3 + 1216215.0 * x**3 * y * z 
				// -6081075.0 * x * y**3 * z**3 + 1216215.0 * x * y**3 * z + 1216215.0 * x * y * z**3 -280665.0 * x * y * z
				int numTerm = 8;
				double parametersVal[] = {   34459425.0, 3, 3, 3,
										     -6081075.0, 3, 3, 1,
										     -6081075.0, 3, 1, 3,
										      1216215.0, 3, 1, 1,
										     -6081075.0, 1, 3, 3,
										      1216215.0, 1, 3, 1,
										      1216215.0, 1, 1, 3,
										      -280665.0, 1, 1, 1};
				double *parameters = parametersVal;
				int typesVal[] = {7, 7, 7, 7, 7, 7, 7, 7};
				int *types = typesVal;
				evaluateSeriesPolynomial(x, y, z, parameters, types, numTerm, size, result);
			}
			else{
				printf("\nWARNING: n is not valid %s \n", n);
			}

			break;


		default:
			printf("\nWARNING: l is not valid %d \n", l);
			break;
	}

	//int i;
	for (i = 0; i < size; i++){	
		if (r[i] > rCutoff){
			result[i] = 0.0;
		}
	}

	//free(uncutResult);
	free(r);
	// return result;
}

/**
* @brief  Calculate the stencil using a combination of angular and radial functions
*		 for descriptor calculation.
*
* 		This function calls the MaxwellCartesianSphericalHarmonics and LegendrePolynomial functions
* 		to calculate the stencil for convolution.
*/
void calculateStencil(const int stencilDimX, const int stencilDimY, const int stencilDimZ, const double hx, const double hy, const double hz, 
					  const double rCutoff, const int l, const char *n, const int radialFunctionType, const int radialFunctionOrder, 
					  const double *U, const int accuracy, double *stencil)
{
	int pixelEvalArrSize = accuracy * accuracy * accuracy;
	
	double dv = calcDv(hx, hy, hz, accuracy,U);

	double *refX = calloc( pixelEvalArrSize, sizeof(double));
	double *refY = calloc( pixelEvalArrSize, sizeof(double));
	double *refZ = calloc( pixelEvalArrSize, sizeof(double));
	
	getCentralCoords(hx, hy, hz, accuracy, refX, refY, refZ);

	int centerX = (stencilDimX - 1)/2;
    int centerY = (stencilDimY - 1)/2;
    int centerZ = (stencilDimZ - 1)/2;
	
	double *tempXArr = calloc( pixelEvalArrSize, sizeof(double));
	double *tempYArr = calloc( pixelEvalArrSize, sizeof(double));
	double *tempZArr = calloc( pixelEvalArrSize, sizeof(double));
	double *tempMCSHResult = calloc( pixelEvalArrSize, sizeof(double));
	double *tempRadialResult = calloc( pixelEvalArrSize, sizeof(double));
	double xOffset, yOffset, zOffset;
	int i, j, k, index = 0;
	for (k = 0; k < stencilDimZ; k++){
		for ( j = 0; j < stencilDimY; j++) {
			for ( i = 0; i < stencilDimX; i++) {
				xOffset = (i-centerX) * hx;
				yOffset = (j-centerY) * hy;
				zOffset = (k-centerZ) * hz;

				addScalarVector(refX, xOffset, tempXArr, pixelEvalArrSize);
				addScalarVector(refY, yOffset, tempYArr, pixelEvalArrSize);
				addScalarVector(refZ, zOffset, tempZArr, pixelEvalArrSize);

				applyU2(tempXArr, tempYArr, tempZArr, U, pixelEvalArrSize);
				MaxwellCartesianSphericalHarmonics(tempXArr, tempYArr, tempZArr, l, n, rCutoff, tempMCSHResult, pixelEvalArrSize);

				if (radialFunctionType == 2)
				{
					LegendrePolynomial(tempXArr, tempYArr, tempZArr, radialFunctionOrder, rCutoff, tempRadialResult, pixelEvalArrSize);
					multiplyVector(tempMCSHResult, tempRadialResult, tempMCSHResult, pixelEvalArrSize);
				}
				stencil[index] = sumArr(tempMCSHResult, pixelEvalArrSize) * dv;
				index++;
			}
		}
	}
	free(refX);
	free(refY);
	free(refZ);
	free(tempXArr);
	free(tempYArr);
	free(tempZArr);
	free(tempMCSHResult);
}

/**
* @brief  function to calculate the convolution of the stencil with the image.
* 		This function calls the calculateStencil function to calculate the stencil and then convolves with the image.	    
*/
void calcStencilAndConvolveAndAddResult(const double *image, MULTIPOLE_OBJ *mp, const double rCutoff, const int l, const char *n, 
					const int radialFunctionType, const int radialFunctionOrder, const double *U, double *convolveResult,
					int DMVerts[6])
{
	double start_t, end_stencil_t, end_convolve_t; 
	//time(&start_t); 
	start_t = MPI_Wtime();


	int stencilDimX, stencilDimY, stencilDimZ;

	GetDimensionsPlane(mp, U, &stencilDimX, &stencilDimY, &stencilDimZ);

	double *stencil = calloc( stencilDimX * stencilDimY * stencilDimZ, sizeof(double));
	calculateStencil(stencilDimX, stencilDimY, stencilDimZ, mp->hx, mp->hy, mp->hz, 
					 rCutoff, l, n, radialFunctionType, radialFunctionOrder, U, mp->accuracy, stencil);

	end_stencil_t  = MPI_Wtime();

	convolve6(image, stencil, mp->imageDimX, mp->imageDimY, mp->imageDimZ, stencilDimX, stencilDimY, stencilDimZ, convolveResult, DMVerts);

// #define DEBUGCONV
#ifdef DEBUGCONV
	#define max(a,b) ((a)>(b)?(a):(b))
	double *ref_conv_result = calloc( imageDimX * imageDimY * imageDimZ, sizeof(double));
	convolve5(image, stencil, imageDimX, imageDimY, imageDimZ, stencilDimX, stencilDimY, stencilDimZ, ref_conv_result);
	int is = DMVerts[0];
	int ie = DMVerts[1];
	int js = DMVerts[2];
	int je = DMVerts[3];
	int ks = DMVerts[4];
	int ke = DMVerts[5];
	// check convolve6 answer
	double err = 0.0;
	int outputIndex = 0;
	for (int k = ks; k <= ke; k++) {
		for (int j = js; j <= je; j++) {
			for (int i = is; i <= ie; i++) {
				int ind_global = k * imageDimX * imageDimY + j * imageDimX + i;
				err = max(fabs(convolveResult[outputIndex] - ref_conv_result[ind_global]), err);
				outputIndex++;
			}
		}
	}
	assert(err < 1e-12);
	printf("Test passed!\n");
	free(ref_conv_result);
#endif

	end_convolve_t  = MPI_Wtime();
	
	int featDimX = DMVerts[1] - DMVerts[0] + 1;
	int featDimY = DMVerts[3] - DMVerts[2] + 1;
	int featDimZ = DMVerts[5] - DMVerts[4] + 1;
	int featSize = featDimX * featDimY * featDimZ; // size of local part of the feature vector
	printf("\n r: %f \t l: %d \t n: %s \t total_time: %f \t stencil: %f \t convolve: %f \n",rCutoff, l, n,end_convolve_t - start_t, end_stencil_t - start_t, end_convolve_t - end_stencil_t);
	free(stencil);
}

/**
* @brief  function to calculate the components (groups) for a given angular and radial order.
*
*/
void calcFeature( const int type, const double *image, MULTIPOLE_OBJ *mp, const double rCutoff, const int l, char **n_list,
		const int radialFunctionType, const int radialFunctionOrder, const double *U, double coeff,
		double *componentGroup, int DMVerts[6])
{	
	// type 1: one member group
	// type 2: three member group
	// type 3: six member group

	int featDimX = DMVerts[1] - DMVerts[0] + 1;
	int featDimY = DMVerts[3] - DMVerts[2] + 1;
	int featDimZ = DMVerts[5] - DMVerts[4] + 1;
	int featSize = featDimX * featDimY * featDimZ; // size of local part of the feature vector
	int imageSize = mp->imageDimX * mp->imageDimY * mp->imageDimZ;// this is the global vector size
	// int imageSize = featSize; // this is technically incorrect, since we don't distribute image like feature currently, but we should
	
	if (type == 1){
		double *component1 = calloc( imageSize, sizeof(double));
		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[0], radialFunctionType, radialFunctionOrder, U, component1, DMVerts);
		powVector(component1, 2, component1, imageSize);
		multiplyScalarVector(component1, coeff, componentGroup, imageSize);
	}
	else if (type == 2){
		double *component1 = calloc( imageSize, sizeof(double));
		double *component2 = calloc( imageSize, sizeof(double));
		double *component3 = calloc( imageSize, sizeof(double));

		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[0], radialFunctionType, radialFunctionOrder, U, component1, DMVerts);
		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[1], radialFunctionType, radialFunctionOrder, U, component2, DMVerts);
		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[2], radialFunctionType, radialFunctionOrder, U, component3, DMVerts);

		powVector(component1, 2, component1, imageSize);
		powVector(component2, 2, component2, imageSize);
		powVector(component3, 2, component3, imageSize);

		multiplyScalarVector(component1, coeff, component1, imageSize);
		multiplyScalarVector(component2, coeff, component2, imageSize);
		multiplyScalarVector(component3, coeff, component3, imageSize);

		addVector(component1, component2, componentGroup, imageSize);
		addVector(componentGroup, component3, componentGroup, imageSize);

		free(component1);
		free(component2);
		free(component3);
	}
	else if (type == 3){
		double *component1 = calloc( imageSize, sizeof(double));
		double *component2 = calloc( imageSize, sizeof(double));
		double *component3 = calloc( imageSize, sizeof(double));
		double *component4 = calloc( imageSize, sizeof(double));
		double *component5 = calloc( imageSize, sizeof(double));
		double *component6 = calloc( imageSize, sizeof(double));

		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[0], radialFunctionType, radialFunctionOrder, U, component1, DMVerts);
		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[1], radialFunctionType, radialFunctionOrder, U, component2, DMVerts);
		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[2], radialFunctionType, radialFunctionOrder, U, component3, DMVerts);
		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[3], radialFunctionType, radialFunctionOrder, U, component4, DMVerts);
		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[4], radialFunctionType, radialFunctionOrder, U, component5, DMVerts);
		calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, n_list[5], radialFunctionType, radialFunctionOrder, U, component6, DMVerts);

		powVector(component1, 2, component1, imageSize);
		powVector(component2, 2, component2, imageSize);
		powVector(component3, 2, component3, imageSize);
		powVector(component4, 2, component4, imageSize);
		powVector(component5, 2, component5, imageSize);
		powVector(component6, 2, component6, imageSize);

		multiplyScalarVector(component1, coeff, component1, imageSize);
		multiplyScalarVector(component2, coeff, component2, imageSize);
		multiplyScalarVector(component3, coeff, component3, imageSize);
		multiplyScalarVector(component4, coeff, component4, imageSize);
		multiplyScalarVector(component5, coeff, component5, imageSize);
		multiplyScalarVector(component6, coeff, component6, imageSize);

		addVector(component1, component2, componentGroup, imageSize);
		addVector(componentGroup, component3, componentGroup, imageSize);
		addVector(componentGroup, component4, componentGroup, imageSize);
		addVector(componentGroup, component5, componentGroup, imageSize);
		addVector(componentGroup, component6, componentGroup, imageSize);

		free(component1);
		free(component2);
		free(component3);
		free(component4);
		free(component5);
		free(component6);
	}
}

/**
* @brief  function to calculate the final descriptor for a given angular and radial order.
* 		
*        This function calls the calcFeature function number of times depending on the number of groups
*        for a given angular and radial order.
*/
void prepareMCSHFeatureAndSave(const double *image, MULTIPOLE_OBJ *mp, const double rCutoff, const int l,
                              const int radialFunctionType, const int radialFunctionOrder, const double *U,
			      int DMVerts[6], MPI_Comm comm, double *featureVectorGlobal)
{	
	int featDimX = DMVerts[1] - DMVerts[0] + 1;
	int featDimY = DMVerts[3] - DMVerts[2] + 1;
	int featDimZ = DMVerts[5] - DMVerts[4] + 1;
	int featSize = featDimX * featDimY * featDimZ; // size of local part of the feature vector
	
	int imageSize = mp->imageDimX * mp->imageDimY * mp->imageDimZ;
	double *featureVector = calloc( imageSize, sizeof(double));

	switch (l) {
		case 0:
			calcStencilAndConvolveAndAddResult(image, mp, rCutoff, l, "000", radialFunctionType, radialFunctionOrder, U, featureVector, DMVerts);
			break;

		case 1:
			{
				//group = 1
				char *n_list_11[] = {"100", "010", "001"};
				char **n_list_ptr_11 = n_list_11;
				double *componentGroup11 = calloc( imageSize, sizeof(double));
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_11, radialFunctionType, radialFunctionOrder, U, 1.0, componentGroup11, DMVerts);	
				sqrtVector(componentGroup11, featureVector, imageSize);
			}
			break;

		case 2:
			{
				//group = 1 and group = 2 combined
				double *componentGroup21 = calloc( imageSize, sizeof(double));
				double *componentGroup22 = calloc( imageSize, sizeof(double));
				char *n_list_21[] = {"200", "020", "002"};
				char *n_list_22[] = {"110", "101", "011"};
				char **n_list_ptr_21 = n_list_21;
				char **n_list_ptr_22 = n_list_22;

				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_21, radialFunctionType, radialFunctionOrder, U, 1.0, componentGroup21, DMVerts);	
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_22, radialFunctionType, radialFunctionOrder, U, 2.0, componentGroup22, DMVerts);			

				addVector(componentGroup21, componentGroup22, featureVector, imageSize);
				sqrtVector(featureVector, featureVector, imageSize);
			}
			break;

		case 3:
			{
				double *componentGroup31 = calloc( imageSize, sizeof(double));
				double *componentGroup32 = calloc( imageSize, sizeof(double));
				double *componentGroup33 = calloc( imageSize, sizeof(double));

				char *n_list_31[] = {"300", "030", "003"};
				char *n_list_32[] = {"210", "120", "201", "102", "021", "012"};
				char *n_list_33[] = {"111"};
				char **n_list_ptr_31 = n_list_31;
				char **n_list_ptr_32 = n_list_32;
				char **n_list_ptr_33 = n_list_33;

				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_31, radialFunctionType, radialFunctionOrder, U, 1.0, componentGroup31, DMVerts);	
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_32, radialFunctionType, radialFunctionOrder, U, 3.0, componentGroup32, DMVerts);			
				calcFeature(1, image, mp, rCutoff, l, n_list_ptr_33, radialFunctionType, radialFunctionOrder, U, 6.0, componentGroup33, DMVerts);
				addVector(componentGroup31, componentGroup32, featureVector, imageSize);
				addVector(featureVector, componentGroup33, featureVector, imageSize);
				sqrtVector(featureVector, featureVector, imageSize);
			}
			break;
			
		case 4:
			{
				double *componentGroup41 = calloc( imageSize, sizeof(double));
				double *componentGroup42 = calloc( imageSize, sizeof(double));
				double *componentGroup43 = calloc( imageSize, sizeof(double));
				double *componentGroup44 = calloc( imageSize, sizeof(double));

				char *n_list_41[] = {"400", "040", "004"};
				char *n_list_42[] = {"310", "130", "301", "103", "031", "013"};
				char *n_list_43[] = {"220", "202", "022"};
				char *n_list_44[] = {"211", "121", "112"};

				char **n_list_ptr_41 = n_list_41;
				char **n_list_ptr_42 = n_list_42;
				char **n_list_ptr_43 = n_list_43;
				char **n_list_ptr_44 = n_list_44;

				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_41, radialFunctionType, radialFunctionOrder, U, 1.0, componentGroup41, DMVerts);	
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_42, radialFunctionType, radialFunctionOrder, U, 4.0, componentGroup42, DMVerts);			
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_43, radialFunctionType, radialFunctionOrder, U, 6.0, componentGroup43, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_44, radialFunctionType, radialFunctionOrder, U, 12.0, componentGroup44, DMVerts);
				
				addVector(componentGroup41, componentGroup42, featureVector, imageSize);
				addVector(featureVector, componentGroup43, featureVector, imageSize);
				addVector(featureVector, componentGroup44, featureVector, imageSize);

				sqrtVector(featureVector, featureVector, imageSize);
			}
			break;

		case 5:
			{
				double *componentGroup51 = calloc( imageSize, sizeof(double));
				double *componentGroup52 = calloc( imageSize, sizeof(double));
				double *componentGroup53 = calloc( imageSize, sizeof(double));
				double *componentGroup54 = calloc( imageSize, sizeof(double));
				double *componentGroup55 = calloc( imageSize, sizeof(double));

				char *n_list_51[] = {"500", "050", "005"};
				char *n_list_52[] = {"410", "140", "401", "104", "041", "014"};
				char *n_list_53[] = {"320", "230", "302", "203", "032", "023"};
				char *n_list_54[] = {"311", "131", "113"};
				char *n_list_55[] = {"221", "212", "122"};

				char **n_list_ptr_51 = n_list_51;
				char **n_list_ptr_52 = n_list_52;
				char **n_list_ptr_53 = n_list_53;
				char **n_list_ptr_54 = n_list_54;
				char **n_list_ptr_55 = n_list_55;

				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_51, radialFunctionType, radialFunctionOrder, U, 1.0, componentGroup51, DMVerts);	
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_52, radialFunctionType, radialFunctionOrder, U, 5.0, componentGroup52, DMVerts);			
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_53, radialFunctionType, radialFunctionOrder, U, 10.0, componentGroup53, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_54, radialFunctionType, radialFunctionOrder, U, 20.0, componentGroup54, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_55, radialFunctionType, radialFunctionOrder, U, 30.0, componentGroup55, DMVerts);
				
				addVector(componentGroup51, componentGroup52, featureVector, imageSize);
				addVector(featureVector, componentGroup53, featureVector, imageSize);
				addVector(featureVector, componentGroup54, featureVector, imageSize);
				addVector(featureVector, componentGroup55, featureVector, imageSize);

				sqrtVector(featureVector, featureVector, imageSize);
			}
			break;
			
		case 6:
			{
				double *componentGroup61 = calloc( imageSize, sizeof(double));
				double *componentGroup62 = calloc( imageSize, sizeof(double));
				double *componentGroup63 = calloc( imageSize, sizeof(double));
				double *componentGroup64 = calloc( imageSize, sizeof(double));
				double *componentGroup65 = calloc( imageSize, sizeof(double));
				double *componentGroup66 = calloc( imageSize, sizeof(double));
				double *componentGroup67 = calloc( imageSize, sizeof(double));

				char *n_list_61[] = {"600", "060", "006"};
				char *n_list_62[] = {"510", "150", "501", "105", "051", "015"};
				char *n_list_63[] = {"420", "240", "402", "204", "042", "024"};
				char *n_list_64[] = {"411", "141", "114"};
				char *n_list_65[] = {"330", "303", "033"};
				char *n_list_66[] = {"321", "231", "312", "213", "132", "123"};
				char *n_list_67[] = {"222"};

				char **n_list_ptr_61 = n_list_61;
				char **n_list_ptr_62 = n_list_62;
				char **n_list_ptr_63 = n_list_63;
				char **n_list_ptr_64 = n_list_64;
				char **n_list_ptr_65 = n_list_65;
				char **n_list_ptr_66 = n_list_66;
				char **n_list_ptr_67 = n_list_67;

				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_61, radialFunctionType, radialFunctionOrder, U, 1.0, componentGroup61, DMVerts);	
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_62, radialFunctionType, radialFunctionOrder, U, 6.0, componentGroup62, DMVerts);			
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_63, radialFunctionType, radialFunctionOrder, U, 15.0, componentGroup63, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_64, radialFunctionType, radialFunctionOrder, U, 30.0, componentGroup64, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_65, radialFunctionType, radialFunctionOrder, U, 20.0, componentGroup65, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_66, radialFunctionType, radialFunctionOrder, U, 60.0, componentGroup66, DMVerts);
				calcFeature(1, image, mp, rCutoff, l, n_list_ptr_67, radialFunctionType, radialFunctionOrder, U, 90.0, componentGroup67, DMVerts);
				
				addVector(componentGroup61, componentGroup62, featureVector, imageSize);
				addVector(featureVector, componentGroup63, featureVector, imageSize);
				addVector(featureVector, componentGroup64, featureVector, imageSize);
				addVector(featureVector, componentGroup65, featureVector, imageSize);
				addVector(featureVector, componentGroup66, featureVector, imageSize);
				addVector(featureVector, componentGroup67, featureVector, imageSize);

				sqrtVector(featureVector, featureVector, imageSize);
			}
			break;
			
		case 7:
			{
				double *componentGroup71 = calloc( imageSize, sizeof(double));
				double *componentGroup72 = calloc( imageSize, sizeof(double));
				double *componentGroup73 = calloc( imageSize, sizeof(double));
				double *componentGroup74 = calloc( imageSize, sizeof(double));
				double *componentGroup75 = calloc( imageSize, sizeof(double));
				double *componentGroup76 = calloc( imageSize, sizeof(double));
				double *componentGroup77 = calloc( imageSize, sizeof(double));
				double *componentGroup78 = calloc( imageSize, sizeof(double));

				char *n_list_71[] = {"700", "070", "007"};
				char *n_list_72[] = {"610", "160", "601", "106", "061", "016"};
				char *n_list_73[] = {"520", "250", "502", "205", "052", "025"};
				char *n_list_74[] = {"511", "151", "115"};
				char *n_list_75[] = {"430", "340", "403", "304", "043", "034"};
				char *n_list_76[] = {"421", "241", "412", "214", "142", "124"};
				char *n_list_77[] = {"331", "313", "133"};
				char *n_list_78[] = {"322", "232", "223"};

				char **n_list_ptr_71 = n_list_71;
				char **n_list_ptr_72 = n_list_72;
				char **n_list_ptr_73 = n_list_73;
				char **n_list_ptr_74 = n_list_74;
				char **n_list_ptr_75 = n_list_75;
				char **n_list_ptr_76 = n_list_76;
				char **n_list_ptr_77 = n_list_77;
				char **n_list_ptr_78 = n_list_78;

				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_71, radialFunctionType, radialFunctionOrder, U, 1.0, componentGroup71, DMVerts);	
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_72, radialFunctionType, radialFunctionOrder, U, 7.0, componentGroup72, DMVerts);			
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_73, radialFunctionType, radialFunctionOrder, U, 21.0, componentGroup73, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_74, radialFunctionType, radialFunctionOrder, U, 42.0, componentGroup74, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_75, radialFunctionType, radialFunctionOrder, U, 35.0, componentGroup75, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_76, radialFunctionType, radialFunctionOrder, U, 105.0, componentGroup76, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_77, radialFunctionType, radialFunctionOrder, U, 140.0, componentGroup77, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_78, radialFunctionType, radialFunctionOrder, U, 210.0, componentGroup78, DMVerts);
				
				addVector(componentGroup71, componentGroup72, featureVector, imageSize);
				addVector(featureVector, componentGroup73, featureVector, imageSize);
				addVector(featureVector, componentGroup74, featureVector, imageSize);
				addVector(featureVector, componentGroup75, featureVector, imageSize);
				addVector(featureVector, componentGroup76, featureVector, imageSize);
				addVector(featureVector, componentGroup77, featureVector, imageSize);
				addVector(featureVector, componentGroup78, featureVector, imageSize);

				sqrtVector(featureVector, featureVector, imageSize);
			}
			break;

		case 8:
			{
				double *componentGroup81 = calloc( imageSize, sizeof(double));
				double *componentGroup82 = calloc( imageSize, sizeof(double));
				double *componentGroup83 = calloc( imageSize, sizeof(double));
				double *componentGroup84 = calloc( imageSize, sizeof(double));
				double *componentGroup85 = calloc( imageSize, sizeof(double));
				double *componentGroup86 = calloc( imageSize, sizeof(double));
				double *componentGroup87 = calloc( imageSize, sizeof(double));
				double *componentGroup88 = calloc( imageSize, sizeof(double));
				double *componentGroup89 = calloc( imageSize, sizeof(double));
				double *componentGroup810 = calloc( imageSize, sizeof(double));

				char *n_list_81[] = {"800", "080", "008"};
				char *n_list_82[] = {"710", "170", "701", "107", "071", "017"};
				char *n_list_83[] = {"620", "260", "602", "206", "062", "026"};
				char *n_list_84[] = {"611", "161", "116"};
				char *n_list_85[] = {"530", "350", "503", "305", "053", "035"};
				char *n_list_86[] = {"521", "251", "512", "215", "152", "125"};
				char *n_list_87[] = {"440", "404", "044"};
				char *n_list_88[] = {"431", "341", "413", "314", "143", "134"};
				char *n_list_89[] = {"422", "242", "224"};
				char *n_list_810[] = {"332", "323", "233"};

				char **n_list_ptr_81 = n_list_81;
				char **n_list_ptr_82 = n_list_82;
				char **n_list_ptr_83 = n_list_83;
				char **n_list_ptr_84 = n_list_84;
				char **n_list_ptr_85 = n_list_85;
				char **n_list_ptr_86 = n_list_86;
				char **n_list_ptr_87 = n_list_87;
				char **n_list_ptr_88 = n_list_88;
				char **n_list_ptr_89 = n_list_89;
				char **n_list_ptr_810 = n_list_810;

				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_81, radialFunctionType, radialFunctionOrder, U, 1.0, componentGroup81, DMVerts);	
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_82, radialFunctionType, radialFunctionOrder, U, 8.0, componentGroup82, DMVerts);			
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_83, radialFunctionType, radialFunctionOrder, U, 28.0, componentGroup83, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_84, radialFunctionType, radialFunctionOrder, U, 56.0, componentGroup84, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_85, radialFunctionType, radialFunctionOrder, U, 56.0, componentGroup85, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_86, radialFunctionType, radialFunctionOrder, U, 168.0, componentGroup86, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_87, radialFunctionType, radialFunctionOrder, U, 70.0, componentGroup87, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_88, radialFunctionType, radialFunctionOrder, U, 280.0, componentGroup88, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_89, radialFunctionType, radialFunctionOrder, U, 420.0, componentGroup89, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_810, radialFunctionType, radialFunctionOrder, U, 560.0, componentGroup810, DMVerts);
				
				addVector(componentGroup81, componentGroup82, featureVector, imageSize);
				addVector(featureVector, componentGroup83, featureVector, imageSize);
				addVector(featureVector, componentGroup84, featureVector, imageSize);
				addVector(featureVector, componentGroup85, featureVector, imageSize);
				addVector(featureVector, componentGroup86, featureVector, imageSize);
				addVector(featureVector, componentGroup87, featureVector, imageSize);
				addVector(featureVector, componentGroup88, featureVector, imageSize);
				addVector(featureVector, componentGroup89, featureVector, imageSize);
				addVector(featureVector, componentGroup810, featureVector, imageSize);

				sqrtVector(featureVector, featureVector, imageSize);
			}
			break;
			
		case 9:
			{
				double *componentGroup91 = calloc( imageSize, sizeof(double));
				double *componentGroup92 = calloc( imageSize, sizeof(double));
				double *componentGroup93 = calloc( imageSize, sizeof(double));
				double *componentGroup94 = calloc( imageSize, sizeof(double));
				double *componentGroup95 = calloc( imageSize, sizeof(double));
				double *componentGroup96 = calloc( imageSize, sizeof(double));
				double *componentGroup97 = calloc( imageSize, sizeof(double));
				double *componentGroup98 = calloc( imageSize, sizeof(double));
				double *componentGroup99 = calloc( imageSize, sizeof(double));
				double *componentGroup910 = calloc( imageSize, sizeof(double));
				double *componentGroup911 = calloc( imageSize, sizeof(double));
				double *componentGroup912 = calloc( imageSize, sizeof(double));

				char *n_list_91[] = {"900", "090", "009"};
				char *n_list_92[] = {"810", "180", "801", "108", "081", "018"};
				char *n_list_93[] = {"720", "270", "702", "207", "072", "027"};
				char *n_list_94[] = {"711", "171", "117"};
				char *n_list_95[] = {"630", "360", "603", "306", "063", "036"};
				char *n_list_96[] = {"621", "261", "612", "216", "162", "126"};
				char *n_list_97[] = {"540", "450", "504", "405", "054", "045"};
				char *n_list_98[] = {"531", "351", "513", "315", "153", "135"};
				char *n_list_99[] = {"522", "252", "225"};
				char *n_list_910[] = {"441", "414", "144"};
				char *n_list_911[] = {"432", "342", "423", "324", "243", "234"};
				char *n_list_912[] = {"333"};

				char **n_list_ptr_91 = n_list_91;
				char **n_list_ptr_92 = n_list_92;
				char **n_list_ptr_93 = n_list_93;
				char **n_list_ptr_94 = n_list_94;
				char **n_list_ptr_95 = n_list_95;
				char **n_list_ptr_96 = n_list_96;
				char **n_list_ptr_97 = n_list_97;
				char **n_list_ptr_98 = n_list_98;
				char **n_list_ptr_99 = n_list_99;
				char **n_list_ptr_910 = n_list_910;
				char **n_list_ptr_911 = n_list_911;
				char **n_list_ptr_912 = n_list_912;

				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_91, radialFunctionType, radialFunctionOrder, U, 1.0, componentGroup91, DMVerts);	
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_92, radialFunctionType, radialFunctionOrder, U, 9.0, componentGroup92, DMVerts);			
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_93, radialFunctionType, radialFunctionOrder, U, 36.0, componentGroup93, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_94, radialFunctionType, radialFunctionOrder, U, 72.0, componentGroup94, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_95, radialFunctionType, radialFunctionOrder, U, 84.0, componentGroup95, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_96, radialFunctionType, radialFunctionOrder, U, 252.0, componentGroup96, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_97, radialFunctionType, radialFunctionOrder, U, 126.0, componentGroup97, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_98, radialFunctionType, radialFunctionOrder, U, 504.0, componentGroup98, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_99, radialFunctionType, radialFunctionOrder, U, 756.0, componentGroup99, DMVerts);
				calcFeature(2, image, mp, rCutoff, l, n_list_ptr_910, radialFunctionType, radialFunctionOrder, U, 630.0, componentGroup910, DMVerts);
				calcFeature(3, image, mp, rCutoff, l, n_list_ptr_911, radialFunctionType, radialFunctionOrder, U, 1260.0, componentGroup911, DMVerts);
				calcFeature(1, image, mp, rCutoff, l, n_list_ptr_912, radialFunctionType, radialFunctionOrder, U, 1680.0, componentGroup912, DMVerts);
				
				addVector(componentGroup91, componentGroup92, featureVector, imageSize);
				addVector(featureVector, componentGroup93, featureVector, imageSize);
				addVector(featureVector, componentGroup94, featureVector, imageSize);
				addVector(featureVector, componentGroup95, featureVector, imageSize);
				addVector(featureVector, componentGroup96, featureVector, imageSize);
				addVector(featureVector, componentGroup97, featureVector, imageSize);
				addVector(featureVector, componentGroup98, featureVector, imageSize);
				addVector(featureVector, componentGroup99, featureVector, imageSize);
				addVector(featureVector, componentGroup910, featureVector, imageSize);
				addVector(featureVector, componentGroup911, featureVector, imageSize);
				addVector(featureVector, componentGroup912, featureVector, imageSize);

				sqrtVector(featureVector, featureVector, imageSize);
			}
			break;
			
		default:
			die("\nERROR: l is not valid\n");
			break;
	}

	int gridsizes[3] = {mp->imageDimX, mp->imageDimY, mp->imageDimZ};
	gather_distributed_vector(featureVector, DMVerts, featureVectorGlobal, gridsizes, comm, 1);
}

/**
* @brief  function to calculate score for a given angular order for parallelization of 
* 		  descriptor calculation.	    
*/
double scoreTask(const double rCutoff, const int l)
{
	double result;
		switch(l)
		{
			case 0:
				result = 1;
				break;
			case 1:
				result = 3;
				break;
			case 2:
				result = 6;
				break;
			case 3:
				result = 10;
				break;
			case 4:
				result = 15;
				break;
			case 5: 
				result = 21;
				break;
			case 6:
				result = 28;
				break;
			case 7:
				result = 36;
				break;
			case 8:
				result = 45;
				break;
			case 9:
				result = 55;
				break;
			
			default:
			printf("\n***** WARNING: l not valid *****\n");
			result = 1;
			break;
		}
		result *= rCutoff * rCutoff * rCutoff;
		return result;
}

/**
* @brief  function to define the parameters for orders of Legendre polynomials
*/
void LegendrePolynomial(const double *x, const double *y, const double *z, const int polynomialOrder, const double rCutoff, double *result, const int size)
{
	// (2*r_array-r)/r
	double *r = calloc( size, sizeof(double));
	getRArray(x, y, z, r, size);

	int i;
	for (i = 0; i < size; i++){
		r[i] = (2.0 * r[i] - rCutoff) / rCutoff;
	}


	if (polynomialOrder == 0){
		// 1
		for ( i = 0; i < size; i++)
		{
			result[i] = 1.0;
		}
	} else if (polynomialOrder == 1){
		// x
		for ( i = 0; i < size; i++)
		{
			result[i] = r[i];
		}
	} else if (polynomialOrder == 2){
		// 0.5 * (3*x*x - 1)
		double *temp1 = malloc( size * sizeof(double));

		polyArray(r, 2, 3.0, temp1, size);

		addScalarVector(temp1, -1.0, result, size);
		multiplyScalarVector(result, 0.5, result, size);

		free(temp1);
	} else if (polynomialOrder == 3){
		// 0.5 * (5*x*x*x - 3x)
		double *temp1 = malloc( size * sizeof(double));
		double *temp2 = malloc( size * sizeof(double));

		polyArray(r, 3, 5.0, temp1, size);

		multiplyScalarVector(r, -3.0, temp2, size);

		addVector(temp1, temp2, result, size);
		multiplyScalarVector(result, 0.5, result, size);

		free(temp1);
		free(temp2);
	} else if (polynomialOrder == 4){
		// (1/8) * (35*x*x*x*x - 30*x*x +3)
		double *temp1 = malloc( size * sizeof(double));
		double *temp2 = malloc( size * sizeof(double));

		polyArray(r, 4, 35.0, temp1, size);
		polyArray(r, 2, -30.0, temp2, size);

		addVector(temp1, temp2, result, size);
		addScalarVector(result, 3.0, result, size);
		multiplyScalarVector(result, (1.0/8.0), result, size);

		free(temp1);
		free(temp2);
	} else if (polynomialOrder == 5){
		// (1/8) * (63*x*x*x*x*x - 70*x*x*x + 15*x)
		double *temp1 = malloc( size * sizeof(double));
		double *temp2 = malloc( size * sizeof(double));
		double *temp3 = malloc( size * sizeof(double));

		polyArray(r, 5, 63.0, temp1, size);
		polyArray(r, 3, -70.0, temp2, size);
		multiplyScalarVector(r, 15.0, temp3, size);

		addVector(temp1, temp2, result, size);
		addVector(result, temp3, result, size);
		multiplyScalarVector(result, (1.0/8.0), result, size);

		free(temp1);
		free(temp2);
		free(temp3);
	} else if (polynomialOrder == 6){
		// (1/16) * (231*x*x*x*x*x*x - 315*x*x*x*x + 105*x*x -5)
		double *temp1 = malloc( size * sizeof(double));
		double *temp2 = malloc( size * sizeof(double));
		double *temp3 = malloc( size * sizeof(double));

		polyArray(r, 6, 231.0, temp1, size);
		polyArray(r, 4, -315.0, temp2, size);
		polyArray(r, 2, 105.0, temp3, size);

		addVector(temp1, temp2, result, size);
		addVector(result, temp3, result, size);
		addScalarVector(result, -5.0, result, size);
		multiplyScalarVector(result, (1.0/16.0), result, size);

		free(temp1);
		free(temp2);
		free(temp3);
	} else if (polynomialOrder == 7){
		// (1/16) * (429*x*x*x*x*x*x*x - 693*x*x*x*x*x + 315*x*x*x - 35*x)
		double *temp1 = malloc( size * sizeof(double));
		double *temp2 = malloc( size * sizeof(double));
		double *temp3 = malloc( size * sizeof(double));
		double *temp4 = malloc( size * sizeof(double));

		polyArray(r, 7, 231.0, temp1, size);
		polyArray(r, 5, -315.0, temp2, size);
		polyArray(r, 3, 105.0, temp3, size);
		multiplyScalarVector(r, 15.0, temp4, size);

		addVector(temp1, temp2, result, size);
		addVector(result, temp3, result, size);
		addVector(result, temp4, result, size);
		multiplyScalarVector(result, (1.0/16.0), result, size);

		free(temp1);
		free(temp2);
		free(temp3);
		free(temp4);
	} else if (polynomialOrder == 8){
		// (1/128) * (6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x*2 + 35)
		double *temp1 = malloc( size * sizeof(double));
		double *temp2 = malloc( size * sizeof(double));
		double *temp3 = malloc( size * sizeof(double));
		double *temp4 = malloc( size * sizeof(double));

		polyArray(r, 8, 6435.0, temp1, size);
		polyArray(r, 6, -12012.0, temp2, size);
		polyArray(r, 4, 6930.0, temp3, size);
		polyArray(r, 2, -1260.0, temp4, size);

		addVector(temp1, temp2, result, size);
		addVector(result, temp3, result, size);
		addVector(result, temp4, result, size);
		addScalarVector(result, 35.0, result, size);
		multiplyScalarVector(result, (1.0/128.0), result, size);

		free(temp1);
		free(temp2);
		free(temp3);
		free(temp4);
	} else if (polynomialOrder == 9){
		// (1/128) * (12155*x**9 - 25740*x**7 + 18018*x**5 - 4620*x**3 + 315x)
		double *temp1 = malloc( size * sizeof(double));
		double *temp2 = malloc( size * sizeof(double));
		double *temp3 = malloc( size * sizeof(double));
		double *temp4 = malloc( size * sizeof(double));
		double *temp5 = malloc( size * sizeof(double));

		polyArray(r, 9, 12155.0, temp1, size);
		polyArray(r, 7, -25740.0, temp2, size);
		polyArray(r, 5, 18018.0, temp3, size);
		polyArray(r, 3, -4620.0, temp4, size);
		multiplyScalarVector(r, 315.0, temp5, size);

		addVector(temp1, temp2, result, size);
		addVector(result, temp3, result, size);
		addVector(result, temp4, result, size);
		addVector(result, temp5, result, size);
		multiplyScalarVector(result, (1.0/128.0), result, size);

		free(temp1);
		free(temp2);
		free(temp3);
		free(temp4);
		free(temp5);
	} else if (polynomialOrder == 10){
		// (1/256) * (46189*x**10 - 109395*x**8 + 90090*x**6 - 30030*x**4 + 3465*x**2 -63)
		double *temp1 = malloc( size * sizeof(double));
		double *temp2 = malloc( size * sizeof(double));
		double *temp3 = malloc( size * sizeof(double));
		double *temp4 = malloc( size * sizeof(double));
		double *temp5 = malloc( size * sizeof(double));

		polyArray(r, 10, 46189.0, temp1, size);
		polyArray(r, 8, -109395.0, temp2, size);
		polyArray(r, 6, 90090.0, temp3, size);
		polyArray(r, 4, 30030.0, temp4, size);
		polyArray(r, 2, 3465.0, temp5, size);

		addVector(temp1, temp2, result, size);
		addVector(result, temp3, result, size);
		addVector(result, temp4, result, size);
		addVector(result, temp5, result, size);
		addScalarVector(result, -63.0, result, size);
		multiplyScalarVector(result, (1.0/256.0), result, size);

		free(temp1);
		free(temp2);
		free(temp3);
		free(temp4);
		free(temp5);
	} else {
		printf("\nERROR: Legendre Order Not Valid\n");
	}

	for (i = 0; i < size; i++){	
		if (r[i] > rCutoff)
		{
			result[i] = 0.0;
		}
	}
	free(r);
}







