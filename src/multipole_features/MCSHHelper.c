/**
* @file MCSHHelper.h
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
/* BLAS, LAPACK, LAPACKE routines */
#ifdef USE_MKL
    // #define MKL_Complex16 double complex
    #include <mkl.h>
#else
    #include <cblas.h>
#endif

#include "MCSHHelper.h"
#include "MP_types.h"

/**
* @brief function to print the error message and exit the program
*/
void die(const char *message)
{
	printf( "ERROR: %s\n", message );
	exit(1);
}

/**
* @brief function to get the plane equation formed by three points
*/
void GetPlaneEquation(const Point p1, const Point p2, const Point p3, double *a, double *b, double *c, double *d)
{
	double a1 = p2.x - p1.x;
	double b1 = p2.y - p1.y;
	double c1 = p2.z - p1.z;
	double a2 = p3.x - p1.x;
	double b2 = p3.y - p1.y;
	double c2 = p3.z - p1.z;

	double tempA = b1 * c2 - b2 * c1;
	double tempB = a2 * c1 - a1 * c2;
	double tempC = a1 * b2 - b1 * a2;
	double tempD = - tempA * p1.x - tempB * p1.y - tempC * p1.z;

	*a = tempA;
	*b = tempB;
	*c = tempC;
	*d = tempD;
}

/**
* @brief function to check if the plane intersects with the sphere
*/
int CheckPlaneIntersectWithSphere(const Point p1, const Point p2, const Point p3, const double rCutoff, const Point origin)
{
	double xs = origin.x;
	double ys = origin.y;
	double zs = origin.z;
	double a, b, c, d; 
	int result;
	GetPlaneEquation(p1, p2, p3, &a, &b, &c, &d);

	double numerator = fabs(a * xs + b * ys + c * zs + d);
	double denominator = sqrt(a*a + b*b + c*c);
	double test_d = numerator / denominator;

	if (rCutoff > test_d)
	{
	    result = 1;
	}
	else
	{
	    result = 0;
	}
	return result;
}

/**
* @brief function to transform the point using the transformation matrix
*/
Point UTransform(const double x, const double y, const double z, const double *U)
{
	double tempX = x * U[0] + y * U[1] + z * U[2];
	double tempY = x * U[3] + y * U[4] + z * U[5];
	double tempZ = x * U[6] + y * U[7] + z * U[8];

	Point p = {tempX,tempY,tempZ};

	return p;
}

/**
* @brief function to get the dimensions of the plane
*/
void GetDimensionsPlane(MULTIPOLE_OBJ *mp, const double *U, int *dimXResult, int *dimYResult, int *dimZResult)
{
	double hx = mp->hx;
	double hy = mp->hy;
	double hz = mp->hz;
	double rCutoff = mp->MCSHMaxR;
	int dimX = 2 * ceil(rCutoff / hx) + 1;
	int dimY = 2 * ceil(rCutoff / hy) + 1;
	int dimZ = 2 * ceil(rCutoff / hz) + 1;

	Point origin = {0,0,0};

	double ref_x_min, ref_x_max, ref_y_min, ref_y_max, ref_z_min, ref_z_max;
	Point p1x_1, p2x_1, p3x_1, p1x_2, p2x_2, p3x_2;  

	while (1)
	{
		ref_x_min = - hx * dimX * 0.5;
		ref_x_max = hx * dimX * 0.5;
		ref_y_min = - hy * dimY * 0.5;
		ref_y_max = hy * dimY * 0.5;
		ref_z_min = - hz * dimZ * 0.5;
		ref_z_max = hz * dimZ * 0.5;

		p1x_1 = UTransform(ref_x_min,ref_y_min,ref_z_min,U);
		p2x_1 = UTransform(ref_x_min,ref_y_max,ref_z_min,U);
		p3x_1 = UTransform(ref_x_min,ref_y_min,ref_z_max,U);

		p1x_2 = UTransform(ref_x_max,ref_y_min,ref_z_min,U);
		p2x_2 = UTransform(ref_x_max,ref_y_max,ref_z_min,U);
		p3x_2 = UTransform(ref_x_max,ref_y_min,ref_z_max,U);

		if (CheckPlaneIntersectWithSphere(p1x_1, p2x_1, p3x_1, rCutoff, origin) || CheckPlaneIntersectWithSphere(p1x_2, p2x_2, p3x_2, rCutoff, origin))
		{
			dimX += 2;
		}
		else
		{
			break;
		}
	}

	Point p1y_1, p2y_1, p3y_1, p1y_2, p2y_2, p3y_2;  

	while (1)
	{
		ref_x_min = - hx * dimX * 0.5;
		ref_x_max = hx * dimX * 0.5;
		ref_y_min = - hy * dimY * 0.5;
		ref_y_max = hy * dimY * 0.5;
		ref_z_min = - hz * dimZ * 0.5;
		ref_z_max = hz * dimZ * 0.5;

		p1y_1 = UTransform(ref_x_min,ref_y_min,ref_z_min,U);
		p2y_1 = UTransform(ref_x_max,ref_y_min,ref_z_min,U);
		p3y_1 = UTransform(ref_x_min,ref_y_min,ref_z_max,U);

		p1y_2 = UTransform(ref_x_min,ref_y_max,ref_z_min,U);
		p2y_2 = UTransform(ref_x_max,ref_y_max,ref_z_min,U);
		p3y_2 = UTransform(ref_x_min,ref_y_max,ref_z_max,U);

		if (CheckPlaneIntersectWithSphere(p1y_1, p2y_1, p3y_1, rCutoff, origin) || CheckPlaneIntersectWithSphere(p1y_2, p2y_2, p3y_2, rCutoff, origin))
		{
			dimY += 2;
		}
		else
		{
			break;
		}
	}

	Point p1z_1, p2z_1, p3z_1, p1z_2, p2z_2, p3z_2;  

	while (1)
	{
		ref_x_min = - hx * dimX * 0.5;
		ref_x_max = hx * dimX * 0.5;
		ref_y_min = - hy * dimY * 0.5;
		ref_y_max = hy * dimY * 0.5;
		ref_z_min = - hz * dimZ * 0.5;
		ref_z_max = hz * dimZ * 0.5;

		p1z_1 = UTransform(ref_x_min,ref_y_min,ref_z_min,U);
		p2z_1 = UTransform(ref_x_max,ref_y_min,ref_z_min,U);
		p3z_1 = UTransform(ref_x_min,ref_y_max,ref_z_min,U);

		p1z_2 = UTransform(ref_x_min,ref_y_min,ref_z_max,U);
		p2z_2 = UTransform(ref_x_max,ref_y_min,ref_z_max,U);
		p3z_2 = UTransform(ref_x_min,ref_y_max,ref_z_max,U);

		if (CheckPlaneIntersectWithSphere(p1z_1, p2z_1, p3z_1, rCutoff, origin) || CheckPlaneIntersectWithSphere(p1z_2, p2z_2, p3z_2, rCutoff, origin))
		{
			dimZ += 2;
		}
		else
		{
			break;
		}
	}

	*dimXResult = dimX;
	*dimYResult = dimY;
	*dimZResult = dimZ;
}

/**
* @brief function to print the array
*/
void printArr(double *arr, int size)
{
	int i;
	for ( i = 0; i < size; i++ ){
		printf("%f\n", arr[i]);
	}
}

/**
* @brief function to evaluate square root of a vector elementwise
*/
void sqrtVector(const double *arr, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		result[i] = sqrt(arr[i]);
	}
}

/**
* @brief function to evaluate power of a vector elementwise
*/
void powVector(const double *arr, const int power, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		result[i] = pow(arr[i], power);
	}
}

/**
* @brief function to add two vectors elementwise
*/
void addVector(const double *arr1, const double *arr2, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		result[i] = arr1[i] + arr2[i];
	}
}

/**
* @brief function to subtract two vectors elementwise
*/
void subtractVector(const double *arr1, const double *arr2, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		result[i] = arr1[i] - arr2[i];
	}
}

/**
* @brief function to multiply two vectors elementwise
*/
void multiplyVector(const double *arr1, const double *arr2, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		result[i] = arr1[i] * arr2[i];
	}
}

/**
* @brief function to divide two vectors elementwise
*/
void divideVector(const double *arr1, const double *arr2, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		result[i] = arr1[i] / arr2[i];
	}
}

/**
* @brief function to add a scaler to every element of a vector
*/
void addScalarVector(const double *arr, const double a, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		result[i] = arr[i] + a;
	}
}

/**
* @brief function to multiply a scaler to every element of a vector
*/
void multiplyScalarVector(const double *arr1, const double a, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		result[i] = arr1[i] * a;
	}
}

/**
* @brief function to multiply a scaler to every element of a vector and add
*		 it to the resulting vector
*/
void multiplyScalarVector_add(const double *arr1, const double a, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		result[i] += arr1[i] * a;
	}
}

/**
* @brief function to normalize position vectors
*/
void getRArray(const double *x, const double *y, const double *z, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++){
		double r = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
		result[i] = r;
	}
}

/**
* @brief function to evaluate a polynomial in a single variable
*/
void polyArray(const double *x, const int powX, const double a, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++)
	{
		result[i] = a * pow(x[i], powX);
	}
}

/**
* @brief function to evaluate a polynomial in a single variable and add to the result
*/
void polyArray_add(const double *x, const int powX, const double a, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++)
	{
		result[i] += a * pow(x[i], powX);
	}
}

/**
* @brief function to evaluate a polynomial in three variables
*/
void polyXYZArray(const double *x, const double *y, const double *z, const int powX, const int powY, const int powZ, const double a, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++)
	{
		result[i] = a * pow(x[i], powX) * pow(y[i], powY) * pow(z[i], powZ) ;
	}
}

/**
* @brief function to evaluate a polynomial in three variables and add to the result
*/
void polyXYZArray_add(const double *x, const double *y, const double *z, const int powX, const int powY, const int powZ, const double a, double *result, const int size)
{
	int i = 0;
	for (i = 0; i < size; i++)
	{
		result[i] += a * pow(x[i], powX) * pow(y[i], powY) * pow(z[i], powZ) ;
	}
}

/**
* @brief function to apply a transformation matrix to the position vectors
*/
void applyU(double *X, double *Y, double*Z, const double *U, const int size)
{
	double *combinedArr = malloc( 3 * size * sizeof(double));
	double *matMutResult = malloc( 3 * size * sizeof(double));

	int i;
	for ( i = 0; i < size; i++ )
	{
		combinedArr[ i * 3 ] = X[i];
		combinedArr[ i * 3 + 1 ] = Y[i];
		combinedArr[ i * 3 + 2 ] = Z[i];
	}

	for ( i = 0; i < size; i++ )
	{
		X[i] = matMutResult[ i * 3 ];
		Y[i] = matMutResult[ i * 3 + 1 ];
		Z[i] = matMutResult[ i * 3 + 2 ];
	}

	free(combinedArr);
	free(matMutResult);

}

/**
* @brief function to apply a transformation matrix to the position vectors
*/
void applyU2(double *X, double *Y, double*Z, const double *U, const int size)
{
	int i;
	double tempX, tempY, tempZ;
	for (i = 0; i < size; i++){
		tempX = U[0] * X[i] + U[1] * Y[i] + U[2] * Z[i];
		tempY = U[3] * X[i] + U[4] * Y[i] + U[5] * Z[i];
		tempZ = U[6] * X[i] + U[7] * Y[i] + U[8] * Z[i];

		X[i] = tempX;
		Y[i] = tempY;
		Z[i] = tempZ;
	}
}

/**
* @brief function to calculate volume of a voxel
*/
double calcDv(const double hx, const double hy, const double hz, const int accuracy, const double *U)
{
	double hxAcc = hx / accuracy;
	double hyAcc = hy / accuracy;
	double hzAcc = hz / accuracy;
	
	Point l1 = {U[0] * hxAcc, U[1] * hxAcc, U[2] * hxAcc};
	Point l2 = {U[3] * hyAcc, U[4] * hyAcc, U[5] * hyAcc};
	Point l3 = {U[6] * hzAcc, U[7] * hzAcc, U[8] * hzAcc};

	Point crossL2L3 = {l2.y * l3.z - l2.z * l3.y, l2.z * l3.x - l2.x * l3.z, l2.x * l3.y - l2.y * l3.x};

	double result = fabs(l1.x * crossL2L3.x + l1.y * crossL2L3.y + l1.z * crossL2L3.z);
	
	return result;
}

/**
* @brief function to calculate sum of all elements of an array
*/
double sumArr(const double *arr, const int size)
{	
	double result = 0;
	int i;
	for (i = 0; i < size; i++)
	{
		result += arr[i];
	}
	return result;

}

/**
* @brief function to calculate sum of absolute value of all elements of an array
*/
double sumAbsArr(const double *arr, const int size)
{	
	double result = 0;
	int i;
	for (i = 0; i < size; i++)
	{
		result += fabs(arr[i]);
	}
	return result;

}

/**
* @brief function to create an array of linearly spaced numbers
*/
void linspace(double start, double end, double *result, int num)
{	
	double stepsize = (end - start) / (double)(num-1);
	double current = start;

	int i;
	for (i = 0; i < num; i++)
	{
		result[i] = current;
		current += stepsize;
	}
}

/**
* @brief function to create a 3D meshgrid
*/
void meshgrid3D(const double *x, const double *y, const double *z, const int sizex, const int sizey, const int sizez, double *X, double *Y, double *Z)
{
	int i,j,k;
	
	for (k = 0; k < sizez; k++){
		for ( j = 0; j < sizey; j++) {
			for ( i = 0; i < sizex; i++) {
				X[ k * sizex * sizey + j * sizex + i ] = x[i];
				Y[ k * sizex * sizey + j * sizex + i ] = y[j];
				Z[ k * sizex * sizey + j * sizex + i ] = z[k];
			}
		}
	}
}

/**
* @brief function to get the reference coordinates of the voxel
*/
void getCentralCoords(const double hx, const double hy, const double hz, const int accuracy, double *refX, double *refY, double *refZ)
{
	double hxAcc = hx / accuracy;
	double hyAcc = hy / accuracy;
	double hzAcc = hz / accuracy;

	double *ref_x_li = calloc( accuracy, sizeof(double));
	double *ref_y_li = calloc( accuracy, sizeof(double));
	double *ref_z_li = calloc( accuracy, sizeof(double));

	double xStart = -((hx / 2.0) - (hxAcc / 2.0));
	double xEnd = (hx / 2.0) - (hxAcc / 2.0);
	linspace(xStart, xEnd, ref_x_li, accuracy);

	double yStart = -((hy / 2.0) - (hyAcc / 2.0));
	double yEnd = (hy / 2.0) - (hyAcc / 2.0);
	linspace(yStart, yEnd, ref_y_li, accuracy);

	double zStart = -((hz / 2.0) - (hzAcc / 2.0));
	double zEnd = (hz / 2.0) - (hzAcc / 2.0);
	linspace(zStart, zEnd, ref_z_li, accuracy);

	meshgrid3D(ref_x_li, ref_y_li, ref_z_li, accuracy, accuracy, accuracy, refX, refY, refZ);

	free(ref_x_li);
	free(ref_y_li);
	free(ref_z_li);

}

/**
* @brief function to get the norm of a vector
*/
double calcNorm3(double x1, double x2, double x3)
{
	double result = sqrt(x1*x1 + x2*x2 + x3*x3);
	return result;
}

/**
* @brief function to normalize a lattice vector
*/
void normalizeU(const double *U, double *normalizedU)
{
	double norm1 = calcNorm3(U[0], U[3], U[6]);
	double norm2 = calcNorm3(U[1], U[4], U[7]);
	double norm3 = calcNorm3(U[2], U[5], U[8]);

	normalizedU[0] = U[0] / norm1;
	normalizedU[1] = U[1] / norm2;
	normalizedU[2] = U[2] / norm3;

	normalizedU[3] = U[3] / norm1;
	normalizedU[4] = U[4] / norm2;
	normalizedU[5] = U[5] / norm3;

	normalizedU[6] = U[6] / norm1;
	normalizedU[7] = U[7] / norm2;
	normalizedU[8] = U[8] / norm3;
}

/**
* @brief function to calculate mod of two numbers
*/
int mod(int a, int b)
{
	if (a<0)
	{
		return a + b;
	}
	else if (a >= b)
	{
		return a - b;
	}
	else
	{
		return a;
	}
}

// /**
// * @brief function to calculate a single convolution step
// */
// void calcSingleConvolveStep(const double *image, const double stencilCoeff, const int shiftX, const int shiftY, const int shiftZ, double *result, const int imageSize, const int imageDimX, const int imageDimY, const int imageDimZ)
// {
// 	double *tempResult = calloc( imageSize, sizeof(double));
// 	multiplyScalarVector(image, stencilCoeff, tempResult, imageSize);
	
// 	int i, j, k, new_i, new_j, new_k, original_index, new_index;
// 	// double tempResult;
// 	for (k = 0; k < imageDimZ; k++){
// 		for ( j = 0; j < imageDimY; j++) {
// 			for ( i = 0; i < imageDimX; i++) {

// 				// printf("start add %d  %d  %d\n", i, j, k);
// 				original_index = k * imageDimX * imageDimY + j * imageDimX + i;
// 				new_i = mod ((i - shiftX), imageDimX);
// 				new_j = mod ((j - shiftY), imageDimY);
// 				new_k = mod ((k - shiftZ), imageDimZ);
// 				new_index = new_k * imageDimX * imageDimY + new_j * imageDimX + new_i;

// 				result[original_index] += tempResult[new_index];
// 			}
// 		}
// 	}
// }

// /**
// * @brief function to perform convolution operation
// */
// void convolve(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result)
// {
// 	int imageSize = imageDimX * imageDimY * imageDimZ;
// 	int stencilSize = stencilDimX * stencilDimY * stencilDimZ;

// 	int *xShiftList = malloc( stencilSize * sizeof(int));
// 	int *yShiftList = malloc( stencilSize * sizeof(int));
// 	int *zShiftList = malloc( stencilSize * sizeof(int));
// 	double *coeffList = malloc( stencilSize * sizeof(double));

	

// 	int i, j, k, index = 0;
// 	for (k = 0; k < stencilDimZ; k++){
// 		for ( j = 0; j < stencilDimY; j++) {
// 			for ( i = 0; i < stencilDimX; i++) {
// 				// index = k * stencilDimX * stencilDimY + j * stencilDimX + i;
// 				// ((nx+1)/2)-i, ((ny+1)/2)-j, ((nz+1)/2)-k
// 				xShiftList[index] = ((stencilDimX - 1) / 2) - i;
// 				yShiftList[index] = ((stencilDimY - 1) / 2) - j;
// 				zShiftList[index] = ((stencilDimZ - 1) / 2) - k;
// 				coeffList[index] = stencil[index];
// 				index++;

// 				// fprintf(output_fp,"%d,%d,%d,%22f\n",i,j,k,stencil[index]);
// 				// printf("%10.8f\t",stencil[index]);
// 			}
// 		}
// 	} 

// 	// printf("end ordering \n");


// 	for (i = 0; i < stencilSize; i++)
// 	{
// 		// printf("start convolve step %d \n", i);
// 		//const double *image, const double kernelCoeff, const int shiftX, const int shiftY, const int shiftZ, double *result, const int imageSize, const int kernelDimX, const int kernelDimY, const int kernelDimZ
// 		calcSingleConvolveStep(image, coeffList[i], xShiftList[i], yShiftList[i], zShiftList[i], result, imageSize, imageDimX, imageDimY, imageDimZ);
// 		//printf("after result %10f \n", result[100]);
// 		// printf("end convolve step %d \n", i);
// 	}

// 	// printf("\nafter everything %10f \n\n", result[100]);

// 	free(xShiftList);
// 	free(yShiftList);
// 	free(zShiftList);
// 	free(coeffList);

// }

// /**
// * @brief function to perform convolution operation (optimized)
// */
// void convolve2(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result)
// {
// 	int imageSize = imageDimX * imageDimY * imageDimZ;
// 	int stencilSize = stencilDimX * stencilDimY * stencilDimZ;

// 	int i, j, k, l, m, n;
// 	int stencilCenterX = (stencilDimX - 1) / 2;
// 	int stencilCenterY = (stencilDimY - 1) / 2;
// 	int stencilCenterZ = (stencilDimZ - 1) / 2;
// 	int xShift,yShift,zShift;
// 	int outIndex, outI, outJ, outK;
// 	int imageIndex = 0, stencilIndex = 0;

// 	for (k = 0; k < imageDimZ; k++){
// 		for ( j = 0; j < imageDimY; j++) {
// 			for ( i = 0; i < imageDimX; i++) {

// 				stencilIndex = 0;
// 				for (n = 0; n < stencilDimZ; n++){
// 					for ( m = 0; m < stencilDimY; m++) {
// 						for ( l = 0; l < stencilDimX; l++) {
// 							// xShift = stencilCenterX - l;
// 							// yShift = stencilCenterY - m;
// 							// zShift = stencilCenterZ - n;

// 							xShift = l - stencilCenterX;
// 							yShift = m - stencilCenterY;
// 							zShift = n - stencilCenterZ;

// 							outI = mod ((i - xShift), imageDimX);
// 							outJ = mod ((j - yShift), imageDimY);
// 							outK = mod ((k - zShift), imageDimZ);
// 							// printf("%d \t %d \t %d \t\t %d \t %d \t %d \t\t %d \t %d \t %d \t\t %d \t %d \t %d \n",i,j,k,l,m,n,xShift, yShift,zShift,outI,outJ,outK);
							
// 							outIndex = outK * imageDimX * imageDimY + outJ * imageDimX + outI;
// 							result[outIndex] += stencil[stencilIndex]* image[imageIndex];
// 							//result[outIndex] = fma(stencil[stencilIndex], image[imageIndex], result[outIndex]);
// 							stencilIndex++;

// 						}
// 					}
// 				} 

// 				imageIndex ++;
// 			}
// 		}
// 	} 
// }

// /**
// * @brief function to perform convolution operation (optimized-3)
// */
// void convolve3(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result)
// {
// 	int imageSize = imageDimX * imageDimY * imageDimZ;
// 	int stencilSize = stencilDimX * stencilDimY * stencilDimZ;

// 	int i, j, k, l, m, n;
// 	int stencilCenterX = (stencilDimX - 1) / 2;
// 	int stencilCenterY = (stencilDimY - 1) / 2;
// 	int stencilCenterZ = (stencilDimZ - 1) / 2;
// 	int xShift,yShift,zShift;
// 	int outIndex, outI, outJ, outK;
// 	int imageIndex = 0, stencilIndex = 0;

// 	for (k = 0; k < imageDimZ; k++){
// 		for ( j = 0; j < imageDimY; j++) {
// 			for ( i = 0; i < imageDimX; i++) {

// 				stencilIndex = 0;
// 				for (n = 0, zShift = - stencilCenterZ; n < stencilDimZ; n++, zShift++){
// 					outK = mod ((k - zShift), imageDimZ);
// 					for ( m = 0, yShift = - stencilCenterY; m < stencilDimY; m++, yShift++) {
// 						outJ = mod ((j - yShift), imageDimY);
// 						for ( l = 0, xShift = - stencilCenterX; l < stencilDimX; l++, xShift++) {
// 							outI = mod ((i - xShift), imageDimX);
// 							// printf("%d \t %d \t %d \t\t %d \t %d \t %d \t\t %d \t %d \t %d \t\t %d \t %d \t %d \n",i,j,k,l,m,n,xShift, yShift,zShift,outI,outJ,outK);
							
// 							outIndex = outK * imageDimX * imageDimY + outJ * imageDimX + outI;
// 							result[outIndex] += stencil[stencilIndex]* image[imageIndex];
// 							//result[outIndex] = fma(stencil[stencilIndex], image[imageIndex], result[outIndex]);
// 							stencilIndex++;
// 						}
// 					}
// 				}
// 				imageIndex ++;
// 			}
// 		}
// 	} 
// }

// /**
// * @brief function to perform convolution operation (optimized-4)
// */
// void convolve4(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result)
// {
// 	int imageSize = imageDimX * imageDimY * imageDimZ;
// 	int stencilSize = stencilDimX * stencilDimY * stencilDimZ;

// 	int i, j, k, l, m, n;
// 	int stencilCenterX = (stencilDimX - 1) / 2;
// 	int stencilCenterY = (stencilDimY - 1) / 2;
// 	int stencilCenterZ = (stencilDimZ - 1) / 2;
// 	int xShift,yShift,zShift;
// 	int outIndex, outI, outJ, outK, outKpart, outKJpart;
// 	int imageIndex = 0, stencilIndex = 0;

// 	// for( i =0; i<10; i++)
// 	// to
// 	// for(i=10; i--; )

// 	// for (k = imageDimZ; k--;){
// 	// 	for ( j = imageDimY; j--;) {
// 	// 		for ( i = imageDimX; i--;) {
// 	for (k = 0; k < imageDimZ; k++){
// 		for ( j = 0; j < imageDimY; j++) {
// 			for ( i = 0; i < imageDimX; i++) {

// 				stencilIndex = 0;
// 				for (n = stencilDimZ, zShift = - stencilCenterZ; n--; zShift++){
// 					// outK = mod ((k - zShift), imageDimZ);
// 					outKpart = mod ((k - zShift), imageDimZ) * imageDimX * imageDimY;
					
// 					for ( m = stencilDimY, yShift = - stencilCenterY; m--; yShift++) {
// 						// outJ = mod ((j - yShift), imageDimY);
// 						outKJpart = mod ((j - yShift), imageDimY) * imageDimX + outKpart;

// 						for ( l = stencilDimX,xShift = - stencilCenterX; l--; xShift++ ) {
// 							outI = mod ((i - xShift), imageDimX);
// 							// printf("%d \t %d \t %d \t\t %d \t %d \t %d \t\t %d \t %d \t %d \t\t %d \t %d \t %d \n",i,j,k,l,m,n,xShift, yShift,zShift,outI,outJ,outK);
							
// 							// outIndex = outK * imageDimX * imageDimY + outJ * imageDimX + outI;
// 							outIndex = outKJpart + outI;
// 							result[outIndex] += stencil[stencilIndex]* image[imageIndex];
// 							//result[outIndex] = fma(stencil[stencilIndex], image[imageIndex], result[outIndex]);
// 							stencilIndex++;
// 						}
// 					}
// 				}
// 				imageIndex ++;
// 			}
// 		}
// 	} 
// }

/**
* @brief function to perform convolution operation (fully optimized)
*/
void convolve5(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result)
{
	int imageSize = imageDimX * imageDimY * imageDimZ;
	int stencilSize = stencilDimX * stencilDimY * stencilDimZ;

	int i, j, k, l, m, n;
	int stencilCenterX = (stencilDimX - 1) / 2;
	int stencilCenterY = (stencilDimY - 1) / 2;
	int stencilCenterZ = (stencilDimZ - 1) / 2;
	int xShift,yShift,zShift;
	int outIndex, outI, outJ, outK;
	int imageXYDim = imageDimX * imageDimY, outKpart, outJKpart;
	int outputIndex = 0, imageIndex = 0, stencilIndex = 0;
	double currentPixelValue;

	for (k = 0; k < imageDimZ; k++){
		for ( j = 0; j < imageDimY; j++) {
			for ( i = 0; i < imageDimX; i++) {
				currentPixelValue = 0.0;

				stencilIndex = stencilSize - 1;
				for (n = stencilDimZ, zShift = - stencilCenterZ; n--; zShift++){
					outK = mod ((k - zShift), imageDimZ);
					outKpart = outK * imageXYDim;
					
					for ( m = stencilDimY, yShift = - stencilCenterY; m--; yShift++) {
						outJ = mod ((j - yShift), imageDimY);
						outJKpart = outJ * imageDimX + outKpart;
						
						for ( l = stencilDimX, xShift = - stencilCenterX; l--; xShift++ ) {
							outI = mod ((i - xShift), imageDimX);
							
							imageIndex = outJKpart + outI;
							
							currentPixelValue += stencil[stencilIndex]* image[imageIndex]; // 5.64s

							stencilIndex--;
						}
					}
				}

				result[outputIndex] = currentPixelValue;
				outputIndex ++;
			}
		}
	} 
}

/**
 * @brief Calculate the covolution of an image and a stencil.
 * 
 * @param image Image array (3D unrolled into 1D).
 * @param stencil Convolution stencil/kernel (3D unrolled into 1D).
 * @param imageDimX Dimensions of the image in the x direction.
 * @param imageDimY Dimensions of the image in the y direction.
 * @param imageDimZ Dimensions of the image in the z direction.
 * @param stencilDimX Dimensions of the stencil/kernel in the x direction.
 * @param stencilDimY Dimensions of the stencil/kernel in the y direction.
 * @param stencilDimZ Dimensions of the stencil/kernel in the z direction.
 * @param result The result array (3D unrolled into 1D, note that it's not
 *               the same shape as image, but a subgrid of the global array).
 * @param DMVerts The starting and ending indices of the subgrid assigned to
 *                the current process. The order is [is, ie, js, je, ks, ke].
 */
void convolve6(
	const double *image, const double *stencil, const int imageDimX,
	const int imageDimY, const int imageDimZ, const int stencilDimX,
	const int stencilDimY, const int stencilDimZ, double *result,
	int DMVerts[6])
{
	// subgrid starting and ending indices
	int is = DMVerts[0];
	int ie = DMVerts[1];
	int js = DMVerts[2];
	int je = DMVerts[3];
	int ks = DMVerts[4];
	int ke = DMVerts[5];
	int imageSize = imageDimX * imageDimY * imageDimZ;
	int stencilSize = stencilDimX * stencilDimY * stencilDimZ;
	int i, j, k, l, m, n;
	int stencilCenterX = (stencilDimX - 1) / 2;
	int stencilCenterY = (stencilDimY - 1) / 2;
	int stencilCenterZ = (stencilDimZ - 1) / 2;
	int xShift,yShift,zShift;
	int outIndex, outI, outJ, outK;
	int imageXYDim = imageDimX * imageDimY, outKpart, outJKpart;
	int outputIndex = 0, imageIndex = 0, stencilIndex = 0;
	double currentPixelValue;
	
	for (k = ks; k <= ke; k++) {
		for ( j = js; j <= je; j++) {
			for ( i = is; i <= ie; i++) {
	
				currentPixelValue = 0.0;

				stencilIndex = stencilSize - 1; // starting from the farthest point in stencil

				for (n = stencilDimZ, zShift = - stencilCenterZ; n--; zShift++){
				
					outK = mod ((k - zShift), imageDimZ);
					outKpart = outK * imageXYDim;
					
					for ( m = stencilDimY, yShift = - stencilCenterY; m--; yShift++) {
					
						outJ = mod ((j - yShift), imageDimY);
						outJKpart = outJ * imageDimX + outKpart;
						
						for ( l = stencilDimX,xShift = - stencilCenterX; l--; xShift++ ) {
							outI = mod ((i - xShift), imageDimX);

							imageIndex = outJKpart + outI;
						
							currentPixelValue += stencil[stencilIndex]* image[imageIndex];

							stencilIndex--;
						}
					}
				}

				result[outputIndex] = currentPixelValue;
				outputIndex ++;
			}
		}
	} 
}

/**
* @brief function to write the first row of the HSMP feature output file
*/
void writeHSMPFirstRowToFile(const char *filename, const int MCSHMaxOrder, const double rMaxCutoff, const double rStepsize){
	FILE *output_fp = fopen(filename, "w");
	if (output_fp == NULL){
		printf("\nCannot open file \"%s\"n", filename);
		die("cannot open file");
	}
	fprintf(output_fp, "i,j,k");
	int l, j;
	int numRCutoff = getNumRCutoff(rStepsize, rMaxCutoff);
	printf("R step size: %f, rcut off : %f, Num R cutoff : %d \n", rStepsize, rMaxCutoff, numRCutoff);
	for (l = 0; l <= MCSHMaxOrder; l++){
		for(j = 0; j < numRCutoff; j++){
			printf("l = %d, RCut = %f \n", l, (j+1)*rStepsize);
			fprintf(output_fp, ",l=%d::Rcut=%f",l, (j+1)*rStepsize);
		}
	}
	fprintf(output_fp, "\n");
	fclose(output_fp);
}

/**
* @brief function to write the first row of the LPMP feature output file
*/
void writeLPMPFirstRowToFile(const char *filename, const int MCSHMaxOrder, const int MCSHRadialMaxOrder)
{
	FILE *output_fp = fopen(filename, "w");
	if (output_fp == NULL){
		printf("\nCannot open file \"%s\n", filename);
		die("cannot open file");	
	}
	fprintf(output_fp, "i,j,k");
	
	int l, lp;
	
	for (l = 0; l <= MCSHMaxOrder; l++){
		for (lp = 0; lp <= MCSHRadialMaxOrder; lp++){
			fprintf(output_fp, ",l=%d::lp=%d", l, lp);
		}
	}
	fprintf(output_fp, "\n");
	fclose(output_fp);		
}

/**
* @brief function to write a matrix of features to a file
*/
void writeFeatureMatrixToFile(const char *filename, double *featureMatrixGlobal, const int length, const int dimX, const int dimY, const int dimZ){
	printf("Writing descriptors to file \n");
	FILE *output_fp = fopen(filename, "a");
	if (output_fp == NULL) {
		printf("\nCannot open file \"%s\"\n",filename);
		die("cannot open file");
	}
	fprintf(output_fp, "\n");
	int i, j, k, l, index = 0;
	int imageSize = dimX * dimY * dimZ;
	for (k = 0 ; k < dimZ; k++){
		for (j = 0; j < dimY; j++){
			for (i = 0; i < dimX; i++){
				fprintf(output_fp, "%d,%d,%d",i,j,k);
				index = k * dimX * dimY + j * dimX + i;
				for (l = 0; l < length; l++){
					fprintf(output_fp, ",%.15f",featureMatrixGlobal[l*imageSize + index]);
				}
				fprintf(output_fp, "\n");
			}
		}
	}
	fclose(output_fp);
}

/**
* @brief function to write a matrix to a file
*/
void writeMatToFile(const char *filename, const double *data, const int dimX, const int dimY, const int dimZ)// at the end of SCF cycle
{
	
	FILE *output_fp = fopen(filename,"w");
	if (output_fp == NULL) {
		printf("\nCannot open file \"%s\"\n",filename);
		die("cannot open file");
	}

	int i,j,k, index = 0;
	for (k = 0; k < dimZ; k++){
		for ( j = 0; j < dimY; j++) {
			for ( i = 0; i < dimX; i++) {
				fprintf(output_fp,"%d,%d,%d,%.15f\n",i,j,k,data[index]);
				index ++;
			}
		}
	}


	fclose(output_fp);
}

/**
* @brief function to get main parameters for LPMP features
*/
void getMainParameter_RadialLegendre(MULTIPOLE_OBJ *mp, int* LegendreOrderList, int* lList)
{
	int numLegendre = mp->MCSHMaxRadialOrder + 1;

	int i, j, index = 0;

	for (i = 0; i < mp->MCSHMaxOrder+1; i++)
	{
		for (j = 0; j < numLegendre; j++)
		{
			LegendreOrderList[index] = j;
			lList[index] = getCurrentLNumber(i);
			//groupList[index] = getCurrentGroupNumber(i);
			index++;
		}
	}
}

/**
* @brief function to get the descriptor list length for LPMP descriptors
*/
int getDescriptorListLength_RadialLegendre(MULTIPOLE_OBJ *mp)
{

	//int numGroup = getNumGroup(maxMCSHOrder);
	// printf("\nnumber groups:%d \n", numGroup);
	int numMCSH = mp->MCSHMaxOrder + 1;
	int numLegendre = mp->MCSHMaxRadialOrder + 1;
	// printf("\nnumber r cutoff:%d \n", numRCutoff);
	
	return numMCSH * numLegendre;

}

/**
* @brief function to get main parameters for HSMP features
*/
void getMainParameter_RadialRStep(MULTIPOLE_OBJ *mp, double* rCutoffList, int* lList)
{
	
	int numRCutoff = getNumRCutoff(mp->MCSHRStepSize, mp->MCSHMaxR);

	int i, j, index = 0;

	for (i = 0; i < mp->MCSHMaxOrder + 1; i++)
	{
		for (j = 0; j < numRCutoff; j++)
		{
			rCutoffList[index] = (j+1) * mp->MCSHRStepSize;
			lList[index] = getCurrentLNumber(i);
			index++;
		}
	}
}

/**
* @brief function to get the descriptor list length for HSMP descriptors
*/
int getDescriptorListLength_RadialRStep(MULTIPOLE_OBJ *mp)
{
	int numMCSH = mp->MCSHMaxOrder + 1;
	int numRCutoff = getNumRCutoff(mp->MCSHRStepSize, mp->MCSHMaxR);

	return numMCSH * numRCutoff;

}

/**
* @brief function to get the number of r cutoffs for a given r step size and r max cutoff
*/
int getNumRCutoff(const double rStepsize, const double rMaxCutoff)
{
	return (int) ceil(rMaxCutoff / rStepsize);
}

/**
* @brief function to get the number of groups for a given MCSH order
*/
int getNumGroup(const int maxMCSHOrder)
{
	int numGroupList[5] = {1,1,2,3,4};
	if (maxMCSHOrder > 4) die("\n Error: Maximum Order Not Implemented \n");

	int numGroup=0;
	int i;
	for (i = 0; i < maxMCSHOrder + 1; i++)
	{
		numGroup += numGroupList[i];
	}
	return numGroup;
}

/**
* @brief function to get the current group number for a given MCSH order
*/
int getCurrentGroupNumber(const int currentIndex)
{	
	// 1 --> 1
	// 2 --> 1
	// 3 --> 1
	// 4 --> 2
	// 5 --> 1
	// 6 --> 2
	// 7 --> 3
	// 8 --> 1
	// 9 --> 2
	// 10 --> 3
	// 11 --> 4


	int resultList[11] = {1,1,1,2,1,2,3,1,2,3,4};
	return resultList[currentIndex];
}

/**
* @brief function to get the current L number which is the MCSH order
*/
int getCurrentLNumber(const int currentIndex)
{	
	int resultList[11] = {0,1,2,3,4,5,6,7,8,9};
	return resultList[currentIndex];
}
