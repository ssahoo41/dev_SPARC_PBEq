/**
* @file MCSHHelper.h
* @brief This file contains the declaration of MCSH functions.
*
* @author Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
*		Andrew J. Medford <ajm@gatech.edu>
*
* Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
*/
#include "MP_types.h"

typedef struct Point {
   double x;
   double y;
   double z;
} Point;

/**
* @brief function to print the error message and exit the program
*/
void die(const char *message);

/**
* @brief function to get the plane equation formed by three points
*/
void GetPlaneEquation(const Point p1, const Point p2, const Point p3, double *a, double *b, double *c, double *d);

/**
* @brief function to check if the plane intersects with the sphere
*/
int CheckPlaneIntersectWithSphere(const Point p1, const Point p2, const Point p3, const double rCutoff, const Point origin);

/**
* @brief function to transform the point using the transformation matrix
*/
Point UTransform(const double x, const double y, const double z, const double *U);

/**
* @brief function to get the dimensions of the plane
*/
void GetDimensionsPlane(MULTIPOLE_OBJ *mp, const double *U, int *dimXResult, int *dimYResult, int *dimZResult);

/**
* @brief function to print the array
*/
void printArr(double *arr, int size);

/**
* @brief function to evaluate square root of a vector elementwise
*/
void sqrtVector(const double *arr, double *result, const int size);

/**
* @brief function to evaluate power of a vector elementwise
*/
void powVector(const double *arr, const int power, double *result, const int size);

/**
* @brief function to add two vectors elementwise
*/
void addVector(const double *arr1, const double *arr2, double *result, const int size);

/**
* @brief function to subtract two vectors elementwise
*/
void subtractVector(const double *arr1, const double *arr2, double *result, const int size);

/**
* @brief function to multiply two vectors elementwise
*/
void multiplyVector(const double *arr1, const double *arr2, double *result, const int size);

/**
* @brief function to divide two vectors elementwise
*/
void divideVector(const double *arr1, const double *arr2, double *result, const int size);

/**
* @brief function to add a scaler to every element of a vector
*/
void addScalarVector(const double *arr, const double a, double *result, const int size);

/**
* @brief function to multiply a scaler to every element of a vector
*/
void multiplyScalarVector(const double *arr1, const double a, double *result, const int size);

/**
* @brief function to multiply a scaler to every element of a vector and add
*		 it to the resulting vector
*/
void multiplyScalarVector_add(const double *arr1, const double a, double *result, const int size);

/**
* @brief function to normalize position vectors
*/
void getRArray(const double *x, const double *y, const double *z, double *result, const int size);

/**
* @brief function to evaluate a polynomial in a single variable
*/
void polyArray(const double *x, const int powX, const double a, double *result, const int size);

/**
* @brief function to evaluate a polynomial in a single variable and add to the result
*/
void polyArray_add(const double *x, const int powX, const double a, double *result, const int size);

/**
* @brief function to evaluate a polynomial in three variables
*/
void polyXYZArray(const double *x, const double *y, const double *z, const int powX, const int powY, const int powZ, const double a, double *result, const int size);

/**
* @brief function to evaluate a polynomial in three variables and add to the result
*/
void polyXYZArray_add(const double *x, const double *y, const double *z, const int powX, const int powY, const int powZ, const double a, double *result, const int size);

/**
* @brief function to apply a transformation matrix to the position vectors
*/
void applyU(double *X, double *Y, double*Z, const double *U, const int size);

/**
* @brief function to apply a transformation matrix to the position vectors
*/
void applyU2(double *X, double *Y, double*Z, const double *U, const int size);

/**
* @brief function to calculate volume of a voxel
*/
double calcDv(const double hx, const double hy, const double hz, const int accuracy, const double *U);

/**
* @brief function to calculate sum of all elements of an array
*/
double sumArr(const double *arr, const int size);

/**
* @brief function to calculate sum of absolute value of all elements of an array
*/
double sumAbsArr(const double *arr, const int size);

/**
* @brief function to create an array of linearly spaced numbers
*/
void linspace(double start, double end, double *result, int num);

/**
* @brief function to create a 3D meshgrid
*/
void meshgrid3D(const double *x, const double *y, const double *z, const int sizex, const int sizey, const int sizez, double *X, double *Y, double *Z);

/**
* @brief function to get the reference coordinates of the voxel
*/
void getCentralCoords(const double hx, const double hy, const double hz, const int accuracy, double *refX, double *refY, double *refZ);

/**
* @brief function to get the norm of a vector
*/
double calcNorm3(double x1, double x2, double x3);

/**
* @brief function to normalize a lattice vector
*/
void normalizeU(const double *U, double *normalizedU);

// /**
// * @brief function to calculate a single convolution step
// */
// void calcSingleConvolveStep(const double *image, const double stencilCoeff, const int shiftX, const int shiftY, const int shiftZ, double *result, const int imageSize, const int imageDimX, const int imageDimY, const int imageDimZ);

// /**
// * @brief function to perform convolution operation
// */
// void convolve(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result);

// /**
// * @brief function to perform convolution operation (optimized)
// */
// void convolve2(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result);

// /**
// * @brief function to perform convolution operation (optimized-3)
// */
// void convolve3(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result);

// /**
// * @brief function to perform convolution operation (optimized-4)
// */
// void convolve4(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result);

/**
* @brief function to perform convolution operation (fully optimized)
*/
void convolve5(const double *image, const double *stencil, const int imageDimX, const int imageDimY, const int imageDimZ, const int stencilDimX, const int stencilDimY, const int stencilDimZ, double *result);

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
	int DMVerts[6]);

/**
* @brief function to write the first row of the HSMP feature output file
*/
void writeHSMPFirstRowToFile(const char *filename, const int MCSHMaxOrder, const double rMaxCutoff, const double rStepsize);

/**
* @brief function to write the first row of the LPMP feature output file
*/
void writeLPMPFirstRowToFile(const char *filename, const int MCSHMaxOrder, const int MCSHRadialMaxOrder);

/**
* @brief function to write a matrix to a file
*/
void writeMatToFile(const char *filename, const double *data, const int dimX, const int dimY, const int dimZ);

/**
* @brief function to write a matrix of features to a file
*/
void writeFeatureMatrixToFile(const char *filename, double *featureMatrixGlobal, const int length, const int dimX, const int dimY, const int dimZ);

/**
* @brief function to calculate mod of two numbers
*/
int mod(int a, int b);

/**
* @brief function to get main parameters for LPMP features
*/
void getMainParameter_RadialLegendre(MULTIPOLE_OBJ *mp, int* LegendreOrderList, int* lList);

/**
* @brief function to get the descriptor list length for LPMP descriptors
*/
int getDescriptorListLength_RadialLegendre(MULTIPOLE_OBJ *mp);

/**
* @brief function to get main parameters for HSMP features
*/
void getMainParameter_RadialRStep(MULTIPOLE_OBJ *mp, double* rCutoffList, int* lList);

/**
* @brief function to get the descriptor list length for HSMP descriptors
*/
int getDescriptorListLength_RadialRStep(MULTIPOLE_OBJ *mp);

/**
* @brief function to get the number of r cutoffs for a given r step size and r max cutoff
*/
int getNumRCutoff(const double rStepsize, const double rMaxCutoff);

/**
* @brief function to get the number of groups for a given MCSH order
*/
int getNumGroup(const int maxOrder);

/**
* @brief function to get the current group number for a given MCSH order
*/
int getCurrentGroupNumber(const int currentIndex);

/**
* @brief function to get the current L number which is the MCSH order
*/
int getCurrentLNumber(const int currentIndex);