/**
* @file MCSHMainCalc.h
* @brief This file sets up parallelization and calls the functions main functions to calculate the
*        Heaviside Multipole (HSMP) or Legendre Polynomial Multipole (LPMP) descriptors.
*
* @author Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
*		Andrew J. Medford <ajm@gatech.edu>
*
* Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
*/

/**
 * @brief   function to calculate grid-based multipole features from electron density after SCF calculation
 */
void Calculate_MCSHDescriptors(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp,  const int iterNum);

/**
* @brief function to initialize structure for multipole feature calculation
*/
void Multipole_Initialize(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp);

/**
 * @brief   function to calculate HSMP descriptors
 */
void CalculateHSMPDescriptors(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp, const int iterNum, double *elecDens, const int commIndex, const int numParallelComm,
                             const MPI_Comm communicator, int DMVerts[6], const int nFeatures);

/**
 * @brief   function to calculate LPMP descriptors
 */
void CalculateLPMPDescriptors(SPARC_OBJ *pSPARC, MULTIPOLE_OBJ *mp, const int iterNum, double *elecDens, const int commIndex, 
                                const int numParallelComm, const MPI_Comm communicator, int DMVerts[6], const int nFeatures);

