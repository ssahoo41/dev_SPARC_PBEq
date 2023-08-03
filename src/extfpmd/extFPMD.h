/**
 * @file    extFPMD.h
 * @brief   This file contains the function declarations for the extended
 *          First Principle Molecular Dynamics method.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2022 Material Physics & Mechanics Group, Georgia Tech.
 */

#ifndef EXTFPMD_H
#define EXTFPMD_H 

#include "isddft.h"

/**
 * @brief Calculate the constant effective energy shift U0.
 *
 *        The top eigenvalues of the Hamiltonian should match
 *        the top eigenvalues of the Kinetic operator with a
 *        shift (-1/2*Laplacian + U0). In other words, the top
 *        eigenvalues of H also satisfy the following equation:
 *
 *        H ~= (-1/2 * Laplacian + U0) \psi_i = \lambda_i \psi_i,
 *
 *        where U0 is a constant shift. This is equivalent to
 *
 *          <\psi_i|-1/2*Laplacian|\psi_i> + U0 = \lambda_i.
 *
 *        Since there's only one unkown (i.e., U0), but many
 *        equations. There are different ways to find U0.
 *
 *        In this routine, we find U0 by averaging over a number
 *        of top energy bands (i = Nc_s,...,Nc_e):
 *        U0 = mean(\lambda_i - <\psi_i|-1/2*Laplacian|\psi_i>).
 *
 * @param pSPARC
 * @param Nc_s Start band index (global, 0-based).
 * @param Nc_e End band index (global, 0-based).
 * @return double
 */
double calculate_effective_energy_shift_U0(
	SPARC_OBJ *pSPARC, const int Nc_s, const int Nc_e);

/**
 * @brief Density Of States (DOS) formula for a high-Energy electrons. Can be
 *        represented by free-electron models with a constant potential U0.
 *
 * @param lambda Energy level.
 * @param data_info All data other than the energy level are stored here.
 *                  In this function, we assume two double type parameters
 *                  are stored in Data_info->double_data = [Volume, U0].
 * @return double
 */
double high_E_DOS(double lambda, DATA_INFO *data_info);


/**
 * @brief Occupation times the Density of States for high-Energy electrons.
 * 
 *        Occupation is usually given by the Fermi-Dirac function.
 *        High-energy DOS is given by the free electron DOS. Returns
 *                 func(lambda) = g(lambda) * DOS(lambda).
 * 
 * @param lambda Energy level.
 * @param data All data other than the energy level are stored here.
 *             In this function, we assume the double_data contains the
 *             following data:
 *             [c, U0, smearing_type, beta, lambda_f],
 *             where c = sqrt(2)*Volume/pi^2.
 * @return double 
 */
double occ_times_DOS(double lambda, DATA_INFO *data);


/**
 * @brief   Occupation constraint provided lambda_f when using extFPMD method.
 */
double occ_constraint_extFPMD(SPARC_OBJ *pSPARC, double lambda_f);


/**
 * @brief   Find the charge density of the high energy electrons provided
 *          lambda_f when using the ext-FPMD method. 
 * 
 *          This routine finds
 *          
 *          rho(x) = beta * rho(x) + 1/V * \int_{Ec}^{Emax} f(x) D(x) dx,
 *          
 *          where f(x) is the smearing function (with given smearing and
 *          fermi-level), D(x) is the the density of state (DOS) of high
 *          energy electrons, which is expressed as
 *              D(x) = sqrt(2)*V/pi^2 sqrt(x - U0).
 * 
 * @param pSPARC SPARC object.
 * @param beta Scalar.
 * @param rho Electron density (INPUT/OUTPUT).
 * @param DMnd Number of grid points in the current process.
 */
void highE_rho_extFPMD(SPARC_OBJ *pSPARC, double beta, double *rho, int DMnd);


/**
 * @brief Calculate the total charge of high-energy electrons.
 * 
 * @param pSPARC SPARC object.
 * @param lambda_f Fermi level.
 * @return double Total kinetic energy of high-energy electrons.
 */
double calculate_highE_Charge_extFPMD(SPARC_OBJ *pSPARC, double lambda_f);


/**
 * @brief Calculate the kinetic energy of high-energy electrons.
 * 
 * @param pSPARC SPARC object.
 * @param lambda_f Fermi level.
 * @return double Total kinetic energy of high-energy electrons.
 */
double calculate_highE_Tk_extFPMD(SPARC_OBJ *pSPARC, double lambda_f);



/**
 * @brief Calculate the electronic entropy of high-energy electrons.
 * 
 * @param pSPARC SPARC object.
 * @param lambda_f Fermi level.
 * @return double Total kinetic energy of high-energy electrons.
 */
double calculate_highE_Entropy_extFPMD(SPARC_OBJ *pSPARC, double lambda_f);

#endif // EXTFPMD_H

