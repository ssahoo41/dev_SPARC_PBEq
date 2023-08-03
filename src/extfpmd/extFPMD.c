/**
 * @file    extFPMD.c
 * @brief   This file contains functions for the extended First Principle
 *          Molecular Dynamics method.
 *
 * @authors Qimen Xu <qimenxu@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 *
 * Copyright (c) 2022 Material Physics & Mechanics Group, Georgia Tech.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
/* BLAS and LAPACK routines */
#ifdef USE_MKL
	#include <mkl.h>
#else
	#include <cblas.h>
	#include <lapacke.h>
#endif
/* ScaLAPACK routines */
#ifdef USE_MKL
	#include "blacs.h"     // Cblacs_*
	#include <mkl_blacs.h>
	#include <mkl_pblas.h>
	#include <mkl_scalapack.h>
#endif
#ifdef USE_SCALAPACK
	#include "blacs.h"     // Cblacs_*
	#include "scalapack.h" // ScaLAPACK functions
#endif

#include "tools.h"
#include "lapVecRoutines.h"
#include "occupation.h"
#include "isddft.h"
#include "extFPMD.h"

#define TEMP_TOL 1e-12

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


/**
 * @brief Find kinetic energy density of a single band psi_n.
 *        Ek_n = <psi_n |-1/2*Lap| psi_n>.
 *
 * @param pSPARC SPARC object.
 * @param psi_n Kohn-sham orbital.
 * @param DMnd Number of grid points in the current block of distributed domain.
 * @param DMVertices Domain vertices of the corners [xs,xe,ys,ye,zs,ze].
 * @param comm Communicator over which psi_n is distributed.
 * @return double
 */
double band_kinetic_energy(
	const SPARC_OBJ *pSPARC, double *psi_n, const int DMnd,
	const int *DMVertices, MPI_Comm comm)
{
	double *D2_psi_n = malloc(DMnd * sizeof(*D2_psi_n));
	assert(D2_psi_n != NULL);

	// find Lap
	Lap_vec_mult(pSPARC, DMnd, DMVertices, 1, 0.0, psi_n, D2_psi_n, comm);

	double Ek_n_loc = 0.0;
	for (int i = 0; i < DMnd; i++) {
		Ek_n_loc += -0.5 * psi_n[i] * D2_psi_n[i];
	}

	double Ek_n = Ek_n_loc;

	int size_comm;
	MPI_Comm_size(comm, &size_comm);
	if (size_comm > 1)
		MPI_Allreduce(&Ek_n_loc, &Ek_n, 1, MPI_DOUBLE, MPI_SUM, comm);

	free(D2_psi_n);
	return Ek_n;
}



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
	SPARC_OBJ *pSPARC, const int Nc_s, const int Nc_e)
{
	#ifdef DEBUG
	double t1,t2;
	t1 = MPI_Wtime();
	#endif

	double U0 = 0.0;
	// int nbnd_cnt = 0; // number of local bands counted in the average
	int nstart = pSPARC->band_start_indx;
	int nend = pSPARC->band_end_indx;
	int DMnd = pSPARC->Nd_d_dmcomm;
	int *DMVertices = pSPARC->DMVertices_dmcomm;

	double sum_loc = 0.0, count_loc = 0.0, sum = 0.0;
	for (int n = nstart; n <= nend; n++) {
		if (n < Nc_s || n > Nc_e) continue; // only count the specified bands
		double *psi_n = pSPARC->Xorb + DMnd*(n-nstart);
		double lambda_n = pSPARC->lambda[n];
		double Ek_n = band_kinetic_energy(pSPARC, psi_n, DMnd, DMVertices, pSPARC->dmcomm);
		sum_loc += (lambda_n - Ek_n);
		count_loc += 1.0;
	}

	int size_blacscomm;
	MPI_Comm_size(pSPARC->blacscomm, &size_blacscomm);

	// sum over all bandcomms
	MPI_Allreduce(&sum_loc, &sum, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);

	#ifdef DEBUG_BANDCOUNT
	// for debugging only, check the total number of bands counted
	double count = 0.0;
	MPI_Allreduce(&count_loc, &count, 1, MPI_DOUBLE, MPI_SUM, pSPARC->blacscomm);
	assert(fabs(count - (Nc_e-Nc_s+1.0)) < 1e-12);
	#endif

	U0 = sum / (Nc_e - Nc_s + 1.0);

	// broadcast the value
	// TODO: if there're k-pionts, average over kpoints before bcast
	MPI_Bcast(&U0, 1, MPI_DOUBLE, 0, pSPARC->kptcomm);

	#ifdef DEBUG
	t2 = MPI_Wtime();
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		printf(RED "\n U0 = %.16f, time taken: %.3f ms\n" RESET, U0, (t2-t1)*1e3);
	}
	#endif

	return U0;
}


/**
 * @brief Density Of States (DOS) formula for a free electron in a constant
 *        potential field V = U0.
 *           DOS(lambda) = sqrt(2) * V / pi^2 sqrt(lambda - U0)
 *
 * @param Volume
 * @param U0
 * @param lambda
 * @return double
 */
double free_electron_DOS(const double Volume, const double U0, const double lambda)
{
	return Volume / (M_PI * M_PI) * sqrt(2.0 * (lambda - U0));
}


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
double high_E_DOS_v0(double lambda, DATA_INFO *data_info)
{
	double Volume = data_info->double_data[0];
	double U0 = data_info->double_data[1];
	return free_electron_DOS(Volume, U0, lambda);
}


/**
 * @brief Density Of States (DOS) formula for a high-Energy electrons. Can be
 *        represented by free-electron models with a constant potential U0.
 *
 *        Note that this routine gives the same as _v0 version, except it
 *        requires fewer flops.
 *
 * @param lambda Energy level.
 * @param data_info All data other than the energy level are stored here.
 *                  In this function, we assume two double type parameters
 *                  are stored in Data_info->double_data[0] = sqrt(2)*V/pi^2.
 * @return double
 */
double high_E_DOS(double lambda, DATA_INFO *data_info)
{
	double c = data_info->double_data[0];
	double U0 = data_info->double_data[1];
	return c * sqrt(lambda - U0);
}


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
double occ_times_DOS(double lambda, DATA_INFO *data)
{
	double DOS = high_E_DOS(lambda, data);
	int smearing_type = (int) data->double_data[2];
	double beta = data->double_data[3];
	double lambda_f = data->double_data[4];
	double g = smearing_function(beta, lambda, lambda_f, smearing_type);
	return DOS * g;
}


/**
 * @brief Occupation times the Density of States times kinectic energy
 *        for high-Energy electrons.
 * 
 *        Occupation is usually given by the Fermi-Dirac function.
 *        High-energy DOS is given by the free electron DOS. Returns
 *          func(lambda) = g(lambda) * DOS(lambda) * (lambda - U0).
 * 
 * @param lambda Energy level.
 * @param data All data other than the energy level are stored here.
 *             In this function, we assume the double_data contains the
 *             following data:
 *             [c, U0, smearing_type, beta, lambda_f],
 *             where c = sqrt(2)*Volume/pi^2.
 * @return double 
 */
double occ_times_DOS_times_Ek(double lambda, DATA_INFO *data)
{
	// double DOS = high_E_DOS(lambda, data);
	// int smearing_type = (int) data->double_data[2];
	// double beta = data->double_data[3];
	// double lambda_f = data->double_data[4];
	// double g = smearing_function(beta, lambda, lambda_f, smearing_type);
	// double U0 = data->double_data[1];
	// return DOS * g * (lambda - U0);
	double U0 = data->double_data[1];
	return occ_times_DOS(lambda, data) * (lambda - U0);
}


/**
 * @brief   Find the charge of the high energy electrons provided
 *          lambda_f when using the ext-FPMD method.
 * 
 *          This routine finds
 *              HighECharge = \int_{Ec}^{Emax} f(x) D(x) dx,
 *          where f(x) is the smearing function (with given smearing and
 *          fermi-level), D(x) is the the density of state (DOS) of high
 *          energy electrons, which is expressed as
 *              D(x) = sqrt(2)*V/pi^2 sqrt(x - U0).
 */
double high_E_Charge(
	int elec_T_type, double Beta, double lambda_f, double V, double U0,
	double Ec, double Emax, double mesh, MPI_Comm comm)
{
	// set up data for the integral of occ * DOS for high energy electrons
	DATA_INFO extraData;
	extraData.len_double = 5;
	extraData.double_data = malloc(extraData.len_double * sizeof(double));
	assert(extraData.double_data != NULL);
	extraData.double_data[0] = sqrt(2.0)*V/(M_PI*M_PI); // sqrt(2)*V/pi^2
	extraData.double_data[1] = U0;
	extraData.double_data[2] = (double) elec_T_type;
	extraData.double_data[3] = Beta;
	extraData.double_data[4] = lambda_f;

	size_t N_int = ceil((Emax - Ec) / mesh);
	// next find the high energy charges
	double HighECharge = integral_simpson_paral(
		Ec, Emax, N_int, occ_times_DOS, &extraData, comm);

	free(extraData.double_data);
	return HighECharge;
}



/**
 * @brief   Find the kinetic energy of the high energy electrons provided
 *          lambda_f when using the ext-FPMD method.
 * 
 *          This routine finds
 *              HighECharge = \int_{Ec}^{Emax} f(x) D(x) (x-U0) dx,
 *          where f(x) is the smearing function (with given smearing and
 *          fermi-level), D(x) is the the density of state (DOS) of high
 *          energy electrons, which is expressed as
 *              D(x) = sqrt(2)*V/pi^2 sqrt(x - U0),
 *          and (x-U0) is the kinetic energy of high energy electrons.
 */
double high_E_Tk(
	int elec_T_type, double Beta, double lambda_f, double V, double U0,
	double Ec, double Emax, double mesh, MPI_Comm comm)
{
	// set up data for the integral of occ * DOS for high energy electrons
	DATA_INFO extraData;
	extraData.len_double = 5;
	extraData.double_data = malloc(extraData.len_double * sizeof(double));
	assert(extraData.double_data != NULL);
	extraData.double_data[0] = sqrt(2.0)*V/(M_PI*M_PI); // sqrt(2)*V/pi^2
	extraData.double_data[1] = U0;
	extraData.double_data[2] = (double) elec_T_type;
	extraData.double_data[3] = Beta;
	extraData.double_data[4] = lambda_f;

	size_t N_int = ceil((Emax - Ec) / mesh);
	// next find the high energy charges
	double HighE_Tk = integral_simpson_paral(
		Ec, Emax, N_int, occ_times_DOS_times_Ek, &extraData, comm);

	free(extraData.double_data);
	return HighE_Tk;
}


/**
 * @brief Find max energy (inf) for integration over energy.
 * 
 * @param pSPARC SPARC object.
 * @param lambda_f Fermi level.
 * @param E_start Lower bound of the integral over energy.
 * @return double 
 */
double find_Emax(SPARC_OBJ *pSPARC, double lambda_f, double E_start)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// max eigval for 1D orthogonal -1.0*Lap for h_eff = 1.0
	// const double lambda_ref = 6.8761754299116333;
	// double Emax = 3.0*lambda_ref/ (0.1*0.1);
	double Emax = pSPARC->MaxEigVal_mhalfLap;
	//? Another way is to make sure occ < tol
	//?     Emax = lambda_f + 1/(k_B*T) * ln(1/tol)
	//?          = lambda_f + beta * ln(1/tol)
	//? Let tol = 1e-16, ln(tol) = 36.84.
	double occ_tol = 1e-16;
	double E_unocc = lambda_f + pSPARC->Beta * (-log(occ_tol));
	// if (rank == 0 || 1) {
	// 	printf("rank = %2d, occ_tol = %.2e, lambda_f = %f, E_unocc = %f\n", rank, occ_tol, lambda_f, E_unocc);
	// }

	Emax = min(Emax, E_unocc);
	Emax = max(Emax, E_start + 20.0);
	return Emax;
}


double calculate_highE_Charge_extFPMD(SPARC_OBJ *pSPARC, double lambda_f)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// #ifdef DEBUG
	// 	if (rank == 0) printf("Start evaluating total charge of high-energy electrons\n");
	// #endif
	double V = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z * pSPARC->Jacbdet;
	double U0 = pSPARC->ext_FPMD_U0;
	double Ec = pSPARC->lambda_sorted[pSPARC->Nstates-1];
	double Emax = find_Emax(pSPARC, lambda_f, Ec);
	double dE = 0.1 / 27.21138; // 0.1 eV converted into Hartree
	double HighECharge = high_E_Charge(
		pSPARC->elec_T_type, pSPARC->Beta, lambda_f, V, U0,
		Ec, Emax, dE, pSPARC->kptcomm_active);
	// #ifdef DEBUG
	// if (rank == 0) {
	// 	printf("rank = %2d, Ec = %f, Emax = %f, dE = %e, U0 = %f, HighECharge = %f\n",
	// 		rank, Ec, Emax, dE, U0, HighECharge);
	// }
	// #endif // DEBUG

	return HighECharge;
}



/**
 * @brief   Occupation constraint provided lambda_f when using extFPMD method.
 */
double occ_constraint_extFPMD(SPARC_OBJ *pSPARC, double lambda_f)
{
	// first get the usual part of the constraint sum_{i=1}^Ns g_i + NegativeCharge
	double NetCharge = occ_constraint(pSPARC, lambda_f);
	double HighECharge = calculate_highE_Charge_extFPMD(pSPARC, lambda_f);
	return NetCharge + HighECharge;
}



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
void highE_rho_extFPMD(SPARC_OBJ *pSPARC, double beta, double *rho, int DMnd)
{
	double HighECharge = calculate_highE_Charge_extFPMD(pSPARC, pSPARC->Efermi);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	#ifdef DEBUG
	if (rank==0) printf("== highE_rho_extFPMD ==: HighECharge = %f\n", HighECharge);
	#endif

	double V = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z * pSPARC->Jacbdet;
	double high_E_rho_val = HighECharge / V;
	
	int isBetaZero = fabs(beta) < TEMP_TOL ? 1 : 0; 
	
	// TODO: take care of the prefactor for spin
	if (isBetaZero) {
		for (int i = 0; i < DMnd; i++) {
			rho[i] = high_E_rho_val;
		}
	} else {
		for (int i = 0; i < DMnd; i++) {
			rho[i] = beta * rho[i] + high_E_rho_val;
		}
	}
}


/**
 * @brief Calculate the kinetic energy of high-energy electrons.
 * 
 * @param pSPARC SPARC object.
 * @param lambda_f Fermi level.
 * @return double Total kinetic energy of high-energy electrons.
 */
double calculate_highE_Tk_extFPMD(SPARC_OBJ *pSPARC, double lambda_f)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double V = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z * pSPARC->Jacbdet;
	double U0 = pSPARC->ext_FPMD_U0;
	double Ec = pSPARC->lambda_sorted[pSPARC->Nstates-1];
	double Emax = find_Emax(pSPARC, lambda_f, Ec);
	double dE = 0.1 / 27.21138; // 0.1 eV converted into Hartree
	double HighETk = high_E_Tk(
		pSPARC->elec_T_type, pSPARC->Beta, lambda_f, V, U0,
		Ec, Emax, dE, pSPARC->kptcomm_active);
	#ifdef DEBUG
	if (rank == 0) {
		printf("rank = %2d, Ec = %f, Emax = %f, dE = %e, U0 = %f, HighETk = %f\n",
			rank, Ec, Emax, dE, U0, HighETk);
	}
	#endif
	return HighETk;
}



/**
 * @brief Occupation times the Density of States for high-Energy electrons.
 * 
 *        Occupation is usually given by the Fermi-Dirac function.
 *        High-energy DOS is given by the free electron DOS. Returns
 *            func(lambda) = -1.0 * DOS(lambda) * {g*ln(g)+[1-g]*ln[1-g]},
 *        where g = g(lambda) is the occupation.
 * 
 * @param lambda Energy level.
 * @param data All data other than the energy level are stored here.
 *             In this function, we assume the double_data contains the
 *             following data:
 *             [c, U0, smearing_type, beta, lambda_f],
 *             where c = sqrt(2)*Volume/pi^2.
 * @return double 
 */
double DOS_times_EntropyTerm(double lambda, DATA_INFO *data)
{
	double DOS = high_E_DOS(lambda, data);
	int smearing_type = (int) data->double_data[2];
	double beta = data->double_data[3];
	double lambda_f = data->double_data[4];
	double g = smearing_function(beta, lambda, lambda_f, smearing_type);
	return -DOS * (g * log(g) + (1.0-g) * log(1.0-g));
}



/**
 * @brief   Find the electronic entropy of the high-energy electrons provided
 *          lambda_f when using the ext-FPMD method.
 * 
 *          This routine finds
 *            \int_{Ec}^{Emax} D(x) {f(x)*ln(f(x))+[1-f(x)]*ln[1-f(x)]} dx,
 *          where f(x) is the smearing function (with given smearing and
 *          fermi-level), D(x) is the the density of state (DOS) of high
 *          energy electrons, which is expressed as
 *              D(x) = sqrt(2)*V/pi^2 sqrt(x - U0).
 */
double high_E_Entropy(
	int elec_T_type, double Beta, double lambda_f, double V, double U0,
	double Ec, double Emax, double mesh, MPI_Comm comm)
{
	// set up data for the integral of occ * DOS for high energy electrons
	DATA_INFO extraData;
	extraData.len_double = 5;
	extraData.double_data = malloc(extraData.len_double * sizeof(double));
	assert(extraData.double_data != NULL);
	extraData.double_data[0] = sqrt(2.0)*V/(M_PI*M_PI); // sqrt(2)*V/pi^2
	extraData.double_data[1] = U0;
	extraData.double_data[2] = (double) elec_T_type;
	extraData.double_data[3] = Beta;
	extraData.double_data[4] = lambda_f;

	size_t N_int = ceil((Emax - Ec) / mesh);
	// next find the high energy charges
	double HighE_Tk = integral_simpson_paral(
		Ec, Emax, N_int, DOS_times_EntropyTerm, &extraData, comm);

	free(extraData.double_data);
	return HighE_Tk;
}



/**
 * @brief Calculate the electronic entropy of high-energy electrons.
 * 
 * @param pSPARC SPARC object.
 * @param lambda_f Fermi level.
 * @return double Total kinetic energy of high-energy electrons.
 */
double calculate_highE_Entropy_extFPMD(SPARC_OBJ *pSPARC, double lambda_f)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	#ifdef DEBUG
	if (rank == 0) printf("Start evaluating entropy of high-energy electrons\n");
	#endif

	double V = pSPARC->range_x * pSPARC->range_y * pSPARC->range_z * pSPARC->Jacbdet;
	double U0 = pSPARC->ext_FPMD_U0;
	double Ec = pSPARC->lambda_sorted[pSPARC->Nstates-1];
	
	double Emax = find_Emax(pSPARC, lambda_f, Ec);
	double dE = 0.1 / 27.21138; // 0.1 eV converted into Hartree
	double HighEEntropy = high_E_Entropy(
		pSPARC->elec_T_type, pSPARC->Beta, lambda_f, V, U0,
		Ec, Emax, dE, pSPARC->kptcomm_active);
	#ifdef DEBUG
	if (rank == 0) {
		printf("rank = %2d, Ec = %f, Emax = %f, dE = %e, U0 = %f, HighEEntropy = %f\n",
			rank, Ec, Emax, dE, U0, HighEEntropy);
	}
	#endif
	return HighEEntropy;
}


