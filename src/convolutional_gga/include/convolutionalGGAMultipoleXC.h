/**
 * @file    convolutionalGGAMultipoleXC.h
 * @brief   This file contains declaration of functions required for multipole dependent convolutional GGA.
 *
 * @author  Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
 *          Andrew J. Medford <ajm@gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

/**
 * @brief   function to calculate XC potential for GGA_CONV_PBE_MULTIPOLE.  
 *
 */
void Calculate_Vxc_GGA_CONV_PBE_MULTIPOLE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho);

/**
 * @brief   function to calculate exchange potential for PBEq where the additional terms for potential
 *          are neglected. The exchange enhancement factor is dependent on spatially-resolved alpha 
 *          which is a function of monopole feature.
 */
void Calculate_Vx_GGA_CONV_PBE_MULTIPOLE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho,
                                         double *XPotential, double *e_x, double *Dxdgrho);

/**
* @brief function to calculate XC potential for GGA_CONV_PBE_MULTIPOLE XC potential for spin-polarized case.
*
*/
void Calculate_Vxc_GSGA_CONV_PBE_MULTIPOLE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho);

/**
 * @brief   function to calculate exchange potential for spin-polarized case for GGA_CONV_PBE_MULTIPOLE
 *          similar to exchange potential for spin paired case
 */
void Calculate_Vx_GSGA_CONV_PBE_MULTIPOLE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *XPotential, double *e_x, double *Dxdgrho);


/**
 * @brief function to calculate dipole feature by taking L2 norm of individual components 
 *
 */
void Construct_feature(SPARC_OBJ *pSPARC, const double *Dx, const double *Dy, const double *Dz, const int DMnd, double *featureVector);

/**
 * @brief function to calculate alpha using any feature vector
 *
 */
void Construct_alpha(SPARC_OBJ *pSPARC, const double *featureVector, const double m, const double n, const int DMnd, double *alpha);
