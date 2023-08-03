/**
 * @file    ConvolutionalGGAexchangeCorrelation.h
 * @brief   This file contains declaration of functions required for convolutional GGA.
 *			In convolutional GGA, the exchange potential is calculated using 3D convolutions
 *			of finite-difference stencils with electron density and correlation potential is
 *			 calculated using SPARC's original implementation.
 * @author  Sushree Jagriti Sahoo <ssahoo41@gatech.edu>
 *          Andrew J. Medford <ajm@gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group, Georgia Tech.
 */

 /**
 * @brief  function to calculate the XC potential using GGA_CONV_PBE
 *
 */
void Calculate_Vxc_GGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho);

/**
 * @brief   function to calculate the exchange potential using GGA_CONV_PBE
 * 
 */
void Calculate_Vx_GGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *XPotential, double *e_x, double *Dxdgrho);

/**
 * @brief   function to calculate the correlation potential using standard GGA_PBE implementation
 *
 */
void Calculate_Vc_GGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *CPotential, double *e_c, double *Dcdgrho);

/**
 * @brief  function to calculate the XC potential using GGA_CONV_PBE for spin-polarized case
 *
 */
void Calculate_Vxc_GSGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho);

/**
* @brief function to calculate exchange potential for spin polarized case using GGA_CONV_PBE
*
*/
void Calculate_Vx_GSGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *XPotential, double *e_x, double *Dxdgrho);

/**
* @brief function to calculate correlation potential for spin polarized case using GGA_CONV_PBE
*
*/
void Calculate_Vc_GSGA_CONV_PBE(SPARC_OBJ *pSPARC, XCCST_OBJ *xc_cst, double *rho, double *CPotential, double *e_c, double *Dcdgrho);
