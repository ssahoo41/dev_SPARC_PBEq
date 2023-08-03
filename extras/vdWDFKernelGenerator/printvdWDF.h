#define L_STRING 50

void vdWDF_generate_kernel(char *kernelFileName, char *splineFileName);

int kernel_label(int firstq, int secondq, int nqs);

void vdWDF_generate_kernel(char *kernelFileName, char *splineFileName);

void prepare_Gauss_quad(int nIntegratePoints, double aMin, double aMax, double *aPoints, double *aPoints2, double *weights);

void phi_value(int nrpoints, int nIntegratePoints, double **WabMatrix, double *aPoints, double *aPoints2,
 double qmesh1, double qmesh2, double vdWdr, double *realKernel);

double h_function(double aDivd);

void radial_FT(double* realKernel, int nrpoints, double vdWdr, double vdWdk, double* reciKernel);

void d2_of_kernel(double* reciKernel, int nrpoints, double vdWdk, double* d2reciKerneldk2);

void spline_d2_qmesh(double* qmesh, int nqs, double** d2ydx2);

void print_kernel(char* outputName, double **vdWDFkernelPhi, double ** vdWDFd2Phidk2, int nrpoints, int nqs);

void print_d2ydx2(char* outputName2, int nqs, double **vdWDFd2Splineydx2);