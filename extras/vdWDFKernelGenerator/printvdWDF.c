#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "printvdWDF.h"

int main() {
    char folderRoute[] = "./";
    char kernelFileRoute[L_STRING]; // name of the file of the output kernels and 2nd derivative of kernels
    char splineD2FileRoute[L_STRING]; //name of the file of the 2nd derivatives of spline functions
    snprintf(kernelFileRoute,       L_STRING, "%svdWDFreadKernel.c"  ,     folderRoute);
    snprintf(splineD2FileRoute,       L_STRING, "%svdWDFreadSpline.c"  ,     folderRoute);
    FILE *outputFile = NULL;
    outputFile = fopen(kernelFileRoute,"w");
    fprintf(outputFile, "#include <stdlib.h>\n");
    fprintf(outputFile, "#include <stdio.h>\n");
    fprintf(outputFile, "#include <string.h>\n");
    fprintf(outputFile, "#include \"isddft.h\"\n");
    fprintf(outputFile, "#include \"vdWDFreadKernel.h\"\n\n");
    fprintf(outputFile, "void vdWDF_read_kernel_new(SPARC_OBJ *pSPARC) {\n");
    fclose(outputFile);
    outputFile = fopen(splineD2FileRoute,"w");
    fprintf(outputFile, "#include <stdlib.h>\n");
    fprintf(outputFile, "#include <stdio.h>\n");
    fprintf(outputFile, "#include \"isddft.h\"\n");
    fprintf(outputFile, "#include \"vdWDFreadSpline.h\"\n\n");
    fprintf(outputFile, "void read_spline_d2_qmesh_new(SPARC_OBJ *pSPARC) {\n");
    fclose(outputFile);
    vdWDF_generate_kernel(kernelFileRoute, splineD2FileRoute);
    outputFile = fopen(kernelFileRoute,"a");
    fprintf(outputFile, "}\n");
    fclose(outputFile);
    outputFile = fopen(splineD2FileRoute,"a");
    fprintf(outputFile, "}\n");
    fclose(outputFile);

    return 0;
}