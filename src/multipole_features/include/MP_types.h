// make structures for multipole objects

#ifndef MP_TYPES_H
#define MP_TYPES_H
typedef struct _MULTIPOLE_OBJ
{
int imageDimX, imageDimY, imageDimZ;
double hx, hy, hz;
int MCSHMaxOrder, MCSHMaxRadialOrder;
double MCSHMaxR, MCSHRStepSize;
int accuracy;
double U[9];

}MULTIPOLE_OBJ;

#endif