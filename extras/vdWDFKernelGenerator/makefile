# Makefile for MPI helloworld code
CC=mpicc
CCFLAGS=-Wall -g
# activate for compiler optimizations:
#CCFLAGS=-Wall -O3
LDFLAGS=
CCFLAGS += -I.
LDLIBS=-lm

all: printvdWDF

printvdWDF: printvdWDF.o vdWDFgenerateKernelSpline.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

printvdWDF.o: printvdWDF.c printvdWDF.h
	$(CC) $(CCFLAGS) -c $<

vdWDFgenerateKernelSpline.o: vdWDFgenerateKernelSpline.c
	$(CC) $(CCFLAGS) -c $< 

clean:
	rm -f *.o printvdWDF