#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

//**************************************************************************
// Floyd kernel

__global__ void floyd_kernel_2d(int * M, const int nverts, const int k) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y; 
	int ij=i*nverts+j; 

	if (i < nverts && j < nverts) {
  	int Mij = M[ij];
  	int i= ij / nverts;
  	int j= ij - i * nverts;

  	if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
  		Mij = (Mij > Mikj) ? Mikj : Mij;
  		M[ij] = Mij;
		}
	}
}

__global__ void floyd_kernel_1d(int * M, const int nverts, const int k) {
	int ij = threadIdx.x + blockDim.x * blockIdx.x;

  if (ij < nverts * nverts) {
		int Mij = M[ij];
    int i= ij / nverts;
    int j= ij - i * nverts;

    if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
    	Mij = (Mij > Mikj) ? Mikj : Mij;
    	M[ij] = Mij;
		}
  }
}

//**************************************************************************
// Maximo del vector kernel

__global__ void reduceMax(int * V_in, int * V_out, const int N) {
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? V_in[i] : 0);
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s) 
			if(sdata[tid] < sdata[tid+s]) 
				sdata[tid] = sdata[tid+s];		
		__syncthreads();
	}

	if (tid == 0) 
		V_out[blockIdx.x] = sdata[0];
}

int main (int argc, char *argv[]) {

	if (argc != 3) {
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << " <block size>" << endl;
		return(-1);
	}

	int blockSize = atoi(argv[2]);

	// This will pick the best possible CUDA capable device
	// int devID = findCudaDevice(argc, (const char **)argv);

	//Get GPU information
	int devID;
	cudaDeviceProp props;
	cudaError_t err;
	err = cudaGetDevice(&devID);
	if(err != cudaSuccess) {
		cout << "ERRORRR" << endl;
	}

	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	Graph G;
	G.lee(argv[1]);// Read the Graph

	//cout << "EL Grafo de entrada es:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;

	const int nverts2 = nverts * nverts;
	cout << "Con N = " << nverts2 << " y block size = " << blockSize << endl << endl;

	int *c_Out_M = new int[nverts2];
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	int *A = G.Get_Matrix();

	// Fase CPU
	double t1 = cpuSecond();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;

	for(int k = 0; k < niters; k++) {
		      kn = k * nverts;
		for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++)
		     			if (i!=j && i!=k && j!=k){
			 	    inj = in + j;
			 	    A[inj] = min(A[in+k] + A[kn+j], A[inj]);
		     }
		 }
	}

	double Tcpu = cpuSecond() - t1;

	// Fase GPU 1d
	t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	for(int k = 0; k < niters; k++) {
	 	int threadsPerBlockFloyd_1d = blockSize;
	 	int blocksPerGridFloyd_1d = (nverts2 + threadsPerBlockFloyd_1d - 1) / threadsPerBlockFloyd_1d;

	  floyd_kernel_1d<<<blocksPerGridFloyd_1d,threadsPerBlockFloyd_1d>>>(d_In_M, nverts, k);
	  err = cudaGetLastError();

	  if (err != cudaSuccess) {
	  	fprintf(stderr, "Failed to launch kernel!\n");
	  	exit(EXIT_FAILURE);
		}
	}

	cudaDeviceSynchronize();
	double Tgpu1d = cpuSecond()-t1;
	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);

	// Comprobamos si resultado correcto
	for(int i = 0; i < nverts; i++)
		for(int j = 0;j < nverts; j++)
		   if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
		     cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl; 

	// Fase GPU 2d
	t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	for(int k = 0; k < niters; k++) {
	 dim3 threadsPerBlockFloyd_2d(sqrt(blockSize),sqrt(blockSize));
	 dim3 numBlocksFloyd_2d(ceil((float)(nverts)/threadsPerBlockFloyd_2d.x),
			ceil((float)(nverts)/threadsPerBlockFloyd_2d.y));

		floyd_kernel_2d<<<numBlocksFloyd_2d, threadsPerBlockFloyd_2d>>>(d_In_M, nverts, k);
		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch kernel!\n");
			exit(EXIT_FAILURE);
		}
	}

	cudaDeviceSynchronize();
	double Tgpu2d = cpuSecond()-t1;
	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);

	// Comprobamos si resultado correcto
	for(int i = 0; i < nverts; i++)
		for(int j = 0;j < nverts; j++)
		   if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
		     cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl; 


	cout << "Tiempo gastado CPU (Tcpu) = " << Tcpu << endl;
	cout << "Tiempo gastado GPU (Tgpu_1d) = " << Tgpu1d << endl;
	cout << "Ganancia utilizando Tgpu_1d (Sgpu_1d) = " << Tcpu / Tgpu1d << endl; 
	cout << "Tiempo gastado GPU (Tgpu_2d) = " << Tgpu2d << endl;
	cout << "Ganancia utilizando Tgpu_2d (Sgpu_2d) = " << Tcpu / Tgpu2d << endl; 

	// Calculo maximo longitud

	int threadsPerBlockReduccion = blockSize;
	int numBlocksReduccion = ceil((float)nverts2/threadsPerBlockReduccion);
	int smemSize = threadsPerBlockReduccion*sizeof(int); 

	int *vectorReducir = NULL;
	err = cudaMalloc((void **) &vectorReducir, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	int *vectorResultado = new int[nverts2];

	int *vectorReducido = NULL;
	err = cudaMalloc((void **) &vectorReducido, numBlocksReduccion*sizeof(int));
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	err = cudaMemcpy(vectorReducir, c_Out_M, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	reduceMax<<<numBlocksReduccion, threadsPerBlockReduccion, smemSize>>>(vectorReducir, vectorReducido, nverts2);

	err = cudaMemcpy(vectorResultado, vectorReducido, numBlocksReduccion*sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	// Hacemos la ultima reduccion
	int maximaLongitud = 0;
	for (int i=0; i<numBlocksReduccion; i++)
		if (maximaLongitud < vectorResultado[i])
			maximaLongitud = vectorResultado[i];

	// Version CPU
	int maximaLongitudCPU = 0;
	for(int i = 0; i < nverts; i++)
		for(int j = 0;j < nverts; j++)
		   if (maximaLongitudCPU < c_Out_M[i*nverts+j])
		     maximaLongitudCPU = c_Out_M[i*nverts+j];

	if (maximaLongitudCPU != maximaLongitud)
		cout << "Error reduccion, CPU: " << maximaLongitudCPU << " GPU: " << maximaLongitud << endl;
	else
		cout << "Longitud maxima es: " << maximaLongitud << endl;
}
