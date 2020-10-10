#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <math.h>

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
// Kernels

__global__ void fase1SinMemoriaCompartida(float * A, float * B, float * C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int k = blockIdx.x;
	int istart = k * blockDim.x;
	int iend = istart + blockDim.x;

	float resultado = 0;
	for (int j=istart; j<iend;j++){
		float a=A[j]*i;
		if ((int)ceil(a) % 2 ==0)
			resultado += a + B[j];
		else
			resultado += a - B[j];
	}

	C[i] = resultado;
}

__global__ void fase1ConMemoriaCompartida(float * A, float * B, float * C) {
	extern __shared__ float sdata[];	
	float *A_s = sdata; // Bsize floats
	float *B_s = (float*)&A_s[blockDim.x]; // otros Bsize floats

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	A_s[tid] = A[i];
	B_s[tid] = B[i];
	__syncthreads();

	float resultado = 0;
	for (int j=0; j<blockDim.x;j++){
		float a=A_s[j]*i;
		if ((int)ceil(a) % 2 ==0)
			resultado += a + B_s[j];
		else
			resultado += a - B_s[j];
	}

	C[i] = resultado;
}

__global__ void fase2(float * C, float * D) {
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = C[i];
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s)
				sdata[tid] += sdata[tid+s];		
		__syncthreads();
	}

	if (tid == 0) 
		D[blockIdx.x] = sdata[0];	
}

__global__ void fase3(float * C, float * resultado) {
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = C[i];
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1){
		if (tid < s) 
			if(sdata[tid] < sdata[tid+s]) 
				sdata[tid] = sdata[tid+s];		
		__syncthreads();
	}

	if (tid == 0) 
		resultado[blockIdx.x] = sdata[0];	
}


//**************************************************************************
int main(int argc, char *argv[]) {
	int Bsize, NBlocks;

	if (argc != 3){
		cout << "Uso: transformacion Num_bloques Tam_bloque  "<<endl;
		return(-1);
	} else {
		NBlocks = atoi(argv[1]);
		Bsize= atoi(argv[2]);
		if (Bsize != 64 && Bsize != 128 && Bsize != 256){
			cout << "Tamano de bloque debe ser: 64, 128 o 256" << endl;
			return (-1);
		}
	}

	// This will pick the best possible CUDA capable device
	// int devID = findCudaDevice(argc, (const char **)argv);

	//Get GPU information
	int devID;
	cudaDeviceProp props;
	cudaError_t err;
	err = cudaGetDevice(&devID);
	if(err != cudaSuccess) {
		cout << "ERROR" << endl;
		return (-1);
	}

	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);
	
	// Preparamos los vectores

	const int N=Bsize*NBlocks;

	// Los resultados de las versiones CPU se guardan en _CPU y los de la GPU en _h	

	// Para la primera fase
	float *A = new float[N];
	float *B = new float[N];
	float *C_CPU = new float[N];
	float *C_h = new float[N];
	float *C_h_compartida = new float[N];

	// Segunda fase
	float *D_CPU = new float[NBlocks];
	float *D_h = new float[NBlocks];
	float *D_h_compartida = new float[NBlocks];
	
	// Tercera fase
	float *reduccionMax_h = new float[NBlocks];
	float *reduccionMax_d;
	float maximo_c_CPU;
	float maximo_c_h;
	float maximo_c_h_compartida;

	for (int i=0; i<N;i++){
		A[i]= (float) (1  -(i%100)*0.001);
		B[i]= (float) (0.5+(i%10) *0.1  );    
	}

	float *A_d, *B_d, *C_d, *D_d;
	int size = N*sizeof(float);

	cudaMalloc((void**)&A_d, size);
	cudaMalloc((void**)&B_d, size);
	cudaMalloc((void**)&C_d, size);
	
	err = cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
		return (-1);
	}

	err = cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
		return (-1);
	}

	// Fases CPU
	// Primera fase, calcular C

	double t1 = cpuSecond();

	for (int i=0; i<N;i++){
		C_CPU[i]=0.0;
		int k = floor((float)i/Bsize);
		int istart=k*Bsize;
		int iend  =istart+Bsize;

		for (int j=istart; j<iend;j++){
			float a=A[j]*i;
			if ((int)ceil(a) % 2 ==0)
				C_CPU[i]+= a + B[j];
			else
				C_CPU[i]+= a - B[j];
		}
	}

	double tFase1CPU = cpuSecond() - t1;

	// Segunda fase, calcular D
	
	t1 = cpuSecond();

	for (int k=0; k<NBlocks;k++){
		int istart=k*Bsize;
		int iend  =istart+Bsize;
		D_CPU[k]=0.0;

		for (int i=istart; i<iend;i++)
			D_CPU[k]+=C_CPU[i];
	}

	double tFase2CPU = cpuSecond() - t1;

	// Tercera fase, calcular maximo

	t1 = cpuSecond();

	maximo_c_CPU = 0.0;
	for (int i=0; i<N; i++)
		if (maximo_c_CPU < C_CPU[i])
			maximo_c_CPU = C_CPU[i];

	double tFase3CPU = cpuSecond() - t1;


	// Fases GPU
	// Primera fase sin memoria compartida, calcular C

	t1 = cpuSecond();

	int threadsPerBlock = Bsize;
	int numBlocks = NBlocks;

	fase1SinMemoriaCompartida<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d);

	double tFase1GPU = cpuSecond() - t1;
	
	err = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
		return (-1);
	}

	// Primera fase con memoria compartida, calcular C

	t1 = cpuSecond();

	int threadsPerBlockCompartida = Bsize;
	int numBlocksCompartida = NBlocks;
	int smemSizeCompartida = threadsPerBlockCompartida*sizeof(float)*2;

	fase1ConMemoriaCompartida<<<numBlocksCompartida, threadsPerBlockCompartida, smemSizeCompartida>>>(A_d, B_d, C_d);

	double tFase1GPUCompartida = cpuSecond() - t1;
	
	err = cudaMemcpy(C_h_compartida, C_d, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU2" << endl;
		return (-1);
	}

	cudaFree(A_d);
	cudaFree(B_d);

	// Segunda fase, calcular D, con C obtenido con memoria compartida

	cudaMalloc((void**)&D_d, NBlocks*sizeof(float));

	t1 = cpuSecond();

	int threadsPerBlock2 = Bsize;
	int numBlocks2 = NBlocks;
	int smemSize = threadsPerBlock2*sizeof(float);

	fase2<<<numBlocks2, threadsPerBlock2, smemSize>>>(C_d, D_d);

	double tFase2GPUCompartida = cpuSecond() - t1;

	err = cudaMemcpy(D_h_compartida, D_d, NBlocks*sizeof(float),cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
		return (-1);
	}

	cudaFree(D_d);

	// Tercera fase, calcular maximo, con C obtenido con memoria compartida

	cudaMalloc((void**)&reduccionMax_d, NBlocks*sizeof(float));

	t1 = cpuSecond();

	fase3<<<numBlocks2, threadsPerBlock2, smemSize>>>(C_d, reduccionMax_d);

	err = cudaMemcpy(reduccionMax_h, reduccionMax_d, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
		return (-1);
	}

	// ultima reduccion
	maximo_c_h_compartida = 0.0;
	for (int i=0; i<NBlocks; i++)
		if (maximo_c_h_compartida < reduccionMax_h[i])
			maximo_c_h_compartida = reduccionMax_h[i];

	double tFase3GPUCompartida = cpuSecond() - t1;

	cudaFree(reduccionMax_d);

	// Segunda fase, calcular D, con C obtenido sin memoria compartida

	err = cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
		return (-1);
	}

	cudaMalloc((void**)&D_d, NBlocks*sizeof(float));

	t1 = cpuSecond();

	fase2<<<numBlocks2, threadsPerBlock2, smemSize>>>(C_d, D_d);

	double tFase2GPU = cpuSecond() - t1;

	err = cudaMemcpy(D_h, D_d, NBlocks*sizeof(float),cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
		return (-1);
	}

	cudaFree(D_d);

	// Tercera fase, calcular maximo

	cudaMalloc((void**)&reduccionMax_d, NBlocks*sizeof(float));

	t1 = cpuSecond();

	fase3<<<numBlocks2, threadsPerBlock2, smemSize>>>(C_d, reduccionMax_d);

	err = cudaMemcpy(reduccionMax_h, reduccionMax_d, NBlocks*sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
		return (-1);
	}

	// ultima reduccion
	maximo_c_h = 0.0;
	for (int i=0; i<NBlocks; i++)
		if (maximo_c_h < reduccionMax_h[i])
			maximo_c_h = reduccionMax_h[i];

	double tFase3GPU = cpuSecond() - t1;

	cudaFree(reduccionMax_d);


	// Obtenemos la mayor diferencia

	float dif_fase1 = 0;
	float dif_fase2 = 0;
	float dif_fase3 = abs(maximo_c_CPU - maximo_c_h);
	float dif_fase1_compartida = 0;
	float dif_fase2_compartida = 0;
	float dif_fase3_compartida = abs(maximo_c_CPU - maximo_c_h_compartida);

	for (int i=0; i<N; i++) {
		if (dif_fase1 < abs(C_CPU[i] - C_h[i])) 
			dif_fase1 = abs(C_CPU[i] - C_h[i]);
		if (dif_fase1_compartida < abs(C_CPU[i] - C_h_compartida[i])) 
			dif_fase1_compartida = abs(C_CPU[i] - C_h_compartida[i]);
	}

	for (int i=0; i<NBlocks; i++){
		if (dif_fase2 < abs(D_CPU[i] - D_h[i])) 
			dif_fase2 = abs(D_CPU[i] - D_h[i]);
		if (dif_fase2_compartida < abs(D_CPU[i] - D_h_compartida[i])) 
			dif_fase2_compartida = abs(D_CPU[i] - D_h_compartida[i]);
	}


	// Imprimimos resultados

	cout<<endl<<"N="<<N<<"= "<<Bsize<<"*"<<NBlocks<<endl<<endl;
	cout << "Tiempo gastado CPU fase 1 = " << tFase1CPU << endl;
	cout << "Tiempo gastado GPU fase 1 sin memoria compartida = " << tFase1GPU << endl;
	cout << "Ganancia GPU fase 1 sin memoria compartida = " << tFase1CPU / tFase1GPU << endl;
	cout << "Tiempo gastado GPU fase 1 con memoria compartida = " << tFase1GPUCompartida << endl;
	cout << "Ganancia GPU fase 1 con memoria compartida = " << tFase1CPU / tFase1GPUCompartida << endl;
	cout << endl;

	cout << "Tiempo gastado CPU fase 2 = " << tFase2CPU << endl;
	cout << "Tiempo gastado GPU fase 2 = " << tFase2GPU << endl;
	cout << "Ganancia GPU fase 2 = " << tFase2CPU / tFase2GPU << endl;
	cout << "Tiempo gastado CPU fase 3 = " << tFase3CPU << endl;
	cout << "Tiempo gastado GPU fase 3 = " << tFase3GPU << endl;
	cout << "Ganancia GPU fase 3 = " << tFase3CPU / tFase3GPU << endl;
	cout << endl;

	cout << "Diferencia GPU fase 1 (no memoria compartida) = " << dif_fase1 << endl;
	cout << "Diferencia GPU fase 1 (con memoria compartida) = " << dif_fase1_compartida << endl;
	cout << "Diferencia GPU fase 2 (con C obtenido sin memoria compartida) = " << dif_fase2 << endl;
	cout << "Diferencia GPU fase 2 (con C obtenido con memoria compartida) = " << dif_fase2_compartida << endl;
	cout << "Diferencia GPU fase 3 (con C obtenido sin memoria compartida) = " << dif_fase3 << endl;
	cout << "Diferencia GPU fase 3 (con C obtenido con memoria compartida) = " << dif_fase3_compartida << endl;
}
