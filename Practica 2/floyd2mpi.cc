#include "mpi.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sys/time.h>
#include "Graph.h"
using namespace std;

//**************************************************************************

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

//**************************************************************************
 
int main(int argc, char *argv[])
{
    int rank, size, contador;
    MPI_Status estado;
 
    MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		if (argc != 3) {
			if (rank == 0) cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << " <num vertices>" << endl;
			MPI_Barrier(MPI_COMM_WORLD);
			return(-1);
		}

		if (strstr(argv[1], argv[2]) == NULL){
			if (rank == 0) cerr << "ERROR: El numero de vertices no coincide con los del grafo" << endl;
			MPI_Barrier(MPI_COMM_WORLD);
			return (-1);
		}

		int N = atoi(argv[2]);
		int raiz_P = sqrt(size);
		int tam = N/raiz_P;
		int *buf_envio = (int *) malloc(N*N*sizeof(int));
		MPI_Comm comm_misma_fila, comm_misma_columna;
 		Graph G;

		if (fmod(N, sqrt(size)) != 0) {
			if (rank == 0) cerr << "ERROR: El numero de vertices debe ser multiplo de la raiz del numero de procesos" << endl;
			MPI_Barrier(MPI_COMM_WORLD);
			return (-1);
		}

		if (rank == 0) {
			cout << "P: " << size << endl;
			cout << "N: " << N << endl;
		}

    int fila = rank / raiz_P;
    int columna = rank % raiz_P;

		MPI_Comm_split(MPI_COMM_WORLD, fila, columna, &comm_misma_fila);
		MPI_Comm_split(MPI_COMM_WORLD, columna, fila, &comm_misma_columna);

		// Creacion de las submatrices y distribucion
    if (rank == 0) 
				G.lee(argv[1]);	

		int *buf_recep = G.repartirMatriz(N, size, raiz_P, tam, rank);
 
		// Empieza el procesamiento
		double t1, tMPI;
		int kLocal, posicionK, iGlobal, jGlobal, in, inj;

		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 0)
			t1 = MPI_Wtime();

		for(int k = 0; k < N; k++) {
			kLocal = k % tam;
			posicionK = floor(k/tam);
		
			int *vectorFila = (int *) malloc(tam*sizeof(int));
			int *vectorColumna = (int *) malloc(tam*sizeof(int));	

			if (fila == posicionK)
				for (int j=0; j<tam; j++)
					vectorFila[j] = buf_recep[kLocal*tam+j];		

			if (columna == posicionK)
				for (int i=0; i<tam; i++)
					vectorColumna[i] = buf_recep[i*tam+kLocal];

			MPI_Bcast (vectorFila, tam, MPI_INT, posicionK, comm_misma_columna);
			MPI_Bcast (vectorColumna, tam, MPI_INT, posicionK, comm_misma_fila);

			for(int i = 0; i < tam; i++) {
				iGlobal = fila*tam + i;
				in = i * tam;

				for(int j = 0; j < tam; j++) {
						jGlobal = columna*tam + j;
						inj = in + j;

						if (iGlobal != jGlobal && iGlobal != k && jGlobal != k)
							buf_recep[inj] = min(vectorColumna[i] + vectorFila[j], buf_recep[inj]);
				}
			}

			free(vectorFila);
			free(vectorColumna);
		} // fin for

		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 0)
			tMPI = MPI_Wtime() - t1;

		// Fin procesamiento

		// Obtenemos matriz resultado en el proceso 0

		int *matriz_resultado = (int *) malloc(N*N*sizeof(int));
		G.obtenerMatrizResultado(N, size, raiz_P, tam, rank, buf_recep, matriz_resultado);

		free(buf_recep);

		if (rank == 0){
			// Comprobamos que el resultado sea correcto
			int inj, in, kn;
			int niters = N;
			int nverts = N;

			int *A = G.Get_Matrix();

			// Version secuencial
			t1 = cpuSecond();

			for(int k = 0; k < N; k++) {
				kn = k * N;
				for(int i=0;i<N;i++) {
					in = i * N;
					for(int j = 0; j < N; j++)
						if (i!=j && i!=k && j!=k){
							inj = in + j;
							A[inj] = min(A[in+k] + A[kn+j], A[inj]);
						}
				}
			}

			double tSecuencial = cpuSecond() - t1;

			// Comprobacion
			for(int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
					 if (abs(matriz_resultado[i*N+j] - G.arista(i,j)) > 0)
						 cout << "Error (" << i << "," << j << ")   " << matriz_resultado[i*N+j] << "..." << G.arista(i,j) << endl;
			

			cout << "Tiempo de ejecucion secuencial: " << tSecuencial << endl;
			cout << "Tiempo de ejecucion paralelo: " << tMPI << endl;
			cout << "Ganancia de velocidad algoritmo paralelo: " << tSecuencial / tMPI << endl;
			
			cout << endl;

			cout << "Para copiar los datos: " << endl;
			cout << tSecuencial << " " << tMPI << " " << tSecuencial / tMPI << endl;
			cout << tMPI << endl;

			/*for (int i=0; i < N; i++){
				for (int j=0; j < N; j++)
					cout << G.arista(i,j) << " ";
				cout << endl;
			} 

			cout << endl << endl;
			
			for (int i=0; i < N; i++){
				for (int j=0; j < N; j++)
					cout << matriz_resultado[i*N+j] << " ";
				cout << endl;
			}*/

		}

    MPI_Finalize();
    return 0;
}
