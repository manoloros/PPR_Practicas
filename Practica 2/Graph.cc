//***********************************************************************
#include "Graph.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

using namespace std;

//***********************************************************************
Graph::Graph ()		// Constructor
{
}
//***********************************************************************
void Graph::fija_nverts (const int nverts)
{
A=new int[nverts*nverts];
vertices=nverts;
}
//***********************************************************************
void Graph::inserta_arista(const int vertA, const int vertB, const int edge) // inserta A->B
{
  A[vertA*vertices+vertB]=edge;
}
//***********************************************************************
int Graph::arista(const int ptA,const int ptB)
{
  return A[ptA*vertices+ptB];
}
//***********************************************************************
void Graph::imprime()
{
 int i,j,vij;
 for(i=0;i<vertices;i++)
 {cout << "A["<<i << ",*]= ";
   
  for(j=0;j<vertices;j++)
   {
      if (A[i*vertices+j]==INF) 
        cout << "INF";
      else  
        cout << A[i*vertices+j];
      if (j<vertices-1) 
        cout << ",";
      else
        cout << endl;
   }
 }
}
//***********************************************************************
void Graph::lee(char *filename)
{
#define BUF_SIZE 100
std::ifstream infile(filename);

if (!infile)
	{
	 cerr << "Nombre de archivo inválido \"" << filename << "\" !!" << endl;
	 cerr << "Saliendo........." << endl;
	 exit(-1);
	}
//Obten el numero de vertices
  char buf[BUF_SIZE];
  infile.getline(buf,BUF_SIZE,'\n');
  vertices=atoi(buf);
  A=new int[vertices*vertices];
 
  int i,j;
  for(i=0;i<vertices;i++)
     for(j=0;j<vertices;j++)
	 if (i==j) A[i*vertices+j]=0;
         else A[i*vertices+j]=INF;
    
  while (infile.getline(buf,BUF_SIZE) && infile.good() && !infile.eof())
	{
	 char *vertname2 = strpbrk(buf, " \t");
	 *vertname2++ = '\0';
	 char *buf2 = strpbrk(vertname2, " \t");
	 *buf2++ = '\0';
	 int weight = atoi(buf2);
	 i=atoi(buf);
	 j=atoi(vertname2);
         A[i*vertices+j]=weight;
	 }
}
//***********************************************************************
int * Graph::Get_Matrix() 
{
	return A;
}
//***********************************************************************

int * Graph::repartirMatriz(int N, int size, int raiz_P, int tam, int rank)
{
	int *buf_envio = (int *) malloc(N*N*sizeof(int));

	if (rank == 0){
		int fila_P, columna_P, comienzo;

		MPI_Datatype MPI_BLOQUE;
		/* Defino el tipo bloque cuadrado */
		MPI_Type_vector (tam, tam, N, MPI_INT, &MPI_BLOQUE);
		/* Creo el nuevo tipo */
		MPI_Type_commit (&MPI_BLOQUE);

		/* Empaqueta bloque a bloque en el buffer de envío*/
		for (int i=0, posicion=0; i<size; i++) {
			/* Calculo la posicion de comienzo de cada submatriz */
			fila_P = i/raiz_P;
			columna_P = i%raiz_P;
			comienzo = (columna_P*tam)+(fila_P*tam*tam*raiz_P);
			MPI_Pack (&A[comienzo], 1, MPI_BLOQUE,
			buf_envio, sizeof(int)*N*N, &posicion, MPI_COMM_WORLD);
		}

		MPI_Type_free (&MPI_BLOQUE);
	} 

	/*Creo un buffer de recepcion*/
	int *buf_recep = (int *) malloc(tam*tam*sizeof(int));

	/* Distribuimos la matriz entre los procesos */
	MPI_Scatter (buf_envio, sizeof(int)*tam*tam, MPI_PACKED,
	buf_recep, tam*tam, MPI_INT, 0, MPI_COMM_WORLD);

	free(buf_envio);

	return (buf_recep);
}

//***********************************************************************

void Graph::obtenerMatrizResultado(int N, int size, int raiz_P, int tam, int rank, int *buf_recep, int *matriz_resultado)
{
	int *buf_resultado = (int *) malloc(N*N*sizeof(int));

	// Reunimos el resultado en el proceso 0
	MPI_Gather (buf_recep, tam*tam, MPI_INT,
	buf_resultado, tam*tam, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0){
		// Volvemos a crear la matriz a partir de las submatrices
		int posicion = 0;
		int fila_P, columna_P, comienzo;

		MPI_Datatype MPI_BLOQUE;
		MPI_Type_vector (tam, tam, N, MPI_INT, &MPI_BLOQUE);
		MPI_Type_commit (&MPI_BLOQUE);

		for (int i=0; i<size; i++){
			fila_P = i/raiz_P;
			columna_P = i%raiz_P;
			comienzo = (columna_P*tam)+(fila_P*tam*tam*raiz_P);
			MPI_Unpack (buf_resultado, N*N*sizeof(int), &posicion, &matriz_resultado[comienzo], 1, MPI_BLOQUE, MPI_COMM_WORLD);
		}

		MPI_Type_free (&MPI_BLOQUE);
	}

	free(buf_resultado);
}
