//**************************************************************************
#ifndef GRAPH_H
#define GRAPH_H

//**************************************************************************
const int INF= 100000;

//**************************************************************************
class Graph //Adjacency List clas
{
	private:
	  int *A;
	public:
		Graph();
		int vertices;
		void fija_nverts(const int verts);
		void inserta_arista(const int ptA,const int ptB, const int edge);
		int arista(const int ptA,const int ptB);
		void imprime();
		void lee(char *filename);
		int * Get_Matrix();
		int * repartirMatriz(int N, int size, int raiz_P, int tam, int rank);
		void obtenerMatrizResultado(int N, int size, int raiz_P, int tam, int rank, int *buf_recep, int *matrizResultado);
};

//**************************************************************************
#endif
//**************************************************************************
