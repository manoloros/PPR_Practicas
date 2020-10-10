/* ******************************************************************** */ 
/*               Algoritmo Branch-And-Bound Paralelo                    */
/* ******************************************************************** */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include "libbb.h"

using namespace MPI;

unsigned int NCIUDADES;
int rank, size;
MPI_Comm comunicadorCarga;	// Para la distribuci�n de la carga
MPI_Comm comunicadorCota;	// Para la difusi�n de una nueva cota superior detectada

main (int argc, char **argv) {

  MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size); 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	switch (argc) {
		case 3:	NCIUDADES = atoi(argv[1]);
						break;
		default:	std::cerr << "La sintaxis es: " << argv[0] << " <tamanio> <archivo>" << std::endl;
							exit(1);
							break;
	}

	if (rank == 0) std::cout << "Con " << size << " procesos y con NCIUDADES = " << NCIUDADES << std::endl;

	int** tsp0 = reservarMatrizCuadrada(NCIUDADES);
	tNodo	nodo,         // nodo a explorar
			lnodo,        // hijo izquierdo
			rnodo,        // hijo derecho
			solucion;     // mejor solucion
	bool fin,        // condicion de fin
		nueva_U;       // hay nuevo valor de c.s.
	int  U;             // valor de c.s.
	int iteraciones = 0;
	tPila pila;         // pila de nodos a explorar

	// Inicializacion de variables

	MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comunicadorCarga);
	MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comunicadorCota);

	U = INFINITO;                  // inicializa cota superior
	InicNodo (&nodo);              // inicializa estructura nodo
	InicNodo (&solucion);

	inicializacion_variables_comunicacion();

	fin = false;

	double t;

	// Broadcast de la matriz tsp0

	if (rank == 0) {
		LeerMatriz (argv[2], tsp0);    // lee matriz de fichero
		MPI_Bcast (tsp0[0], NCIUDADES*NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);

		fin = Inconsistente(tsp0);

		t = MPI_Wtime();
	} else {
		MPI_Bcast (tsp0[0], NCIUDADES*NCIUDADES, MPI_INT, 0, MPI_COMM_WORLD);

		fin = Inconsistente(tsp0);

		t = MPI_Wtime();
		equilibrado_carga(&pila, &fin, &solucion);
		if (!fin) pila.pop(nodo);
	}

	// Algoritmo Branch&Bound

	while (!fin) {       // ciclo del Branch&Bound
		Ramifica (&nodo, &lnodo, &rnodo, tsp0);		
		nueva_U = false;

		if (Solucion(&rnodo)) {
			if (rnodo.ci() < U) {    // se ha encontrado una solucion mejor
				U = rnodo.ci();
				nueva_U = true;
				CopiaNodo (&rnodo, &solucion);
			}
		} else {                    //  no es un nodo solucion
			if (rnodo.ci() < U) {     //  cota inferior menor que cota superior
				if (!pila.push(rnodo)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}

		if (Solucion(&lnodo)) {
			if (lnodo.ci() < U) {    // se ha encontrado una solucion mejor
				U = lnodo.ci();
				nueva_U = true;
				CopiaNodo (&lnodo,&solucion);
			}
		} else {                     // no es nodo solucion
			if (lnodo.ci() < U) {      // cota inferior menor que cota superior
				if (!pila.push(lnodo)) {
					printf ("Error: pila agotada\n");
					liberarMatriz(tsp0);
					exit (1);
				}
			}
		}

		difusion_cota_superior(U, nueva_U);
		if (nueva_U) pila.acotar(U);

		equilibrado_carga(&pila, &fin, &solucion);
		if (!fin) pila.pop(nodo);
		iteraciones++;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double tParalelo = MPI_Wtime() - t;

	std::cout << "Proceso " << rank << " iteraciones: " << iteraciones << std::endl;

	int iteracionesSuma;

	MPI_Reduce(&iteraciones, &iteracionesSuma, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	// Comprobamos que el resultado sea correcto

	if (rank == 0){
		// Version secuencial
		t = MPI_Wtime();

		bool activo = true;
		tNodo	nodoSecuencial,         // nodo a explorar
			lnodoSecuencial,        // hijo izquierdo
			rnodoSecuencial,        // hijo derecho
			solucionSecuencial;     // mejor solucion
		bool nueva_U_secuencial;       // hay nuevo valor de c.s.
		int  U_secuencial;             // valor de c.s.
		tPila pilaSecuencial;         // pila de nodos a explorar
		int iteracionesSecuencial = 0;

		U_secuencial = INFINITO;                  // inicializa cota superior
		InicNodo (&nodoSecuencial);              // inicializa estructura nodo
		
		while (activo) {       // ciclo del Branch&Bound
			Ramifica (&nodoSecuencial, &lnodoSecuencial, &rnodoSecuencial, tsp0);		
			nueva_U_secuencial = false;

			if (Solucion(&rnodoSecuencial)) {
				if (rnodoSecuencial.ci() < U_secuencial) {    // se ha encontrado una solucion mejor
					U_secuencial = rnodoSecuencial.ci();
					nueva_U_secuencial = true;
					CopiaNodo (&rnodoSecuencial, &solucionSecuencial);
				}
			} else {                    //  no es un nodo solucion
				if (rnodoSecuencial.ci() < U_secuencial) {     //  cota inferior menor que cota superior
					if (!pilaSecuencial.push(rnodoSecuencial)) {
						printf ("Error: pila agotada\n");
						liberarMatriz(tsp0);
						exit (1);
					}
				}
			}

			if (Solucion(&lnodoSecuencial)) {
				if (lnodoSecuencial.ci() < U_secuencial) {    // se ha encontrado una solucion mejor
					U_secuencial = lnodoSecuencial.ci();
					nueva_U_secuencial = true;
					CopiaNodo (&lnodoSecuencial,&solucionSecuencial);
				}
			} else {                     // no es nodo solucion
				if (lnodoSecuencial.ci() < U_secuencial) {      // cota inferior menor que cota superior
					if (!pilaSecuencial.push(lnodoSecuencial)) {
						printf ("Error: pila agotada\n");
						liberarMatriz(tsp0);
						exit (1);
					}
				}
			}

			if (nueva_U_secuencial) pilaSecuencial.acotar(U_secuencial);
			activo = pilaSecuencial.pop(nodoSecuencial);
			iteracionesSecuencial++;
		}

		double tSecuencial = MPI_Wtime() - t;

		// Comprobamos el resultado
		bool correcto = true;
		for (int i=0; i < 2 * NCIUDADES; i++)
			if (solucion.datos[i] != solucionSecuencial.datos[i])
				correcto = false;


		if (correcto) {
			printf ("Solucion: \n");
			EscribeNodo(&solucion);
			std::cout << "Tiempo gastado secuencial = " << tSecuencial << std::endl;
			std::cout << "Tiempo gastado paralelo = " << tParalelo << std::endl;
			std::cout << "Numero de iteraciones paralelo = " << iteracionesSuma << std::endl;
			std::cout << "Numero de iteraciones secuencial = " << iteracionesSecuencial << std::endl << std::endl;

			// Para copiar datos:
			
			if (size == 2) {
				std::cout << tSecuencial << ", " << tParalelo << ", " << iteracionesSecuencial << ", " << iteracionesSuma << std::endl;
			} else if (size == 3) {
				std::cout << tParalelo << ", " << iteracionesSuma << std::endl;
			}

		} else {
			std::cout << "Resultado del algoritmo paralelo no coincide con el secuencial " << std::endl;
			std::cout << "Resultado algoritmo secuencial: " << std::endl;
			EscribeNodo(&solucionSecuencial);
			std::cout << "Resultado algoritmo paralelo: " << std::endl;
			EscribeNodo(&solucion);
		}
	}

	liberarMatriz(tsp0);

	MPI_Finalize();
}

 
