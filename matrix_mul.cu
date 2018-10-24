#include <stdio.h>
#include <time.h>
#include <cuda.h>

#define N 16

//Cuda error checking - non mandatory
void cudaCheckError() {
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) {
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
   exit(0); 
 }
}

void matrix_mul(int *a,int *b, int *c){
	int *tempA;
	int *tempB;
	int *tempC;
	for (int i = 0 ; i < 4 ; i++ ){
		for (int k = 0 ; k < 4 ; k++ ){
			int temporal = 0 ;
			for (int j = 0 ; j < 4 ; j++ ){
				tempA = a+(4*i+j);
				tempB = b+(4*j+k);
				tempC = c+(4*i+k);
				temporal += (*tempA)*(*tempB);
				*tempC = temporal;
			}
		}
	}
}

void Filling_Matrix(int *a){
	int *temp = a;
	srand(time(NULL));
	for(int i=0;i<17;i++){
		*temp = rand();
		temp++;
	}
}

__global__ void matrix_multiplication( int *a, int *b, int *c ) {

	int Filas = blockIdx.y*blockDim.y+threadIdx.y;
	int Columnas = blockIdx.x*blockDim.x+threadIdx.x;

	float SumaTemporal = 0;

	if (Filas < N && Columnas < N) {// cada thread se encarga de un bloque de la sub matrix
		for (int i = 0; i < N; i++) {
			SumaTemporal += a[Filas * N + i] * b[i * N + Columnas];
		}
	}
	c[Filas * N + Columnas] = SumaTemporal;
}

int main( void ) {
	
	clock_t t_ini, t_fin;	

	int *a, *b, *c;           // datos en el host    
	int *dev_a, *dev_b, *dev_c;   //datos en el dispositivo
	int size = N * sizeof( int ); // asignación de memoria
	
	// asginación de memoria para los dispositivos
	cudaMalloc( (void**)&dev_a, size );
	cudaMalloc( (void**)&dev_b, size );
	cudaMalloc( (void**)&dev_c, size );
	
	a = (int*)malloc( size );
	b = (int*)malloc( size );
	c = (int*)malloc( size );
	
	//se encarga de llenar de datos las matrices
	Filling_Matrix(a);
	Filling_Matrix(b);
		
	// entradas copiadas a los dispotivos
	cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );

//se ejecuta la multiplicación con cuda
	t_ini = clock();
	matrix_multiplication<<<1,16>>>(dev_a,dev_b,dev_c);
	t_fin = clock();
	// copy device result back to host copy of c
	cudaMemcpy( c, dev_c, size, cudaMemcpyDeviceToHost );
	
	printf("CUDA TIME %f \n \n",(double)(t_fin - t_ini));

//Calculo  sin cuda
        t_ini = clock();
        matrix_mul(a,b,c);
        t_fin = clock();
        printf("CPU TIME %f \n \n",(double)(t_fin - t_ini));
	
	free( a );
	free( b );
	free( c );
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	
	return 0;
}

