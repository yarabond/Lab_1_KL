#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function for random definition of matrix and vector elements
void RandomDataInitialization(double *pMatrix, double *pVector, int Size) {
    int i, j; // Loop variables
    srand(unsigned(clock()));
    for (i = 0; i < Size; i++) {
        pVector[i] = rand() / double(1000);
        for (j = 0; j < Size; j++)
            pMatrix[i * Size + j] = rand() / double(1000);
    }
}

// Function for memory allocation and definition of object’s elements
void ProcessInitialization(double *&pMatrix, double *&pVector,
                           double *&pResult, int &Size) {
    // Size of initial matrix and vector definition
    do {
        printf("\nEnter size of the initial objects: ");
        scanf("%d", &Size);
        printf("\nChosen objects size = %d\n", Size);
        if (Size <= 0)
            printf("\nSize of objects must be greater than 0!\n");
    } while (Size <= 0);
    // Memory allocation
    pMatrix = new double[Size * Size];
    pVector = new double[Size];
    pResult = new double[Size];
    // Definition of matrix and vector elements
    RandomDataInitialization(pMatrix, pVector, Size);
}

// Function for matrix-vector multiplication
void ResultCalculation(double *pMatrix, double *pVector, double *pResult,
                       int Size) {
    int i, j;
    for (i = 0; i < Size; i++) {
        pResult[i] = 0;
        for (j = 0; j < Size; j++)
            pResult[i] += pMatrix[i * Size + j] * pVector[j];
    }
}

void ProcessTermination(double *pMatrix, double *pVector, double *pResult) {
    delete[] pMatrix;
    delete[] pVector;
    delete[] pResult;
}

int main() {
    double *pMatrix; // The first argument - initial matrix
    double *pVector; // The second argument - initial vector
    double *pResult; // Result vector for matrix-vector multiplication
    int Size; // Sizes of initial matrix and vector
    time_t start, finish;
    double duration;
    printf("Serial matrix-vector multiplication program\n");

    ProcessInitialization(pMatrix, pVector, pResult, Size);

    start = clock();
    ResultCalculation(pMatrix, pVector, pResult, Size);
    finish = clock();

    duration = (finish - start) / double(CLOCKS_PER_SEC);
    printf("\n Time of execution: %f\n", duration);

    ProcessTermination(pMatrix, pVector, pResult);
}