#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int ProcNum = 0;      // Number of available processes
int ProcRank = 0;     // Rank of current process

// Function for random definition of matrix and vector elements
void RandomDataInitialization(double *pMatrix, double *pVector, int Size) {
    int i, j;  // Loop variables
    srand(unsigned(clock()));
    for (i = 0; i < Size; i++) {
        pVector[i] = rand() / double(1000);
        for (j = 0; j < Size; j++)
            pMatrix[j * Size + i] = rand() / double(1000);
    }
}

// Function for memory allocation and data initialization
void ProcessInitialization(double *&pMatrix, double *&pVector, double *&pProcVector,
                           double *&pResult, double *&pProcColumns, double *&pProcResult,
                           int &Size, int &ColumnNum) {
    int RestColumns; // Number of columns, that haven’t been distributed yet
    int i;             // Loop variable

    setvbuf(stdout, 0, _IONBF, 0);
    if (ProcRank == 0) {
        do {
            printf("\nEnter size of the initial objects: ");
            scanf("%d", &Size);
            if (Size < ProcNum) {
                printf("Size of the objects must be greater than number of processes! \n ");
            }
        } while (Size < ProcNum);
    }
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine the number of matrix rows stored on each process
    RestColumns = Size;
    for (i = 0; i < ProcRank; i++)
        RestColumns = RestColumns - RestColumns / (ProcNum - i);
    ColumnNum = RestColumns / (ProcNum - ProcRank);

    // Memory allocation
    pProcVector = new double[ColumnNum];
    pProcColumns = new double[ColumnNum * Size];
    pProcResult = new double[Size];

    // Obtain the values of initial objects elements
    if (ProcRank == 0) {
        // Initial matrix exists only on the pivot process
        pMatrix = new double[Size * Size];
        // Initial vector exists only on the pivot process
        pVector = new double[Size];
        // Result
        pResult = new double[Size];
        // Values of elements are defined only on the pivot process
        RandomDataInitialization(pMatrix, pVector, Size);
    }
}

// Function for distribution of the initial objects between the processes
void DataDistribution(double *pMatrix, double *pProcColumns, double *pVector, double *pProcVector,
                      int Size, int ColumnNum) {
    int *pSendNum; // The number of elements sent to the process
    int *pSendInd; // The index of the first data element sent to the process
    int RestColumns = Size; // Number of rows, that haven’t been distributed yet
    int *pSendVecNum; // The number of elements of vector send to the process
    int *pSendVecInd; // The index of the first element in vector sent to the process

    // Alloc memory for temporary objects
    pSendInd = new int[ProcNum];
    pSendNum = new int[ProcNum];
    pSendVecInd = new int[ProcNum];
    pSendVecNum = new int[ProcNum];

    // Define the disposition of the matrix rows for current process
    ColumnNum = (Size / ProcNum);
    pSendNum[0] = ColumnNum * Size;
    pSendInd[0] = 0;
    pSendVecNum[0] = ColumnNum;
    pSendVecInd[0] = 0;
    for (int i = 1; i < ProcNum; i++) {
        RestColumns -= ColumnNum;
        ColumnNum = RestColumns / (ProcNum - i);
        pSendNum[i] = ColumnNum * Size;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
        pSendVecNum[i] = ColumnNum;
        pSendVecInd[i] = pSendVecInd[i - 1] + pSendVecNum[i - 1];
    }

    // Scatter the partial vectors
    MPI_Scatterv(pVector, pSendVecNum, pSendVecInd, MPI_DOUBLE, pProcVector, pSendVecNum[ProcRank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    // Scatter the columns
    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcColumns,
                 pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);


    delete[] pSendNum;
    delete[] pSendInd;
    delete[] pSendVecNum;
    delete[] pSendVecInd;
}

void ResultReplication(double *pProcResult, double *pResult, int Size) {
    MPI_Reduce(pProcResult, pResult, Size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

}

void ParallelResultCalculation(
        double *pProcColumns, double *pProcVector, double *pProcResult, int Size, int ColumnNum) {
    int i, j;  // Loop variables
    for (i = 0; i < Size; i++)
        pProcResult[i] = 0;
    for (j = 0; j < ColumnNum; j++) {
        for (i = 0; i < Size; i++)
            pProcResult[i] += pProcColumns[j * Size + i] * pProcVector[j];
    }
}

void ProcessTermination(double *pMatrix, double *pVector, double *pProcVector, double *pResult,
                        double *pProcColumns, double *pProcResult) {
    if (ProcRank == 0) {
        delete[] pMatrix;
        delete[] pVector;
        delete[] pResult;
    }
    delete[] pProcVector;
    delete[] pProcColumns;
    delete[] pProcResult;
}

int main(int argc, char *argv[]) {
    double *pMatrix;  // The first argument - initial matrix
    double *pVector;  // The second argument - initial vector
    double *pProcVector; // The partial vector on the current process
    double *pResult;  // Result vector for matrix-vector multiplication
    int Size;            // Sizes of initial matrix and vector
    double *pProcColumns;   // Stripe of the matrix on the current process
    double *pProcResult; // Block of the result vector on the current process
    int ColumnNum;          // Number of columns in the matrix stripe
    double Duration, Start, Finish;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) {
        printf("Parallel matrix-vector multiplication program\n");
    }

    ProcessInitialization(pMatrix, pVector, pProcVector, pResult, pProcColumns, pProcResult,
                          Size, ColumnNum);

    Start = MPI_Wtime();

    DataDistribution(pMatrix, pProcColumns, pVector, pProcVector, Size, ColumnNum);
    ParallelResultCalculation(pProcColumns, pProcVector, pProcResult, Size, ColumnNum);
    ResultReplication(pProcResult, pResult, Size);

    Finish = MPI_Wtime();
    Duration = Finish - Start;

    if (ProcRank == 0) {
        printf("Time of execution = %f\n", Duration);
    }

    ProcessTermination(pMatrix, pVector, pProcVector, pResult, pProcColumns, pProcResult);

    MPI_Finalize();
}