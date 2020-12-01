#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi/mpi.h>

int ProcNum = 0;      // Number of available processes
int ProcRank = 0;     // Rank of current process

// Function for random definition of matrix and vector elements
void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
    int i, j;  // Loop variables
    srand(unsigned(clock()));
    for (i=0; i<Size; i++) {
        pVector[i] = rand()/double(1000);
        for (j=0; j<Size; j++)
            pMatrix[i*Size+j] = rand()/double(1000);
    }
}

// Function for memory allocation and data initialization
void ProcessInitialization (double* &pMatrix, double* &pVector,
                            double* &pResult, double* &pProcRows, double* &pProcResult,
                            int &Size, int &RowNum) {
    int RestRows; // Number of rows, that haven’t been distributed yet
    int i;             // Loop variable

    setvbuf(stdout, 0, _IONBF, 0);
    if (ProcRank == 0) {
        do {
            printf("\nEnter size of the initial objects: ");
            scanf("%d", &Size);
            if (Size < ProcNum) {
                printf("Size of the objects must be greater than number of processes! \n ");
            }
        }
        while (Size < ProcNum);
    }
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine the number of matrix rows stored on each process
    RestRows = Size;
    for (i=0; i<ProcRank; i++)
        RestRows = RestRows-RestRows/(ProcNum-i);
    RowNum = RestRows/(ProcNum-ProcRank);

    // Memory allocation
    pVector = new double [Size];
    pResult = new double [Size];
    pProcRows = new double [RowNum*Size];
    pProcResult = new double [RowNum];

    // Obtain the values of initial objects elements
    if (ProcRank == 0) {
        // Initial matrix exists only on the pivot process
        pMatrix = new double [Size*Size];
        // Values of elements are defined only on the pivot process
        RandomDataInitialization(pMatrix, pVector, Size);
    }
}

// Function for distribution of the initial objects between the processes
void DataDistribution(double* pMatrix, double* pProcRows, double* pVector,
                      int Size, int RowNum) {
    int *pSendNum; // The number of elements sent to the process
    int *pSendInd; // The index of the first data element sent to the process
    int RestRows=Size; // Number of rows, that haven’t been distributed yet

    MPI_Bcast(pVector, Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Alloc memory for temporary objects
    pSendInd = new int [ProcNum];
    pSendNum = new int [ProcNum];

    // Define the disposition of the matrix rows for current process
    RowNum = (Size/ProcNum);
    pSendNum[0] = RowNum*Size;
    pSendInd[0] = 0;
    for (int i=1; i<ProcNum; i++) {
        RestRows -= RowNum;
        RowNum = RestRows/(ProcNum-i);
        pSendNum[i] = RowNum*Size;
        pSendInd[i] = pSendInd[i-1]+pSendNum[i-1];
    }

    // Scatter the rows
    MPI_Scatterv(pMatrix , pSendNum, pSendInd, MPI_DOUBLE, pProcRows,
                 pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Free the memory
    delete [] pSendNum;
    delete [] pSendInd;
}

// Result vector replication
void ResultReplication(double* pProcResult, double* pResult, int Size,
                       int RowNum) {
    int i;             // Loop variable
    int *pReceiveNum;  // Number of elements, that current process sends
    int *pReceiveInd;  /* Index of the first element from current process
                        in result vector */
    int RestRows=Size; // Number of rows, that haven’t been distributed yet

    //Alloc memory for temporary objects
    pReceiveNum = new int [ProcNum];
    pReceiveInd = new int [ProcNum];

    //Define the disposition of the result vector block of current processor
    pReceiveInd[0] = 0;
    pReceiveNum[0] = Size/ProcNum;
    for (i=1; i<ProcNum; i++) {
        RestRows -= pReceiveNum[i-1];
        pReceiveNum[i] = RestRows/(ProcNum-i);
        pReceiveInd[i] = pReceiveInd[i-1]+pReceiveNum[i-1];
    }

    //Gather the whole result vector on every processor
    MPI_Allgatherv(pProcResult, pReceiveNum[ProcRank], MPI_DOUBLE, pResult,
                   pReceiveNum, pReceiveInd, MPI_DOUBLE, MPI_COMM_WORLD);

    //Free the memory
    delete [] pReceiveNum;
    delete [] pReceiveInd;
}

// Process rows and vector multiplication
void ParallelResultCalculation(double* pProcRows, double* pVector, double* pProcResult, int Size, int RowNum) {
    int i, j;  // Loop variables
    for (i=0; i<RowNum; i++) {
        pProcResult[i] = 0;
        for (j=0; j<Size; j++)
            pProcResult[i] += pProcRows[i*Size+j]*pVector[j];
    }
}

// Function for formatted matrix output
void PrintMatrix (double* pMatrix, int RowCount, int ColCount) {
    int i, j; // Loop variables
    for (i=0; i<RowCount; i++) {
        for (j=0; j<ColCount; j++)
            printf("%7.4f ", pMatrix[i*ColCount+j]);
        printf("\n");
    }
}

// Function for formatted vector output
void PrintVector (double* pVector, int Size) {
    int i;
    for (i=0; i<Size; i++)
        printf("%7.4f ", pVector[i]);
}

// Function for computational process termination
void ProcessTermination (double* pMatrix, double* pVector, double* pResult,
                         double* pProcRows, double* pProcResult) {
    if (ProcRank == 0)
        delete [] pMatrix;
    delete [] pVector;
    delete [] pResult;
    delete [] pProcRows;
    delete [] pProcResult;
}

int main(int argc, char* argv[])
{
    double* pMatrix;  // The first argument - initial matrix
    double* pVector;  // The second argument - initial vector
    double* pResult;  // Result vector for matrix-vector multiplication
    int Size;		    // Sizes of initial matrix and vector
    double* pProcRows;   // Stripe of the matrix on the current process
    double* pProcResult; // Block of the result vector on the current process
    int RowNum;          // Number of rows in the matrix stripe
    double Duration, Start, Finish;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0) {
        printf ("Parallel matrix-vector multiplication program\n");
    }

    ProcessInitialization(pMatrix, pVector, pResult, pProcRows, pProcResult,
                          Size, RowNum);

    Start = MPI_Wtime();

    DataDistribution(pMatrix, pProcRows, pVector, Size, RowNum);
    ParallelResultCalculation(pProcRows, pVector, pProcResult, Size, RowNum);
    ResultReplication(pProcResult, pResult, Size, RowNum);

    Finish = MPI_Wtime();
    Duration = Finish - Start;

    if (ProcRank == 0) {
        printf("Time of execution = %f\n", Duration);
    }

    ProcessTermination(pMatrix, pVector, pResult, pProcRows, pProcResult);

    MPI_Finalize();
}