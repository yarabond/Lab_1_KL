#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int ProcNum = 0;      // Number of available processes
int ProcRank = 0;     // Rank of current process

void PrintVector(double* pVector, int Size);
void PrintMatrix(double* pMatrix, int RowNum, int ColumnNum);

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

void DummyDataInitialization (double* pMatrix, double* pVector, int Size) {
    int i, j; // Loop variables
    for (i=0; i<Size; i++) {
        pVector[i] = 1;
        for (j=0; j<Size; j++)
            pMatrix[i*Size+j] = i;
    }
}

int determineMaxWidth(int Size) {
    for (int i = sqrt(Size); i >= 1; i--)
        if (Size % i == 0)
            return i;
}

void RandomDataInitialization(double* original, double* pMatrix, double* pVector, int Size, int h, int w) {
    int i, j;  // Loop variables
    srand(unsigned(clock()));
    for (i=0; i<Size; i++) {
        pVector[i] = rand()/double(1000);
        for (j = 0; j < Size; j++) {
            double randVal = rand() / double(1000);
            int ind = (i - (i % h)) * Size + (j - (j % w)) * h + (i % h) * w + (j % w);
            pMatrix[ind] = randVal;
            original[i * Size + j] = randVal;
        }
    }
}

void ProcessInitialization (double* &original, double* &pMatrix, double* &pVector, double* &pProcVector,
                            double* &pResult, double* &pProcMatrix, double* &pProcResult,
                            int &Size, int &RowNum, int &ColumnNum) {

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

        int s = determineMaxWidth(ProcNum), q = ProcNum / s;
        RowNum = Size / q, ColumnNum = Size / s;
    }

    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&RowNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ColumnNum, 1, MPI_INT, 0, MPI_COMM_WORLD);

    pProcVector = new double [ColumnNum];
    pProcMatrix = new double [RowNum * ColumnNum];
    pProcResult = new double [RowNum];

    if (ProcRank == 0) {
        original = new double[Size * Size];
        pMatrix = new double [Size*Size];
        pVector = new double[Size];
        pResult = new double[Size];
        DummyDataInitialization(pMatrix, pVector, Size);
    }
}

void DataDistribution(double* pMatrix, double* pProcMatrix, double* pVector, double* pProcVector,
                      int Size, int RowNum, int ColumnNum) {
    int *pSendNum; // The number of elements sent to the process
    int *pSendInd; // The index of the first data element sent to the process
    int *pSendVecNum; // The number of elements of vector send to the process
    int *pSendVecInd; // The index of the first element in vector sent to the process

    // Alloc memory for temporary objects
    pSendInd = new int [ProcNum];
    pSendNum = new int [ProcNum];
    pSendVecInd = new int[ProcNum];
    pSendVecNum = new int[ProcNum];

    // Define the disposition of the matrix and vector for current process
    pSendNum[0] = RowNum * ColumnNum;
    pSendInd[0] = 0;
    pSendVecNum[0] = ColumnNum;
    pSendVecInd[0] = 0;
    for (int i=1; i<ProcNum; i++) {

        pSendNum[i] = RowNum * ColumnNum;
        pSendInd[i] = pSendInd[i-1]+pSendNum[i-1];
        pSendVecNum[i] = ColumnNum;
        pSendVecInd[i] = (pSendVecInd[i - 1] + pSendVecNum[i - 1]) % Size;
    }

    MPI_Scatterv(pVector, pSendVecNum, pSendVecInd, MPI_DOUBLE, pProcVector, pSendVecNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(pMatrix , pSendNum, pSendInd, MPI_DOUBLE, pProcMatrix,
                 pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] pSendNum;
    delete[] pSendInd;
    delete[] pSendVecNum;
    delete[] pSendVecInd;
}

void ResultReplication(double* pProcResult, double* pResult, int Size, int RowNum, int ColumnNum) {

    int posInRes = (ProcRank / (Size / ColumnNum)) * RowNum;
    double* allProcResult = new double[Size];
    for (int i = 0; i < Size; i++)
        allProcResult[i] = 0;
    for (int i = posInRes; i < posInRes + RowNum; i++) {
        allProcResult[i] = pProcResult[i - posInRes];
    }

    MPI_Reduce(allProcResult, pResult, Size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void ParallelResultCalculation(double* pProcMatrix, double* pProcVector, double* pProcResult, int RowNum, int ColumnNum) {
    int i, j;
    for (i = 0; i < RowNum; i++)
        pProcResult[i] = 0;
    for (i = 0; i < RowNum; i++) {
        for (j = 0; j < ColumnNum; j++)
            pProcResult[i] += pProcMatrix[i*ColumnNum+j] * pProcVector[j];
    }
}

void ProcessTermination (double* original, double* pMatrix, double* pVector, double* pProcVector, double* pResult,
                         double* pProcMatrix, double* pProcResult) {
    if (ProcRank == 0) {
        delete[] pMatrix;
        delete[] pVector;
        delete[] pResult;
        delete[] original;
    }
    delete[] pProcVector;
    delete[] pProcResult;
    delete[] pProcMatrix;
}

int main(int argc, char* argv[])
{
    double* originalMatrix; // The first argument - initial matrix
    double* pMatrix;  // Tranfsormed matrix
    double* pVector;  // The second argument - initial vector
    double* pProcVector; // The partial vector on the current process
    double* pResult;  // Result vector for matrix-vector multiplication
    int Size;		    // Sizes of initial matrix and vector
    double* pProcMatrix;   // Stripe of the matrix on the current process
    double* pProcResult; // Block of the result vector on the current process
    int RowNum;          // Number of rows in the matrix block
    int ColumnNum;          // Number of columns in the matrix block
    double Duration, Start, Finish;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);


    if (ProcRank == 0) {
        printf ("Parallel matrix-vector multiplication program\n");
    }

    // Memory allocation and data initialization
    ProcessInitialization(originalMatrix, pMatrix, pVector, pProcVector, pResult, pProcMatrix, pProcResult,
                          Size, RowNum, ColumnNum);

    Start = MPI_Wtime();

    DataDistribution(pMatrix, pProcMatrix, pVector, pProcVector, Size, RowNum, ColumnNum);
    ParallelResultCalculation(pProcMatrix, pProcVector, pProcResult, RowNum, ColumnNum);
    ResultReplication(pProcResult, pResult, Size, RowNum, ColumnNum);

    Finish = MPI_Wtime();
    Duration = Finish - Start;

    if (ProcRank == 0) {
        printf("Time of execution = %f\n", Duration);
    }

    ProcessTermination(originalMatrix, pMatrix, pVector, pProcVector, pResult, pProcMatrix, pProcResult);

    MPI_Finalize();
}