#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    char matrix_name[200], bin_matrix_name[200], vector_name[200], bin_vector_name[200];
    FILE *matrix_file, *bin_matrix_file, *vector_file, *bin_vector_file;
    double *mat_vals, *vec_vals;
    int dims[2], row, column, rows;
    
    printf("For %s\n", argv[1]);
    printf("Creating binary matrix file...\n");
    sprintf(matrix_name, "%s.mat", argv[1]);
    sprintf(bin_matrix_name, "%s.mat.bin", argv[1]);
    matrix_file = fopen(matrix_name, "r");
    bin_matrix_file = fopen(bin_matrix_name, "wb");
    
    fscanf(matrix_file, "%d %d", dims, dims + 1);
    fwrite(dims, sizeof(int), 2, bin_matrix_file);
    
    mat_vals = (double *) malloc(dims[0] * dims[1] * sizeof(double));
    for (row = 0; row < dims[0]; row++) {
        for (column = 0; column < dims[1]; column++) {
            fscanf(matrix_file, "%lf", &mat_vals[row * dims[1] + column]);
        }
    }
    fwrite(mat_vals, sizeof(double), dims[0] * dims[1], bin_matrix_file);
    
    free(mat_vals);
    fclose(matrix_file);
    fclose(bin_matrix_file);
    
    printf("Creating binary vector file...\n");
    sprintf(vector_name, "%s.vec", argv[1]);
    sprintf(bin_vector_name, "%s.vec.bin", argv[1]);
    vector_file = fopen(vector_name, "r");
    bin_vector_file = fopen(bin_vector_name, "wb");
    
    fscanf(vector_file, "%d", &rows);
    fwrite(&rows, sizeof(int), 1, bin_vector_file);
    
    vec_vals = (double *) malloc(rows * sizeof(double));
    for (row = 0; row < rows; row++) {
        fscanf(vector_file, "%lf", &vec_vals[row]);
    }
    fwrite(vec_vals, sizeof(double), rows, bin_vector_file);
    
    free(vec_vals);
    fclose(vector_file);
    fclose(bin_vector_file);
    return 0;
}

