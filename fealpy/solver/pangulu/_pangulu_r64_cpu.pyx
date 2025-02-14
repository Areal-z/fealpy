cimport ._pangulu_base as base
from libc.stdlib cimport malloc, free
import numpy as np

# 导入特定版本头文件
cdef extern from "pangulu_r64_cpu.h":
    void pangulu_init(
        int n, long long nnz, long* csr_rowptr,
        int* csr_colidx, double* csr_value,
        base.pangulu_init_options* opts, void** handle
    )
    void pangulu_gstrf(
            base.pangulu_gstrf_options *gstrf_options, 
            void **handle);
    void pangulu_gstrs(double *rhs, 
                       base.pangulu_gstrs_options *gstrs_options, 
                       void **handle);
    void pangulu_gssv(double *rhs, 
                      base.pangulu_gstrf_options *gstrf_options, 
                      base.pangulu_gstrs_options *gstrs_options, void **handle);
    void pangulu_finalize(void **handle);

# Python 类封装
cdef class r64_cpu_solver:
    cdef void* handle
    cdef int nthread, nb

    def __cinit__(self, int nthread=4, int nb=64):
        self.nthread = nthread
        self.nb = nb
        self.handle = NULL

    def initialize(self, csr_matrix):
        # 将 SciPy CSR 矩阵转为 C 数组
        cdef long* rowptr = <long*> csr_matrix.indptr.data
        cdef int* colidx = <int*> csr_matrix.indices.data
        cdef double* data = <double*> csr_matrix.data.data

        cdef base.pangulu_init_options opts
        opts.nthread = self.nthread
        opts.nb = self.nb

        pangulu_init(
            csr_matrix.shape[0], csr_matrix.nnz,
            rowptr, colidx, data, &opts, &self.handle
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        cdef double[:] rhs_view = rhs.astype(np.float64)
        pangulu_gstrs(&rhs_view[0], NULL, &self.handle)
        return rhs.copy()

    def __dealloc__(self):
        if self.handle != NULL:
            pangulu_finalize(&self.handle)
