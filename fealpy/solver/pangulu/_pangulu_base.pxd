# base.pxd
cdef extern from "pangulu_interface_common.h":
    ctypedef struct pangulu_init_options:
        int nthread
        int nb

    ctypedef struct pangulu_gstrf_options:
        pass

    ctypedef struct pangulu_gstrs_options:
        pass

    # 通用函数指针类型
    ctypedef void (*InitFunc)(
        int n, long long nnz, long* csr_rowptr,
        int* csr_colidx, void* csr_value,
        pangulu_init_options* opts, void** handle
    )
