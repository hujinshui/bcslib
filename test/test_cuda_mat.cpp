/**
 * @file test_cuda_mat.cpp
 *
 * Unit testing of CUDA matrices
 *
 * @author Dahua Lin
 */

#include "bcs_cuda_test_basics.h"

#include <bcslib/cuda/cuda_mat.h>

using namespace bcs;
using namespace bcs::cuda;

// explicit template instantiation for syntax check

template class bcs::cuda::device_cview2d<float>;
template class bcs::cuda::device_view2d<float>;
template class bcs::cuda::device_mat<float>;


