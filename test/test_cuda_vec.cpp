/**
 * @file test_cuda_vec.cpp
 *
 * Unit testing of cuda vec
 * 
 * @author Dahua Lin
 */

#include <gtest/gtest.h>

#include <bcslib/cuda/cuda_vec.h>

using namespace bcs;
using namespace bcs::cuda;

// explicit template instantiation for syntax check

template class bcs::cuda::device_cview1d<float>;
template class bcs::cuda::device_view1d<float>;
template class bcs::cuda::device_vec<float>;

