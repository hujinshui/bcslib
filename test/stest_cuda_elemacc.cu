/**
 * @file stest_cuda_elemacc.cpp
 *
 * Testing element access of cuda vector & matrix
 *
 * @author Dahua Lin
 */


#include <bcslib/cuda/cuda_vec.h>
#include <bcslib/cuda/cuda_mat.h>
#include <cstdio>

using namespace bcs;
using namespace bcs::cuda;

template<typename T>
bool vequal(int n, const T *a, const T *b)
{
	for (int i = 0; i < n; ++i)
	{
		if (a[i] != b[i]) return false;
	}
	return true;
}


template<typename T>
__global__ void my_transfer1d(int n, device_cview1d<T> a, device_view1d<T> b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		b(i) = a(i);
	}
}

template<typename T>
__global__ void my_transfer2d(int w, int h, device_cview2d<T> a, device_view2d<T> b)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h)
	{
		b(y, x) = a(y, x);
	}
}


int test_main()
{
	const int Nmax = 4096;
	static float src[Nmax];
	static float dst[Nmax];
	for (int i = 0; i < Nmax; ++i) src[i] = float(i);

	const int len = 1024;
	device_vec<float> avec(len, make_host_cptr(src));
	device_vec<float> bvec(len); bvec.set_zeros();

	int num_threads = 64;
	int num_blocks = len / num_threads;
	my_transfer1d<<<num_blocks, num_threads>>>(len, avec.cview(), bvec.view());
	for (int i = 0; i < len; ++i) dst[i] = 0.f;
	copy_memory((size_t)len, bvec.pbase().cptr(), make_host_ptr(dst));

	if (vequal(len, src, dst))
	{
		std::printf("my_transfer1d passed.\n");
	}
	else
	{
		std::printf("my_transfer1d failed!\n");
		return 1;
	}

	const int m = 48;
	const int n = 64;
	device_mat<float> amat(m, n, make_host_cptr(src));
	device_mat<float> bmat(m, n); bmat.set_zeros();

	dim3 dim_threads(8, 8);
	dim3 dim_blocks(n / dim_threads.x, m / dim_threads.y);
	my_transfer2d<<<dim_blocks, dim_threads>>>(n, m, amat.cview(), bmat.view());
	for (int i = 0; i < m*n; ++i) dst[i] = 0.f;
	copy_memory2d((size_t)m, (size_t)n, bmat.pbase().cptr(), (size_t)bmat.pitch(), make_host_ptr(dst), n * sizeof(float));

	if (vequal(m*n, src, dst))
	{
		std::printf("my_transfer2d passed.\n");
	}
	else
	{
		std::printf("my_transfer2d failed!.\n");
	}

	return 0;
}


int main(int argc, char *argv[])
{
	int ret = test_main();
	return ret;
}


