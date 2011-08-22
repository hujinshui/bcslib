/**
 * @file test_cuda_mat.cpp
 *
 * Unit testing of CUDA matrices
 *
 * @author Dahua Lin
 */

#include "bcs_cuda_test_basics.h"

#include <bcslib/cuda/cuda_mat.h>
#include <cstdio>

using namespace bcs;
using namespace bcs::cuda;

// explicit template instantiation for syntax check

template class bcs::cuda::device_cview2d<float>;
template class bcs::cuda::device_view2d<float>;
template class bcs::cuda::device_mat<float>;


// auxiliary functions

template<typename T>
bool verify_devmat(const device_mat<T>& mat, int m, int n, int max_m, int max_n)
{
	if (mat.max_nrows() != max_m) return false;
	if (mat.max_ncolumns() != max_n) return false;

	if (mat.nrows() != m) return false;
	if (mat.ncolumns() != n) return false;

	if (mat.width() != n) return false;
	if (mat.height() != m) return false;

	if (mat.nelems() != m * n) return false;

	if (mat.pitch() < max_n * (int)sizeof(T)) return false;

	return true;
}

template<typename T>
bool verify_dev2d(const device_cview2d<T>& mat, int m, int n)
{
	if (mat.nrows() != m) return false;
	if (mat.ncolumns() != n) return false;

	if (mat.width() != n) return false;
	if (mat.height() != m) return false;

	if (mat.nelems() != m * n) return false;

	if (mat.pitch() < n * (int)sizeof(T)) return false;

	return true;
}

template<typename T>
bool verify_dev2d(const device_view2d<T>& mat, int m, int n)
{
	if (mat.nrows() != m) return false;
	if (mat.ncolumns() != n) return false;

	if (mat.width() != n) return false;
	if (mat.height() != m) return false;

	if (mat.nelems() != m * n) return false;

	if (mat.pitch() < n * (int)sizeof(T)) return false;

	return true;
}



// test cases

TEST( CudaMat, DeviceMat )
{
	device_mat<float> mat0;

	ASSERT_TRUE( verify_devmat(mat0, 0, 0, 0, 0) );
	ASSERT_TRUE( mat0.pbase().is_null() );

	const int m = 16;
	const int n = 32;

	static float ref0[m * n];
	for (int i = 0; i < m * n; ++i) ref0[i] = 0;

	static float ref1[m * n];
	for (int i = 0; i < m * n; ++i) ref1[i] = float(i);

	// test construction

	device_mat<float> mat1(m, n);

	ASSERT_TRUE( verify_devmat(mat1, m, n, m, n) );
	ASSERT_FALSE( mat1.pbase().is_null() );

	mat1.set_zeros();
	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat1.pbase().cptr(), mat1.pitch(), ref0) );

	device_mat<float> mat1e(m, n, 2*m, 2*n);

	ASSERT_TRUE( verify_devmat(mat1e, m, n, 2*m, 2*n) );
	ASSERT_FALSE( mat1e.pbase().is_null() );

	mat1e.set_zeros();
	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat1e.pbase().cptr(), mat1e.pitch(), ref0) );

	device_mat<float> mat2(m, n, make_host_cptr(ref1));

	ASSERT_TRUE( verify_devmat(mat2, m, n, m, n) );
	ASSERT_FALSE( mat2.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat2.pbase().cptr(), mat2.pitch(), ref1) );

	device_mat<float> mat2e(m, n, make_host_cptr(ref1), 2*m, 2*n);

	ASSERT_TRUE( verify_devmat(mat2e, m, n, 2*m, 2*n) );
	ASSERT_FALSE( mat2e.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat2e.pbase().cptr(), mat2e.pitch(), ref1) );

	device_mat<float> mat3(m, n, mat2.pbase().cptr(), mat2.pitch());

	ASSERT_TRUE( verify_devmat(mat3, m, n, m, n) );
	ASSERT_FALSE( mat3.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat3.pbase().cptr(), mat3.pitch(), ref1) );

	device_mat<float> mat3e(m, n, mat2.pbase().cptr(), mat2.pitch(), 2*m, 2*n);

	ASSERT_TRUE( verify_devmat(mat3e, m, n, 2*m, 2*n) );
	ASSERT_FALSE( mat3e.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat3e.pbase().cptr(), mat3e.pitch(), ref1) );

	device_mat<float> mat4(mat2);

	ASSERT_TRUE( verify_devmat(mat4, m, n, m, n) );
	ASSERT_FALSE( mat4.pbase().is_null() );
	ASSERT_NE( mat4.pbase(), mat2.pbase() );

	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat4.pbase().cptr(), mat4.pitch(), ref1) );

	device_mat<float> mat4e(mat2, 2*m, 2*n);

	ASSERT_TRUE( verify_devmat(mat4e, m, n, 2*m, 2*n) );
	ASSERT_FALSE( mat4e.pbase().is_null() );
	ASSERT_NE( mat4e.pbase(), mat2.pbase() );

	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat4e.pbase().cptr(), mat4e.pitch(), ref1) );


	// test assignment

	device_mat<float> mat_a;
	mat_a = mat1;

	ASSERT_TRUE( verify_devmat(mat_a, m, n, m, n) );
	ASSERT_FALSE( mat_a.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat_a.pbase().cptr(), mat_a.pitch(), ref0) );

	device_ptr<float> mat_a_p = mat_a.pbase();

	mat_a = mat2e;

	ASSERT_TRUE( verify_devmat(mat_a, m, n, m, n) );
	ASSERT_FALSE( mat_a.pbase().is_null() );

	ASSERT_EQ( mat_a.pbase(), mat_a_p );

	EXPECT_TRUE( verify_device_mem2d<float>((size_t)m, (size_t)n,
			mat_a.pbase().cptr(), mat_a.pitch(), ref1) );

	mat_a = mat0;

	ASSERT_TRUE( verify_devmat(mat_a, 0, 0, m, n) );

	// test swap

	device_mat<float> u1(6, 8, make_host_cptr(ref0), 12, 16);
	device_mat<float> u2(4, 12, make_host_cptr(ref1), 8, 20);

	device_ptr<float> u1p = u1.pbase();
	device_ptr<float> u2p = u2.pbase();

	int u1pitch = u1.pitch();
	int u2pitch = u2.pitch();

	ASSERT_TRUE( verify_devmat(u1, 6, 8, 12, 16) );
	ASSERT_TRUE( verify_devmat(u2, 4, 12, 8, 20) );

	EXPECT_TRUE( verify_device_mem2d<float>(6, 8, u1.pbase(), (size_t)u1.pitch(), ref0) );
	EXPECT_TRUE( verify_device_mem2d<float>(4, 12, u2.pbase(), (size_t)u2.pitch(), ref1) );

	u1.swap(u2);

	ASSERT_TRUE( verify_devmat(u1, 4, 12, 8, 20) );
	ASSERT_TRUE( verify_devmat(u2, 6, 8, 12, 16) );

	ASSERT_EQ( u1.pitch(), u2pitch );
	ASSERT_EQ( u2.pitch(), u1pitch );

	ASSERT_EQ( u1.pbase(), u2p );
	ASSERT_EQ( u2.pbase(), u1p );

	EXPECT_TRUE( verify_device_mem2d<float>(4, 12, u1.pbase(), (size_t)u1.pitch(), ref1) );
	EXPECT_TRUE( verify_device_mem2d<float>(6, 8, u2.pbase(), (size_t)u2.pitch(), ref0) );

}


TEST( CudaMat, DeviceView2D )
{
	device_mat<float> mat0;

	const int m = 16;
	const int n = 32;
	device_mat<float> mat1(m, n);

	device_cview2d<float> cv0 = mat0.cview();

	ASSERT_TRUE( verify_dev2d(cv0, 0, 0) );
	ASSERT_EQ( cv0.pitch(), 0 );
	ASSERT_TRUE( cv0.pbase().is_null() );

	device_cview2d<float> cv1 = mat1.cview();

	ASSERT_TRUE( verify_dev2d(cv1, m, n)  );
	ASSERT_EQ( cv1.pitch(), mat1.pitch() );
	ASSERT_EQ( cv1.pbase(), mat1.pbase().cptr() );

	device_cview2d<float> cv2( mat1.pbase(), m, n, mat1.pitch() );

	ASSERT_TRUE( verify_dev2d(cv2, m, n)  );
	ASSERT_EQ( cv2.pitch(), mat1.pitch() );
	ASSERT_EQ( cv2.pbase(), mat1.pbase().cptr() );

	device_view2d<float> v0 = mat0.view();

	ASSERT_TRUE( verify_dev2d(v0, 0, 0) );
	ASSERT_EQ( v0.pitch(), 0 );
	ASSERT_TRUE( v0.pbase().is_null() );

	device_view2d<float> v1 = mat1.view();

	ASSERT_TRUE( verify_dev2d(v1, m, n)  );
	ASSERT_EQ( v1.pitch(), mat1.pitch() );
	ASSERT_EQ( v1.pbase(), mat1.pbase() );

	device_view2d<float> v2( mat1.pbase(), m, n, mat1.pitch() );

	ASSERT_TRUE( verify_dev2d(v2, m, n)  );
	ASSERT_EQ( v2.pitch(), mat1.pitch() );
	ASSERT_EQ( v2.pbase(), mat1.pbase() );

	device_cview2d<float> cc1 = v1;

	ASSERT_TRUE( verify_dev2d(cc1, m, n)  );
	ASSERT_EQ( cc1.pitch(), mat1.pitch() );
	ASSERT_EQ( cc1.pbase(), mat1.pbase().cptr() );

	cc1 = v1.cview();

	ASSERT_TRUE( verify_dev2d(cc1, m, n)  );
	ASSERT_EQ( cc1.pitch(), mat1.pitch() );
	ASSERT_EQ( cc1.pbase(), mat1.pbase().cptr() );
}


TEST( CudaMat, BlockViews )
{
	const int m = 16;
	const int n = 32;
	device_mat<float> mat0(m, n);
	device_cview2d<float> cview0 = mat0.cview();
	device_view2d<float> view0 = mat0.view();
	int pitch = mat0.pitch();

	const float* cp0 = mat0.pbase().get();
	float* p0 = mat0.pbase().get();

	const int i = 3;
	const int j = 5;

	device_cview2d<float> cb1 = cview0.cblock(i, j, 8, 12);
	ASSERT_EQ( cb1.nrows(), 8 );
	ASSERT_EQ( cb1.ncolumns(), 12 );
	ASSERT_EQ( cb1.pitch(), pitch );
	ASSERT_EQ( cb1.pbase(),  make_device_cptr((const float*)((const char*)cp0 + pitch *  i) + j) );

	cb1 = view0.cblock(i, j, 8, 12);
	ASSERT_EQ( cb1.nrows(), 8 );
	ASSERT_EQ( cb1.ncolumns(), 12 );
	ASSERT_EQ( cb1.pitch(), pitch );
	ASSERT_EQ( cb1.pbase(),  make_device_cptr((const float*)((const char*)cp0 + pitch *  i) + j) );

	cb1 = mat0.cblock(i, j, 8, 12);
	ASSERT_EQ( cb1.nrows(), 8 );
	ASSERT_EQ( cb1.ncolumns(), 12 );
	ASSERT_EQ( cb1.pitch(), pitch );
	ASSERT_EQ( cb1.pbase(),  make_device_cptr((const float*)((const char*)cp0 + pitch *  i) + j) );

	device_view2d<float> b1 = view0.block(i, j, 8, 12);
	ASSERT_EQ( b1.nrows(), 8 );
	ASSERT_EQ( b1.ncolumns(), 12 );
	ASSERT_EQ( b1.pitch(), pitch );
	ASSERT_EQ( b1.pbase(),  make_device_ptr((float*)((char*)p0 + pitch *  i) + j) );

	b1 = mat0.block(i, j, 8, 12);
	ASSERT_EQ( b1.nrows(), 8 );
	ASSERT_EQ( b1.ncolumns(), 12 );
	ASSERT_EQ( b1.pitch(), pitch );
	ASSERT_EQ( b1.pbase(),  make_device_ptr((float*)((char*)p0 + pitch *  i) + j) );
}


TEST( CudaMat, CopyViewsRowMajor )
{
	const int m = 16;
	const int n = 32;

	static float src[m * n];
	for (int i = 0; i < m * n; ++i) src[i] = float(i);

	static float dst[m * n];
	for (int i = 0; i < m * n; ++i) dst[i] = float(0);

	caview2d<float, row_major_t> a0(src, m, n);
	ASSERT_EQ( a0.nrows(), m );
	ASSERT_EQ( a0.ncolumns(), n );

	device_mat<float> v0(m, n);
	ASSERT_EQ( v0.nrows(), m );
	ASSERT_EQ( v0.ncolumns(), n );

	device_mat<float> v1(m, n);
	ASSERT_EQ( v1.nrows(), m );
	ASSERT_EQ( v1.ncolumns(), n );

	aview2d<float, row_major_t> a1(dst, m, n);
	ASSERT_EQ( a1.nrows(), m );
	ASSERT_EQ( a1.ncolumns(), n );

	copy(a0, v0.view());
	ASSERT_TRUE( verify_device_mem2d<float>(m, n, v0.pbase(), v0.pitch(), src) );

	copy(v0.cview(), v1.view());
	ASSERT_TRUE( verify_device_mem2d<float>(m, n, v1.pbase(), v1.pitch(), src) );

	copy(v1.cview(), a1);
	ASSERT_TRUE( verify_host_mem1d<float>(m * n, make_host_cptr(dst), src) );

}


TEST( CudaMat, CopyViewsColumnMajor )
{
	const int m = 16;
	const int n = 32;

	static float src[m * n];
	for (int i = 0; i < m * n; ++i) src[i] = float(i);

	static float dst[m * n];
	for (int i = 0; i < m * n; ++i) dst[i] = float(0);

	caview2d<float, column_major_t> a0(src, n, m);
	ASSERT_EQ( a0.nrows(), n );
	ASSERT_EQ( a0.ncolumns(), m );

	device_mat<float> v0(m, n);
	ASSERT_EQ( v0.nrows(), m );
	ASSERT_EQ( v0.ncolumns(), n );

	device_mat<float> v1(m, n);
	ASSERT_EQ( v1.nrows(), m );
	ASSERT_EQ( v1.ncolumns(), n );

	aview2d<float, column_major_t> a1(dst, n, m);
	ASSERT_EQ( a1.nrows(), n );
	ASSERT_EQ( a1.ncolumns(), m );

	trans_copy(a0, v0.view());
	ASSERT_TRUE( verify_device_mem2d<float>(m, n, v0.pbase(), v0.pitch(), src) );

	copy(v0.cview(), v1.view());
	ASSERT_TRUE( verify_device_mem2d<float>(m, n, v1.pbase(), v1.pitch(), src) );

	trans_copy(v1.cview(), a1);
	ASSERT_TRUE( verify_host_mem1d<float>(m * n, make_host_cptr(dst), src) );

}



