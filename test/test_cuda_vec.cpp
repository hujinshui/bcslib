/**
 * @file test_cuda_vec.cpp
 *
 * Unit testing of cuda vec
 * 
 * @author Dahua Lin
 */

#include "bcs_cuda_test_basics.h"

#include <bcslib/cuda/cuda_vec.h>

using namespace bcs;
using namespace bcs::cuda;

// explicit template instantiation for syntax check

template class bcs::cuda::device_cview1d<float>;
template class bcs::cuda::device_view1d<float>;
template class bcs::cuda::device_vec<float>;

TEST( CudaVec, DeviceVec )
{
	device_vec<float> v0;

	ASSERT_EQ( v0.capacity(), 0 );
	ASSERT_EQ( v0.nelems(), 0 );
	ASSERT_EQ( v0.length(), 0 );
	ASSERT_TRUE( v0.pbase().is_null() );

	const int N = 128;

	static float ref0[N];
	for (int i = 0; i < N; ++i) ref0[i] = 0;

	static float ref1[N];
	for (int i = 0; i < N; ++i) ref1[i] = float(i);

	// test construction

	device_vec<float> v1(N);

	ASSERT_EQ( v1.capacity(), N );
	ASSERT_EQ( v1.nelems(), N );
	ASSERT_EQ( v1.length(), N );
	ASSERT_FALSE( v1.pbase().is_null() );

	v1.set_zeros();
	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, v1.pbase(), ref0) );

	device_vec<float> v1e(N, 2 * N);

	ASSERT_EQ( v1e.capacity(), 2 * N );
	ASSERT_EQ( v1e.nelems(), N );
	ASSERT_EQ( v1e.length(), N );
	ASSERT_FALSE( v1e.pbase().is_null() );

	v1e.set_zeros();
	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, v1e.pbase(), ref0) );

	device_vec<float> v2(N, make_host_cptr(ref1));

	ASSERT_EQ( v2.capacity(), N );
	ASSERT_EQ( v2.nelems(), N );
	ASSERT_EQ( v2.length(), N );
	ASSERT_FALSE( v2.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, v2.pbase(), ref1) );

	device_vec<float> v2e(N, make_host_cptr(ref1), 2 * N);

	ASSERT_EQ( v2e.capacity(), 2 * N );
	ASSERT_EQ( v2e.nelems(), N );
	ASSERT_EQ( v2e.length(), N );
	ASSERT_FALSE( v2e.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, v2e.pbase(), ref1) );

	device_vec<float> v3(N, v2.pbase().cptr());

	ASSERT_EQ( v3.capacity(), N );
	ASSERT_EQ( v3.nelems(), N );
	ASSERT_EQ( v3.length(), N );
	ASSERT_FALSE( v3.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, v3.pbase(), ref1) );

	device_vec<float> v3e(N, v2.pbase().cptr(), 2 * N);

	ASSERT_EQ( v3e.capacity(), 2 * N );
	ASSERT_EQ( v3e.nelems(), N );
	ASSERT_EQ( v3e.length(), N );
	ASSERT_FALSE( v3e.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, v3e.pbase(), ref1) );

	device_vec<float> v4(v2);

	ASSERT_EQ( v4.capacity(), N );
	ASSERT_EQ( v4.nelems(), N );
	ASSERT_EQ( v4.length(), N );
	ASSERT_FALSE( v4.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, v4.pbase(), ref1) );

	device_vec<float> v4e(v2, 2 * N);

	ASSERT_EQ( v4e.capacity(), 2 * N );
	ASSERT_EQ( v4e.nelems(), N );
	ASSERT_EQ( v4e.length(), N );
	ASSERT_FALSE( v4e.pbase().is_null() );

	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, v4e.pbase(), ref1) );

	// test assignment

	device_vec<float> va;
	va = v1;

	ASSERT_EQ( va.nelems(), N );
	ASSERT_EQ( va.capacity(), N );
	ASSERT_EQ( va.length(), N );
	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, va.pbase(), ref0) );

	device_ptr<float> va_p = va.pbase();

	va = v2e;

	ASSERT_EQ( va.nelems(), N );
	ASSERT_EQ( va.capacity(), N );
	ASSERT_EQ( va.length(), N );
	ASSERT_EQ( va.pbase(), va_p );

	EXPECT_TRUE( verify_device_mem1d<float>((size_t)N, va.pbase(), ref1) );

	va = v0;

	ASSERT_EQ( va.nelems(), 0 );
	ASSERT_EQ( va.capacity(), N );
	ASSERT_EQ( va.pbase(), va_p );

	// test swap

	device_vec<float> u1(6, v1.pbase(), 10);
	device_vec<float> u2(8, v2.pbase(), 12);

	device_ptr<float> u1p = u1.pbase();
	device_ptr<float> u2p = u2.pbase();

	ASSERT_EQ( u1.capacity(), 10 );
	ASSERT_EQ( u1.nelems(), 6 );
	ASSERT_EQ( u2.capacity(), 12 );
	ASSERT_EQ( u2.nelems(), 8 );

	EXPECT_TRUE( verify_device_mem1d<float>(6, u1.pbase(), ref0) );
	EXPECT_TRUE( verify_device_mem1d<float>(8, u2.pbase(), ref1) );

	u1.swap(u2);

	ASSERT_EQ( u1.capacity(), 12 );
	ASSERT_EQ( u1.nelems(), 8 );
	ASSERT_EQ( u2.capacity(), 10 );
	ASSERT_EQ( u2.nelems(), 6 );

	ASSERT_EQ( u1.pbase(), u2p );
	ASSERT_EQ( u2.pbase(), u1p );

	EXPECT_TRUE( verify_device_mem1d<float>(8, u1.pbase(), ref1) );
	EXPECT_TRUE( verify_device_mem1d<float>(6, u2.pbase(), ref0) );


}


TEST( CudaVec, DeviceView1D )
{
	device_vec<float> vec0;

	const int N = 10;
	device_vec<float> vec1(N);

	device_cview1d<float> cv0 = vec0.cview();

	ASSERT_EQ( cv0.nelems(), 0 );
	ASSERT_EQ( cv0.length(), 0 );
	ASSERT_TRUE( cv0.pbase().is_null() );

	device_cview1d<float> cv1 = vec1.cview();

	ASSERT_EQ( cv1.nelems(), N );
	ASSERT_EQ( cv1.length(), N );
	ASSERT_EQ( cv1.pbase(), vec1.pbase().cptr() );

	device_cview1d<float> cv2(vec1.pbase(), N);

	ASSERT_EQ( cv2.nelems(), N );
	ASSERT_EQ( cv2.length(), N );
	ASSERT_EQ( cv2.pbase(), vec1.pbase().cptr() );

	device_view1d<float> v0 = vec0.view();

	ASSERT_EQ( v0.nelems(), 0 );
	ASSERT_EQ( v0.length(), 0 );
	ASSERT_TRUE( v0.pbase().is_null() );

	device_view1d<float> v1 = vec1.view();

	ASSERT_EQ( v1.nelems(), N );
	ASSERT_EQ( v1.length(), N );
	ASSERT_EQ( v1.pbase(), vec1.pbase() );

	device_view1d<float> v2(vec1.pbase(), N);

	ASSERT_EQ( v2.nelems(), N );
	ASSERT_EQ( v2.length(), N );
	ASSERT_EQ( v2.pbase(), vec1.pbase() );

	device_cview1d<float> cc1 = v1;

	ASSERT_EQ( cc1.nelems(), N );
	ASSERT_EQ( cc1.length(), N );
	ASSERT_EQ( cc1.pbase(), vec1.pbase().cptr() );

	cc1 = v1.cview();

	ASSERT_EQ( cc1.nelems(), N );
	ASSERT_EQ( cc1.length(), N );
	ASSERT_EQ( cc1.pbase(), vec1.pbase().cptr() );
}


TEST( CudaVec, BlockViews )
{
	const int N = 20;
	device_vec<float> vec0(N);
	device_cview1d<float> cview0 = vec0.cview();
	device_view1d<float> view0 = vec0.view();

	device_cptr<float> cp0 = vec0.pbase();
	device_ptr<float> p0 = vec0.pbase();

	device_cview1d<float> cb1 = cview0.cblock(3, 5);
	ASSERT_EQ( cb1.nelems(), 5 );
	ASSERT_EQ( cb1.length(), 5 );
	ASSERT_EQ( cb1.pbase(), cp0 + 3 );

	device_view1d<float> b1 = view0.block(3, 5);
	ASSERT_EQ( b1.nelems(), 5 );
	ASSERT_EQ( b1.length(), 5 );
	ASSERT_EQ( b1.pbase(), p0 + 3 );

	b1 = vec0.block(3, 5);
	ASSERT_EQ( b1.nelems(), 5 );
	ASSERT_EQ( b1.length(), 5 );
	ASSERT_EQ( b1.pbase(), p0 + 3 );
}

