/**
 * @file test_cuda_base.cpp
 *
 * Unit testing of CUDA base
 * 
 * @author Dahua Lin
 */

#include <gtest/gtest.h>

#include <bcslib/cuda/cuda_base.h>

using namespace bcs;
using namespace bcs::cuda;

// template instantiation for syntax check

template class bcs::cuda::host_cptr<float>;
template class bcs::cuda::host_ptr<float>;
template class bcs::cuda::device_cptr<float>;
template class bcs::cuda::device_ptr<float>;


TEST( CudaBase, HostCPtr )
{
	host_cptr<float> p0;

	ASSERT_TRUE( p0.get() == BCS_NULL );
	ASSERT_TRUE( p0.is_null() );

	float a[10] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};

	host_cptr<float> p1(a);

	ASSERT_TRUE( p1.get() == a );
	ASSERT_FALSE( p1.is_null() );

	host_cptr<float> p2(p1 + 1);
	ASSERT_EQ( p2.get(), a + 1);

	ASSERT_EQ( (p2 - 1).get(), a);

	ASSERT_EQ( p1.get(), a );
	ASSERT_EQ( p2.get(), a + 1 );

	ASSERT_TRUE( p1 == p1 );
	ASSERT_TRUE( p1 != p2 );

	host_cptr<float> p1a(p1++);
	ASSERT_EQ( p1.get(), a + 1 );
	ASSERT_EQ( p1a.get(), a);

	p1a = ++p1;
	ASSERT_EQ( p1.get(), a + 2 );
	ASSERT_EQ( p1a.get(), a + 2);

	host_cptr<float> p1b(p1--);
	ASSERT_EQ( p1.get(), a + 1 );
	ASSERT_EQ( p1b.get(), a + 2 );

	p1b = --p1;
	ASSERT_EQ( p1.get(), a );
	ASSERT_EQ( p1b.get(), a );
}


TEST( CudaBase, HostPtr )
{
	host_ptr<float> p0;

	ASSERT_TRUE( p0.get() == BCS_NULL );
	ASSERT_TRUE( p0.is_null() );

	float a[10] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};

	host_ptr<float> p1(a);

	ASSERT_TRUE( p1.get() == a );
	ASSERT_FALSE( p1.is_null() );

	host_ptr<float> p2(p1 + 1);
	ASSERT_EQ( p2.get(), a + 1);

	ASSERT_EQ( (p2 - 1).get(), a);

	ASSERT_EQ( p1.get(), a );
	ASSERT_EQ( p2.get(), a + 1 );

	ASSERT_TRUE( p1 == p1 );
	ASSERT_TRUE( p1 != p2 );

	host_ptr<float> p1a(p1++);
	ASSERT_EQ( p1.get(), a + 1 );
	ASSERT_EQ( p1a.get(), a);

	p1a = ++p1;
	ASSERT_EQ( p1.get(), a + 2 );
	ASSERT_EQ( p1a.get(), a + 2);

	host_ptr<float> p1b(p1--);
	ASSERT_EQ( p1.get(), a + 1 );
	ASSERT_EQ( p1b.get(), a + 2 );

	p1b = --p1;
	ASSERT_EQ( p1.get(), a );
	ASSERT_EQ( p1b.get(), a );

	host_cptr<float> cp1 = p1;
	ASSERT_EQ( cp1.get(), p1.get() );
	ASSERT_TRUE( cp1 == p1 );
}


TEST( CudaBase, DeviceCPtr )
{
	device_cptr<float> p0;

	ASSERT_TRUE( p0.get() == BCS_NULL );
	ASSERT_TRUE( p0.is_null() );

	device_ptr<float> pbase = device_allocate<float>(10);
	const float *a = pbase.get();

	ASSERT_FALSE( pbase.is_null() );

	device_cptr<float> p1(pbase);

	ASSERT_EQ( p1.get(), a );
	ASSERT_FALSE( p1.is_null() );

	device_cptr<float> p2(p1 + 1);
	ASSERT_EQ( p2.get(), a + 1 );
	ASSERT_EQ( (p2 - 1).get(), a );

	ASSERT_EQ( p1.get(), a );
	ASSERT_EQ( p2.get(), a + 1 );

	device_cptr<float> p1a(p1++);
	ASSERT_EQ( p1.get(), a + 1 );
	ASSERT_EQ( p1a.get(), a);

	p1a = ++p1;
	ASSERT_EQ( p1.get(), a + 2 );
	ASSERT_EQ( p1a.get(), a + 2);

	device_cptr<float> p1b(p1--);
	ASSERT_EQ( p1.get(), a + 1 );
	ASSERT_EQ( p1b.get(), a + 2 );

	p1b = --p1;
	ASSERT_EQ( p1.get(), a );
	ASSERT_EQ( p1b.get(), a );

	device_free( pbase );
}



TEST( CudaBase, DevicePtr )
{
	device_ptr<float> p0;

	ASSERT_TRUE( p0.get() == BCS_NULL );
	ASSERT_TRUE( p0.is_null() );

	device_ptr<float> pbase = device_allocate<float>(10);
	float *a = pbase.get();

	ASSERT_FALSE( pbase.is_null() );

	device_ptr<float> p1(pbase);

	ASSERT_EQ( p1.get(), a );
	ASSERT_FALSE( p1.is_null() );

	device_ptr<float> p2(p1 + 1);
	ASSERT_EQ( p2.get(), a + 1 );
	ASSERT_EQ( (p2 - 1).get(), a );

	ASSERT_EQ( p1.get(), a );
	ASSERT_EQ( p2.get(), a + 1 );

	device_ptr<float> p1a(p1++);
	ASSERT_EQ( p1.get(), a + 1 );
	ASSERT_EQ( p1a.get(), a);

	p1a = ++p1;
	ASSERT_EQ( p1.get(), a + 2 );
	ASSERT_EQ( p1a.get(), a + 2);

	device_ptr<float> p1b(p1--);
	ASSERT_EQ( p1.get(), a + 1 );
	ASSERT_EQ( p1b.get(), a + 2 );

	p1b = --p1;
	ASSERT_EQ( p1.get(), a );
	ASSERT_EQ( p1b.get(), a );

	device_cptr<float> cp1 = p1;
	ASSERT_EQ( cp1.get(), p1.get() );
	ASSERT_TRUE( cp1 == p1 );

	device_free( pbase );
}


