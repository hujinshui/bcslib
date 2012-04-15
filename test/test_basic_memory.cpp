/**
 * @file test_basic_memory.cpp
 *
 * Unit testing of basic memory facilities
 * 
 * @author Dahua Lin
 */

#include "bcs_test_basics.h"

#include <bcslib/base/mem_op.h>
#include <bcslib/base/block.h>
#include <bcslib/base/monitored_allocator.h>
#include <vector>

using namespace bcs;
using namespace bcs::test;

// explicit template instantiation for syntax check

template class bcs::monitored_allocator<int>;
template class bcs::Block<int, aligned_allocator<int> >;
template class bcs::Block<int, monitored_allocator<int> >;

template class bcs::ScopedBlock<int, aligned_allocator<int> >;
template class bcs::ScopedBlock<int, monitored_allocator<int> >;

typedef class bcs::Block<int, monitored_allocator<int> > blk_t;
typedef class bcs::ScopedBlock<int, monitored_allocator<int> > scblk_t;


bcs::memory_allocation_monitor bcs::global_memory_allocation_monitor;



TEST( BasicMem, ElemWise )
{
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );

	const size_t N = 5;

	int src0[N] = {1, 3, 4, 5, 2};
	const int *src = src0;

	aligned_allocator<int> alloc;

	int *p = alloc.allocate(N, BCS_NULL);
	for (size_t i = 0; i < N; ++i) p[i] = 0;

	fill_elems(N, p, 7);
	ASSERT_TRUE( elems_equal(N, p, 7) );
	p[2] = 0;
	ASSERT_FALSE( elems_equal(N, p, 7) );

	copy_elems(N, src, p);
	ASSERT_TRUE( elems_equal(N, p, src) );

	zero_elems(N, p);
	ASSERT_TRUE( elems_equal(N, p, int(0)) );

	alloc.deallocate(p, N);

	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
}

TEST( BasicMem, ElemWise2D )
{
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );

	const size_t m0 = 3;
	const size_t n0 = 5;

	const size_t m = 3;
	const size_t n = 4;

	const size_t in_dim = 3;
	const size_t out_dim = 2;

	int src0[m0 * n0] = {1, 2, 3, 4, 5,   6, 7, 8, 9, 10,   11, 12, 13, 14, 15};
	const int *src = src0;

	aligned_allocator<int> alloc;

	int *p = alloc.allocate(m * n, BCS_NULL);
	for (size_t i = 0; i < m * n; ++i) p[i] = -1;

	zero_elems_2d(in_dim, out_dim, p, n);
	int r0[m * n] = {0, 0, 0, -1,   0, 0, 0, -1,  -1, -1, -1, -1};
	ASSERT_TRUE( elems_equal(m * n, p, r0) );

	fill_elems_2d(in_dim, out_dim, p, n, 7);
	int r1[m * n] = {7, 7, 7, -1,   7, 7, 7, -1,  -1, -1, -1, -1};
	ASSERT_TRUE( elems_equal(m * n, p, r1) );
	ASSERT_TRUE( elems_equal_2d(in_dim, out_dim, p, n, 7) );
	p[6] = 100;
	ASSERT_FALSE( elems_equal_2d(in_dim, out_dim, p, n, 7) );

	copy_elems_2d(in_dim, out_dim, src, n0, p, n);
	int r2[m * n] = {1, 2, 3, -1,   6, 7, 8, -1,  -1, -1, -1, -1};
	ASSERT_TRUE( elems_equal(m * n, p, r2) );
	ASSERT_TRUE( elems_equal_2d(in_dim, out_dim, p, n, src, n0) );
	p[6] = 100;
	ASSERT_FALSE( elems_equal_2d(in_dim, out_dim, p, n, src, n0) );

	alloc.deallocate(p, m0 * n0);

	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
}




#define CHECK_MEM_PENDING(k) ASSERT_EQ(k, global_memory_allocation_monitor.num_pending_sections() )

TEST( BasicMem, BlockBase )
{
	const size_t N = 5;

	const int src[N] = {1, 3, 4, 5, 2};
	int dst[N] = {-1, -1, -1, -1, -1};
	int tmp[N] = {-1, -1, -1, -1, -1};

	typedef BlockBase<int> bbt;

	bbt B0(0, BCS_NULL);
	ASSERT_EQ(0, B0.size());
	ASSERT_EQ(0, B0.nelems());
	ASSERT_EQ(BCS_NULL, B0.pbase());
	ASSERT_EQ(BCS_NULL, B0.pend());

	bbt B1(index_t(N), dst);
	ASSERT_EQ(N, B1.size());
	ASSERT_EQ(index_t(N), B1.nelems());
	ASSERT_EQ(dst, B1.pbase());
	ASSERT_EQ(dst+N, B1.pend());

	B1.set_zeros();
	EXPECT_TRUE( elems_equal(N, B1.pbase(), 0) );

	B1.fill(7);
	EXPECT_TRUE( elems_equal(N, B1.pbase(), 7) );

	B1.copy_from(src);
	EXPECT_TRUE( elems_equal(N, B1.pbase(), src) );

	B1.copy_to(tmp);
	EXPECT_TRUE( elems_equal(N, B1.pbase(), src) );
	EXPECT_TRUE( elems_equal(N, tmp, src) );

	B0.swap(B1);
	ASSERT_EQ(dst, B0.pbase());
	ASSERT_EQ(index_t(N), B0.nelems());
	ASSERT_EQ(BCS_NULL, B1.pbase());
	ASSERT_EQ(0, B1.nelems());
}


TEST( BasicMem, Block )
{
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );

	const index_t N1 = 3;
	const index_t N2 = 5;
	const int src[N2] = {1, 3, 4, 5, 2};

	{
		// construction

		blk_t B0(0);
		ASSERT_EQ(0, B0.nelems());
		ASSERT_EQ(BCS_NULL, B0.pbase());

		CHECK_MEM_PENDING(0);

		blk_t B1(N1);
		ASSERT_EQ(N1, B1.nelems());
		ASSERT_TRUE(B1.pbase() != BCS_NULL);

		CHECK_MEM_PENDING(1);

		blk_t B2(N1, 7);
		ASSERT_EQ(N1, B2.nelems());
		ASSERT_TRUE(B2.pbase() != BCS_NULL);
		EXPECT_TRUE( elems_equal(size_t(N1), B2.pbase(), 7) );

		CHECK_MEM_PENDING(2);

		blk_t B3(N2, src);
		ASSERT_EQ(N2, B3.nelems());
		ASSERT_TRUE(B3.pbase() != BCS_NULL);
		ASSERT_TRUE(B3.pbase() != src);
		EXPECT_TRUE( elems_equal(size_t(N2), B3.pbase(), src) );

		CHECK_MEM_PENDING(3);

		blk_t B4(B3);
		ASSERT_EQ(N2, B4.nelems());
		ASSERT_TRUE(B4.pbase() != BCS_NULL);
		ASSERT_TRUE(B4.pbase() != B3.pbase());
		EXPECT_TRUE( elems_equal(size_t(N2), B4.pbase(), src) );

		CHECK_MEM_PENDING(4);

		B4 = B2;

		ASSERT_EQ(N1, B4.nelems());
		ASSERT_TRUE(B4.pbase() != BCS_NULL);
		ASSERT_TRUE(B4.pbase() != B2.pbase());
		EXPECT_TRUE( elems_equal(size_t(N1), B4.pbase(), B2.pbase()) );

		CHECK_MEM_PENDING(4);

		B4 = B0;

		ASSERT_EQ(0, B4.nelems());
		ASSERT_EQ(BCS_NULL, B4.pbase());

		CHECK_MEM_PENDING(3);

		// resize

		B3.resize(N2);

		int *pold = B3.pbase();
		ASSERT_TRUE( B3.pbase() == pold );
		ASSERT_EQ( N2, B3.nelems() );

		CHECK_MEM_PENDING(3);

		B3.resize(13);

		ASSERT_EQ( 13, B3.nelems() );
		ASSERT_TRUE( B3.pbase() != BCS_NULL );

		CHECK_MEM_PENDING(3);

		B3.resize(0);

		ASSERT_EQ( 0, B3.nelems() );
		ASSERT_TRUE( B3.pbase() == BCS_NULL );

		CHECK_MEM_PENDING(2);

		B3.resize(15);

		ASSERT_EQ( 15, B3.nelems() );
		ASSERT_TRUE( B3.pbase() != BCS_NULL );

		CHECK_MEM_PENDING(3);
	}

	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
}


TEST( BasicMem, ScopedBlock )
{
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );

	const index_t N1 = 3;
	const index_t N2 = 5;
	const int src[N2] = {1, 3, 4, 5, 2};

	{
		scblk_t B0(0);
		ASSERT_EQ(0, B0.nelems());
		ASSERT_EQ(BCS_NULL, B0.pbase());

		CHECK_MEM_PENDING(0);

		scblk_t B1(N1);
		ASSERT_EQ(N1, B1.nelems());
		ASSERT_TRUE(B1.pbase() != BCS_NULL);

		CHECK_MEM_PENDING(1);

		scblk_t B2(N1, 7);
		ASSERT_EQ(N1, B2.nelems());
		ASSERT_TRUE(B2.pbase() != BCS_NULL);
		EXPECT_TRUE( elems_equal(size_t(N1), B2.pbase(), 7) );

		CHECK_MEM_PENDING(2);

		scblk_t B3(N2, src);
		ASSERT_EQ(N2, B3.nelems());
		ASSERT_TRUE(B3.pbase() != BCS_NULL);
		ASSERT_TRUE(B3.pbase() != src);
		EXPECT_TRUE( elems_equal(size_t(N2), B3.pbase(), src) );

		CHECK_MEM_PENDING(3);
	}

	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
}


#undef CHECK_MEM_PENDING

