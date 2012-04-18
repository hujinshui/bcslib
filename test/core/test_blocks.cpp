/*
 * @file test_blocks.cpp
 *
 * Unit testing of block classes
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/core.h>
#include <bcslib/utils/monitored_allocator.h>

using namespace bcs;


// explicit template instantiation for syntax check

template class bcs::monitored_allocator<int>;
template class bcs::block<int, aligned_allocator<int> >;
template class bcs::block<int, monitored_allocator<int> >;

template class bcs::scoped_block<int, aligned_allocator<int> >;
template class bcs::scoped_block<int, monitored_allocator<int> >;

template class bcs::static_block<int, 2>;
template class bcs::static_block<int, 4>;
template class bcs::static_block<int, 6>;

// typedefs

typedef class bcs::block<int, monitored_allocator<int> > blk_t;
typedef class bcs::scoped_block<int, monitored_allocator<int> > scblk_t;

bcs::memory_allocation_monitor bcs::global_memory_allocation_monitor;



#define CHECK_MEM_PENDING(k) ASSERT_EQ(k, global_memory_allocation_monitor.num_pending_sections() )

TEST( Blocks, Block )
{
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );

	const index_t N1 = 3;
	const index_t N2 = 5;
	const int src[N2] = {1, 3, 4, 5, 2};

	{
		// construction

		blk_t B0(0);
		ASSERT_EQ(0, B0.nelems());
		ASSERT_EQ(BCS_NULL, B0.ptr_begin());

		CHECK_MEM_PENDING(0);

		blk_t B1(N1);
		ASSERT_EQ(N1, B1.nelems());
		ASSERT_TRUE(B1.ptr_begin() != BCS_NULL);

		CHECK_MEM_PENDING(1);

		blk_t B2(N1, 7);
		ASSERT_EQ(N1, B2.nelems());
		ASSERT_TRUE(B2.ptr_begin() != BCS_NULL);
		EXPECT_TRUE( elems_equal(size_t(N1), B2.ptr_begin(), 7) );

		CHECK_MEM_PENDING(2);

		blk_t B3(N2, src);
		ASSERT_EQ(N2, B3.nelems());
		ASSERT_TRUE(B3.ptr_begin() != BCS_NULL);
		ASSERT_TRUE(B3.ptr_begin() != src);
		EXPECT_TRUE( elems_equal(size_t(N2), B3.ptr_begin(), src) );

		CHECK_MEM_PENDING(3);

		blk_t B4(B3);
		ASSERT_EQ(N2, B4.nelems());
		ASSERT_TRUE(B4.ptr_begin() != BCS_NULL);
		ASSERT_TRUE(B4.ptr_begin() != B3.ptr_begin());
		EXPECT_TRUE( elems_equal(size_t(N2), B4.ptr_begin(), src) );

		CHECK_MEM_PENDING(4);

		B4 = B2;

		ASSERT_EQ(N1, B4.nelems());
		ASSERT_TRUE(B4.ptr_begin() != BCS_NULL);
		ASSERT_TRUE(B4.ptr_begin() != B2.ptr_begin());
		EXPECT_TRUE( elems_equal(size_t(N1), B4.ptr_begin(), B2.ptr_begin()) );

		CHECK_MEM_PENDING(4);

		B4 = B0;

		ASSERT_EQ(0, B4.nelems());
		ASSERT_EQ(BCS_NULL, B4.ptr_begin());

		CHECK_MEM_PENDING(3);

		// resize

		B3.resize(N2);

		int *pold = B3.ptr_begin();
		ASSERT_TRUE( B3.ptr_begin() == pold );
		ASSERT_EQ( N2, B3.nelems() );

		CHECK_MEM_PENDING(3);

		B3.resize(13);

		ASSERT_EQ( 13, B3.nelems() );
		ASSERT_TRUE( B3.ptr_begin() != BCS_NULL );

		CHECK_MEM_PENDING(3);

		B3.resize(0);

		ASSERT_EQ( 0, B3.nelems() );
		ASSERT_TRUE( B3.ptr_begin() == BCS_NULL );

		CHECK_MEM_PENDING(2);

		B3.resize(15);

		ASSERT_EQ( 15, B3.nelems() );
		ASSERT_TRUE( B3.ptr_begin() != BCS_NULL );

		CHECK_MEM_PENDING(3);
	}

	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
}


TEST( Blocks, BlockMemOp )
{
	const index_t N = 9;

	const int src[N] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
	int dst[N] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

	blk_t B1(N, 0);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), 0) );
	const int *pb = B1.ptr_begin();

	fill(B1, 2);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( B1.ptr_begin() == pb );
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), 2) );

	copy_from(B1, src);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( B1.ptr_begin() == pb );
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), src) );

	copy_to(B1, dst);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( elems_equal(size_t(N), dst, src) );

	zero(B1);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), 0) );

}



TEST( Blocks, ScopedBlock )
{
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );

	const index_t N1 = 3;
	const index_t N2 = 5;
	const int src[N2] = {1, 3, 4, 5, 2};

	{
		scblk_t B0(0);
		ASSERT_EQ(0, B0.nelems());
		ASSERT_EQ(BCS_NULL, B0.ptr_begin());

		CHECK_MEM_PENDING(0);

		scblk_t B1(N1);
		ASSERT_EQ(N1, B1.nelems());
		ASSERT_TRUE(B1.ptr_begin() != BCS_NULL);

		CHECK_MEM_PENDING(1);

		scblk_t B2(N1, 7);
		ASSERT_EQ(N1, B2.nelems());
		ASSERT_TRUE(B2.ptr_begin() != BCS_NULL);
		EXPECT_TRUE( elems_equal(size_t(N1), B2.ptr_begin(), 7) );

		CHECK_MEM_PENDING(2);

		scblk_t B3(N2, src);
		ASSERT_EQ(N2, B3.nelems());
		ASSERT_TRUE(B3.ptr_begin() != BCS_NULL);
		ASSERT_TRUE(B3.ptr_begin() != src);
		EXPECT_TRUE( elems_equal(size_t(N2), B3.ptr_begin(), src) );

		CHECK_MEM_PENDING(3);
	}

	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
}


TEST( Blocks, ScopedBlockMemOp )
{
	const index_t N = 9;

	const int src[N] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
	int dst[N] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

	scblk_t B1(N, 0);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), 0) );
	const int *pb = B1.ptr_begin();

	fill(B1, 2);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( B1.ptr_begin() == pb );
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), 2) );

	copy_from(B1, src);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( B1.ptr_begin() == pb );
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), src) );

	copy_to(B1, dst);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( elems_equal(size_t(N), dst, src) );

	zero(B1);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), 0) );

}


TEST( Blocks, StaticBlock )
{
	const index_t N = 4;
	typedef static_block<int, N> stblk_t;

	int src[N];
	for (int i = 0; i < N; ++i) src[i] = i + 1;

	stblk_t B0;
	ASSERT_EQ(N, B0.nelems());
	ASSERT_TRUE(B0.ptr_begin() != BCS_NULL);

	stblk_t B2(7);
	ASSERT_EQ(N, B2.nelems());
	ASSERT_TRUE(B2.ptr_begin() != BCS_NULL);
	EXPECT_TRUE( elems_equal(size_t(N), B2.ptr_begin(), 7) );

	stblk_t B3(src);
	ASSERT_EQ(N, B3.nelems());
	ASSERT_TRUE(B3.ptr_begin() != BCS_NULL);
	ASSERT_TRUE(B3.ptr_begin() != src);
	EXPECT_TRUE( elems_equal(size_t(N), B3.ptr_begin(), src) );

}


TEST( Blocks, StaticBlockMemOp )
{
	const index_t N = 4;
	typedef static_block<int, N> stblk_t;

	const int src[N] = {9, 8, 7, 6};
	int dst[N] = {0, 0, 0, 0};

	stblk_t B1(0);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), 0) );
	const int *pb = B1.ptr_begin();

	fill(B1, 2);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( B1.ptr_begin() == pb );
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), 2) );

	copy_from(B1, src);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( B1.ptr_begin() == pb );
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), src) );

	copy_to(B1, dst);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( elems_equal(size_t(N), dst, src) );

	zero(B1);
	ASSERT_EQ(N, B1.nelems());
	ASSERT_TRUE( elems_equal(size_t(N), B1.ptr_begin(), 0) );

}


TEST( Blocks, DynamicStaticInteraction )
{
	const index_t N = 4;
	typedef static_block<int, N> stblk_t;

	const int src[N] = {8, 7, 6, 5};
	const int src2[N] = {8, 7, 6, 3};

	blk_t B1(N, src);
	stblk_t B2(src2);

	ASSERT_TRUE( is_equal(B1, B1) );
	ASSERT_TRUE( is_equal(B2, B2) );
	ASSERT_FALSE( is_equal(B1, B2) );
	ASSERT_FALSE( is_equal(B2, B1) );

	B1 = B2;
	ASSERT_TRUE( is_equal(B1, B2) );

	fill(B2, 0);
	ASSERT_FALSE( is_equal(B1, B2) );

	B2 = B1;
	ASSERT_TRUE( is_equal(B1, B2) );
}




#undef CHECK_MEM_PENDING

