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
template class bcs::cref_blk_t<int>;
template class bcs::ref_blk_t<int>;
template class bcs::copy_blk_t<int>;
template class bcs::const_block<int, aligned_allocator<int> >;
template class bcs::block<int, aligned_allocator<int> >;
template class bcs::const_block<int, monitored_allocator<int> >;
template class bcs::block<int, monitored_allocator<int> >;

typedef class bcs::const_block<int, monitored_allocator<int> > cblk_t;
typedef class bcs::block<int, monitored_allocator<int> > blk_t;


bcs::memory_allocation_monitor bcs::global_memory_allocation_monitor;

class MyInt
{
public:
	explicit MyInt(int v) : p_v(new int(v)) { ++ s_nobjs; }

	~MyInt() { delete p_v; -- s_nobjs; }

	MyInt(const MyInt& r) : p_v(new int(r.value())) { ++ s_nobjs; }

	MyInt& operator = (const MyInt& r)
	{
		if (this != &r)
		{
			int *tmp = new int(r.value());
			delete p_v;
			p_v = tmp;
		}
		return *this;
	}

	int value() const
	{
		return *p_v;
	}

	bool operator == (const MyInt& r) const
	{
		return value() == r.value();
	}

	bool operator != (const MyInt& r) const
	{
		return value() != r.value();
	}

private:
	int *p_v;

public:
	static size_t num_objects()
	{
		return s_nobjs;
	}

private:
	static size_t s_nobjs;

};

size_t MyInt::s_nobjs = 0;


template<typename T, class Allocator>
inline static bool test_block(const const_block<T, Allocator>& blk, bool own, size_t n, const T *pbase = BCS_NULL)
{
	return blk.own_memory() == own && blk.nelems() == n && (blk.pend() == blk.pbase() + n) &&
			(pbase == BCS_NULL || blk.pbase() == pbase);
}

template<typename T, class Allocator>
inline static bool test_block(const block<T, Allocator>& blk, bool own, size_t n, const T *pbase = BCS_NULL)
{
	return blk.own_memory() == own && blk.nelems() == n && (blk.pend() == blk.pbase() + n) &&
			(pbase == BCS_NULL || blk.pbase() == pbase);
}


TEST( BasicMem, ElemWiseOperations )
{
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );

	const size_t N = 5;

	int src0[N] = {1, 3, 4, 5, 2};
	const int *src = src0;

	aligned_allocator<int> alloc;

	int *p = alloc.allocate(N, BCS_NULL);
	for (size_t i = 0; i < N; ++i) p[i] = 0;

	mem<int>::fill(p, N, 7);
	EXPECT_TRUE( mem<int>::equal(p, 7, N) );
	p[2] = 0;
	EXPECT_FALSE( mem<int>::equal(p, 7, N) );

	mem<int>::copy(src, p, N);
	EXPECT_TRUE( mem<int>::equal(p, src, N) );

	mem<int>::zero(p, N);
	EXPECT_TRUE( mem<int>::equal(p, int(0), N) );

	alloc.deallocate(p, N);

	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
}


#define CHECK_MEM_PENDING(k) ASSERT_EQ( global_memory_allocation_monitor.num_pending_sections(), k )

TEST( BasicMem, ConstBlocks )
{
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
	{

	using std::swap;

	const size_t N = 5;
	int src[N] = {4, 3, 1, 2, 5};

	CHECK_MEM_PENDING( 0 );

	cblk_t B0(cref_blk(src, N));
	EXPECT_TRUE( test_block(B0, false, N, src) );

	cblk_t B1(N, int(2));
	EXPECT_TRUE( test_block(B1, true, N) );
	EXPECT_TRUE( mem<int>::equal(B1.pbase(), int(2), N) );

	CHECK_MEM_PENDING( 1 );

	cblk_t B2(copy_blk(src, N));
	EXPECT_TRUE( test_block(B2, true, N) );
	EXPECT_TRUE( B2.pbase() != src );
	EXPECT_TRUE( mem<int>::equal(B2.pbase(), src, N) );

	CHECK_MEM_PENDING( 2 );

	cblk_t B0c(B0);
	EXPECT_TRUE( test_block(B0, false, N, src) );
	EXPECT_TRUE( test_block(B0c, false, N, src) );

	CHECK_MEM_PENDING( 2 );

	const int *p2 = B2.pbase();
	cblk_t B2c(B2);
	EXPECT_TRUE( test_block(B2, true, N, p2) );
	EXPECT_TRUE( test_block(B2c, true, N) );
	EXPECT_TRUE( B2c.pbase() != p2 );
	EXPECT_TRUE( mem<int>::equal(B2.pbase(), B2c.pbase(), N) );

	CHECK_MEM_PENDING( 3 );

	const int *p1 = B1.pbase();
	// const int *p2 = B2.pbase();

	swap(B1, B2);
	swap(p1, p2);
	EXPECT_TRUE( test_block(B1, true, N, p1) );
	EXPECT_TRUE( test_block(B2, true, N, p2) );

	CHECK_MEM_PENDING( 3 );

	swap(B0, B1);

	EXPECT_TRUE( test_block(B0, true, N, p1) );
	EXPECT_TRUE( test_block(B1, false, N, src) );

	CHECK_MEM_PENDING( 3 );

	swap(B0, B1);

	EXPECT_TRUE( test_block(B1, true, N, p1) );
	EXPECT_TRUE( test_block(B0, false, N, src) );

	CHECK_MEM_PENDING( 3 );
	}
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
}

TEST( BasicMem, Blocks )
{
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
	{

	using std::swap;

	const size_t N = 5;
	int src[N] = {4, 3, 1, 2, 5};

	CHECK_MEM_PENDING( 0 );

	blk_t B0( ref_blk(src, N));
	EXPECT_TRUE( test_block(B0, false, N, src) );

	blk_t B1(N, int(2));
	EXPECT_TRUE( test_block(B1, true, N) );
	EXPECT_TRUE( mem<int>::equal(B1.pbase(), int(2), N) );

	CHECK_MEM_PENDING( 1 );

	blk_t B2(copy_blk(src, N));
	EXPECT_TRUE( test_block(B2, true, N) );
	EXPECT_TRUE( B2.pbase() != src );
	EXPECT_TRUE( mem<int>::equal(B2.pbase(), src, N) );

	CHECK_MEM_PENDING( 2 );

	blk_t B0c(B0);
	EXPECT_TRUE( test_block(B0, false, N, src) );
	EXPECT_TRUE( test_block(B0c, false, N, src) );

	CHECK_MEM_PENDING( 2 );

	const int *p2 = B2.pbase();
	blk_t B2c(B2);
	EXPECT_TRUE( test_block(B2, true, N, p2) );
	EXPECT_TRUE( test_block(B2c, true, N) );
	EXPECT_TRUE( B2c.pbase() != p2 );
	EXPECT_TRUE( mem<int>::equal(B2.pbase(), B2c.pbase(), N) );

	CHECK_MEM_PENDING( 3 );

	const int *p1 = B1.pbase();
	// const int *p2 = B2.pbase();

	swap(B1, B2);
	swap(p1, p2);
	EXPECT_TRUE( test_block(B1, true, N, p1) );
	EXPECT_TRUE( test_block(B2, true, N, p2) );

	CHECK_MEM_PENDING( 3 );

	swap(B0, B1);

	EXPECT_TRUE( test_block(B0, true, N, p1) );
	EXPECT_TRUE( test_block(B1, false, N, src) );

	CHECK_MEM_PENDING( 3 );

	swap(B0, B1);

	EXPECT_TRUE( test_block(B1, true, N, p1) );
	EXPECT_TRUE( test_block(B0, false, N, src) );

	CHECK_MEM_PENDING( 3 );

	}
	ASSERT_FALSE( global_memory_allocation_monitor.has_pending() );
}


#undef CHECK_MEM_PENDING

