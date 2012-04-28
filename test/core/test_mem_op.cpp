/*
 * @file test_mem_op.cpp
 *
 * Unit testing for memory operations
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/core.h>

using namespace bcs;


static void test_elemwise1d(const index_t N)
{
	int *src = new int[N];
	for (index_t i = 0; i < N; ++i) src[i] = 2 * int(i) + 1;

	int *p = new int[N];
	for (index_t i = 0; i < N; ++i) p[i] = 0;

	int specval = 712;

	fill_elems(N, p, specval);
	ASSERT_TRUE( elems_equal(N, p, specval) );

	if (N >= 3)
	{
		p[2] = 0;
		ASSERT_FALSE( elems_equal(N, p, specval) );
	}

	copy_elems(N, src, p);
	ASSERT_TRUE( elems_equal(N, p, src) );

	zero_elems(N, p);
	ASSERT_TRUE( elems_equal(N, p, int(0)) );

	delete[] src;
	delete[] p;
}


TEST( MemOp, ElemWise1D_N01 ) { test_elemwise1d(1); }
TEST( MemOp, ElemWise1D_N02 ) { test_elemwise1d(2); }
TEST( MemOp, ElemWise1D_N03 ) { test_elemwise1d(3); }
TEST( MemOp, ElemWise1D_N04 ) { test_elemwise1d(4); }
TEST( MemOp, ElemWise1D_N05 ) { test_elemwise1d(5); }
TEST( MemOp, ElemWise1D_N06 ) { test_elemwise1d(6); }
TEST( MemOp, ElemWise1D_N07 ) { test_elemwise1d(7); }
TEST( MemOp, ElemWise1D_N08 ) { test_elemwise1d(8); }
TEST( MemOp, ElemWise1D_N12 ) { test_elemwise1d(12); }


TEST( MemOp, ElemWise2D )
{
	const index_t m0 = 3;
	const index_t n0 = 5;

	const index_t m = 3;
	const index_t n = 4;

	const index_t in_dim = 3;
	const index_t out_dim = 2;

	int src0[m0 * n0] = {1, 2, 3, 4, 5,   6, 7, 8, 9, 10,   11, 12, 13, 14, 15};
	const int *src = src0;

	int *p = new int[m * n];
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

	delete [] p;
}

template<int N>
static void test_static_elemwise1d()
{
	int src[N];
	for (int i = 0; i < N; ++i) src[i] = 2 * int(i) + 1;

	int p[N];
	for (int i = 0; i < N; ++i) p[i] = 0;

	int specval = 712;

	mem<int, N>::fill(p, specval);
	ASSERT_TRUE( elems_equal(N, p, specval) );

	if (N >= 3)
	{
		p[2] = 0;
		ASSERT_FALSE( elems_equal(N, p, specval) );
	}

	mem<int, N>::copy(src, p);
	ASSERT_TRUE( elems_equal(N, p, src) );

	mem<int, N>::zero(p);
	ASSERT_TRUE( elems_equal(N, p, int(0)) );
}


TEST( MemOp, StaticElemWise1D_N01 ) { test_static_elemwise1d<1>(); }
TEST( MemOp, StaticElemWise1D_N02 ) { test_static_elemwise1d<2>(); }
TEST( MemOp, StaticElemWise1D_N03 ) { test_static_elemwise1d<3>(); }
TEST( MemOp, StaticElemWise1D_N04 ) { test_static_elemwise1d<4>(); }
TEST( MemOp, StaticElemWise1D_N05 ) { test_static_elemwise1d<5>(); }
TEST( MemOp, StaticElemWise1D_N06 ) { test_static_elemwise1d<6>(); }
TEST( MemOp, StaticElemWise1D_N07 ) { test_static_elemwise1d<7>(); }
TEST( MemOp, StaticElemWise1D_N08 ) { test_static_elemwise1d<8>(); }
TEST( MemOp, StaticElemWise1D_N09 ) { test_static_elemwise1d<9>(); }
TEST( MemOp, StaticElemWise1D_N10 ) { test_static_elemwise1d<10>(); }
TEST( MemOp, StaticElemWise1D_N11 ) { test_static_elemwise1d<11>(); }
TEST( MemOp, StaticElemWise1D_N12 ) { test_static_elemwise1d<12>(); }
TEST( MemOp, StaticElemWise1D_N13 ) { test_static_elemwise1d<13>(); }
TEST( MemOp, StaticElemWise1D_N14 ) { test_static_elemwise1d<14>(); }
TEST( MemOp, StaticElemWise1D_N15 ) { test_static_elemwise1d<15>(); }
TEST( MemOp, StaticElemWise1D_N16 ) { test_static_elemwise1d<16>(); }




