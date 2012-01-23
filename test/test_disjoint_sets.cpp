/**
 * @file test_disjoint_sets.cpp
 *
 * Unit testing of disjoint sets
 *
 * @author Dahua Lin
 */


#include "bcs_test_basics.h"
#include <bcslib/data_structs/disjoint_sets.h>
#include <cstdlib>

using namespace bcs;
using namespace bcs::test;



TEST( DisjointSetForest, Init )
{
	disjoint_set_forest s0(0);

	ASSERT_EQ(s0.size(), 0);
	ASSERT_EQ(s0.nclasses(), 0);

	disjoint_set_forest s1(1);

	ASSERT_EQ(s1.size(), 1);
	ASSERT_EQ(s1.parent(0), 0);
	ASSERT_EQ(s1.rank(0), 0);
	ASSERT_EQ(s1.nclasses(), 1);

	size_t n = 10;
	disjoint_set_forest s2(n);

	ASSERT_EQ(s2.size(), n);
	ASSERT_EQ(s2.nclasses(), n);

	for (size_t x = 0; x < n; ++x)
	{
		ASSERT_EQ(s2.parent(x), x);
		ASSERT_EQ(s2.rank(x), 0);
	}
}


TEST( DisjointSetForest, JoinAndCompress )
{
	disjoint_set_forest s(16);
	ASSERT_EQ(s.size(), 16);
	ASSERT_EQ(s.nclasses(), 16);

	for (size_t x = 0; x < 16; ++x)
	{
		ASSERT_EQ( s.trace_root(x), x );
	}

	// join [0, (1)], [2, (3)], ..., [14, (15)]

	for (size_t x = 0; x < 8; ++x)
	{
		s.join(2 * x, 2 * x + 1);
	}

	ASSERT_EQ(s.size(), 16);
	ASSERT_EQ(s.nclasses(), 8);

	for (size_t x = 0; x < 16; ++x)
	{
		if (x % 2 == 0)
		{
			ASSERT_EQ( s.parent(x), x+1 );
			ASSERT_EQ( s.rank(x), 0 );
			ASSERT_EQ( s.trace_root(x), x + 1);
		}
		else
		{
			ASSERT_EQ( s.parent(x), x );
			ASSERT_EQ( s.rank(x), 1 );
			ASSERT_EQ( s.trace_root(x), x);
		}
	}

	// join [0, (1)] - [2, (3)] => [0, 1, 2, (3)]

	s.join(0, 2);

	ASSERT_EQ(s.size(), 16);
	ASSERT_EQ(s.nclasses(), 7);
	ASSERT_EQ(s.rank(1), 1);
	ASSERT_EQ(s.rank(3), 2 );

	ASSERT_EQ( s.parent(0), 3 );
	ASSERT_EQ( s.parent(1), 3 );
	ASSERT_EQ( s.parent(2), 3 );
	ASSERT_EQ( s.parent(3), 3 );

	ASSERT_EQ( s.trace_root(0), 3 );
	ASSERT_EQ( s.trace_root(1), 3 );
	ASSERT_EQ( s.trace_root(2), 3 );
	ASSERT_EQ( s.trace_root(3), 3 );

	// join [0, 1, 2, (3)] - [4, (5)] => [0, 1, 2, (3), 4, 5]

	s.join(1, 5);

	ASSERT_EQ(s.size(), 16);
	ASSERT_EQ(s.nclasses(), 6);
	ASSERT_EQ(s.rank(3), 2 );
	ASSERT_EQ(s.rank(5), 1);

	ASSERT_EQ( s.parent(0), 3 );
	ASSERT_EQ( s.parent(1), 3 );
	ASSERT_EQ( s.parent(2), 3 );
	ASSERT_EQ( s.parent(3), 3 );
	ASSERT_EQ( s.parent(4), 5 );
	ASSERT_EQ( s.parent(5), 3 );

	ASSERT_EQ( s.trace_root(0), 3 );
	ASSERT_EQ( s.trace_root(1), 3 );
	ASSERT_EQ( s.trace_root(2), 3 );
	ASSERT_EQ( s.trace_root(3), 3 );
	ASSERT_EQ( s.trace_root(4), 3 );
	ASSERT_EQ( s.trace_root(5), 3 );

	// join [6, (7)], [0, 1, 2, (3), 4, 5] => [0, 1, 2, (3), 4, 5, 6, 7]

	s.join(7, 2);

	ASSERT_EQ(s.size(), 16);
	ASSERT_EQ(s.nclasses(), 5);
	ASSERT_EQ(s.rank(7), 1);
	ASSERT_EQ(s.rank(3), 2 );

	ASSERT_EQ( s.parent(0), 3 );
	ASSERT_EQ( s.parent(1), 3 );
	ASSERT_EQ( s.parent(2), 3 );
	ASSERT_EQ( s.parent(3), 3 );
	ASSERT_EQ( s.parent(4), 5 );
	ASSERT_EQ( s.parent(5), 3 );
	ASSERT_EQ( s.parent(6), 7 );
	ASSERT_EQ( s.parent(7), 3 );

	ASSERT_EQ( s.trace_root(0), 3 );
	ASSERT_EQ( s.trace_root(1), 3 );
	ASSERT_EQ( s.trace_root(2), 3 );
	ASSERT_EQ( s.trace_root(3), 3 );
	ASSERT_EQ( s.trace_root(4), 3 );
	ASSERT_EQ( s.trace_root(5), 3 );
	ASSERT_EQ( s.trace_root(6), 3 );
	ASSERT_EQ( s.trace_root(7), 3 );

	// join [8, (9)], [10, (11)] => [8, 9, 10, (11)]

	s.join(9, 11);

	ASSERT_EQ(s.size(), 16);
	ASSERT_EQ(s.nclasses(), 4);
	ASSERT_EQ(s.rank(9), 1);
	ASSERT_EQ(s.rank(11), 2 );

	ASSERT_EQ( s.parent(8), 9 );
	ASSERT_EQ( s.parent(9), 11 );
	ASSERT_EQ( s.parent(10), 11 );
	ASSERT_EQ( s.parent(11), 11 );

	ASSERT_EQ( s.trace_root(8), 11 );
	ASSERT_EQ( s.trace_root(9), 11 );
	ASSERT_EQ( s.trace_root(10), 11 );
	ASSERT_EQ( s.trace_root(11), 11 );

	// join [0, 1, 2, (3), 4, 5, 6, 7] - [8, 9, 10, (11)] => [0, ..., (11)]

	s.join(3, 8);

	ASSERT_EQ(s.size(), 16);
	ASSERT_EQ(s.nclasses(), 3);
	ASSERT_EQ(s.rank(3), 2);
	ASSERT_EQ(s.rank(11), 3);

	ASSERT_EQ( s.parent(0), 3 );
	ASSERT_EQ( s.parent(1), 3 );
	ASSERT_EQ( s.parent(2), 3 );
	ASSERT_EQ( s.parent(3), 11 );
	ASSERT_EQ( s.parent(4), 5 );
	ASSERT_EQ( s.parent(5), 3 );
	ASSERT_EQ( s.parent(6), 7 );
	ASSERT_EQ( s.parent(7), 3 );
	ASSERT_EQ( s.parent(8), 11 );
	ASSERT_EQ( s.parent(9), 11 );
	ASSERT_EQ( s.parent(10), 11 );
	ASSERT_EQ( s.parent(11), 11 );

	ASSERT_EQ( s.trace_root(0), 11 );
	ASSERT_EQ( s.trace_root(1), 11 );
	ASSERT_EQ( s.trace_root(2), 11 );
	ASSERT_EQ( s.trace_root(3), 11 );
	ASSERT_EQ( s.trace_root(4), 11 );
	ASSERT_EQ( s.trace_root(5), 11 );
	ASSERT_EQ( s.trace_root(6), 11 );
	ASSERT_EQ( s.trace_root(7), 11 );
	ASSERT_EQ( s.trace_root(8), 11 );
	ASSERT_EQ( s.trace_root(9), 11 );
	ASSERT_EQ( s.trace_root(10), 11 );
	ASSERT_EQ( s.trace_root(11), 11 );

	// compress-all

	s.compress_all();

	ASSERT_EQ(s.size(), 16);
	ASSERT_EQ(s.nclasses(), 3);

	ASSERT_TRUE( s.rank(0) == 0 && s.parent(0) == 11 && s.trace_root(0) == 11 );
	ASSERT_TRUE( s.rank(1) == 0 && s.parent(1) == 11 && s.trace_root(1) == 11 );
	ASSERT_TRUE( s.rank(2) == 0 && s.parent(2) == 11 && s.trace_root(2) == 11 );
	ASSERT_TRUE( s.rank(3) == 0 && s.parent(3) == 11 && s.trace_root(3) == 11 );
	ASSERT_TRUE( s.rank(4) == 0 && s.parent(4) == 11 && s.trace_root(4) == 11 );
	ASSERT_TRUE( s.rank(5) == 0 && s.parent(5) == 11 && s.trace_root(5) == 11 );
	ASSERT_TRUE( s.rank(6) == 0 && s.parent(6) == 11 && s.trace_root(6) == 11 );
	ASSERT_TRUE( s.rank(7) == 0 && s.parent(7) == 11 && s.trace_root(7) == 11 );
	ASSERT_TRUE( s.rank(8) == 0 && s.parent(8) == 11 && s.trace_root(8) == 11 );
	ASSERT_TRUE( s.rank(9) == 0 && s.parent(9) == 11 && s.trace_root(9) == 11 );
	ASSERT_TRUE( s.rank(10) == 0 && s.parent(10) == 11 && s.trace_root(10) == 11 );
	ASSERT_TRUE( s.rank(11) == 1 && s.parent(11) == 11 && s.trace_root(11) == 11 );

	ASSERT_TRUE( s.rank(12) == 0 && s.parent(12) == 13 && s.trace_root(12) == 13 );
	ASSERT_TRUE( s.rank(13) == 1 && s.parent(13) == 13 && s.trace_root(13) == 13 );

	ASSERT_TRUE( s.rank(14) == 0 && s.parent(14) == 15 && s.trace_root(14) == 15 );
	ASSERT_TRUE( s.rank(15) == 1 && s.parent(15) == 15 && s.trace_root(15) == 15 );
}


