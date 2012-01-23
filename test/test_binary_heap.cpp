/**
 * @file test_binary_heap.cpp
 *
 *  Unit Testing of Binary Heap
 *
 * @author Dahua Lin
 */


#include "bcs_test_basics.h"
#include <bcslib/data_structs/binary_heap.h>
#include <cstdlib>

using namespace bcs;
using namespace bcs::test;

// Explicit instantiation for syntax checking

template class bcs::binary_heap<int>;

typedef std::vector<int> vec_t;
typedef binary_heap<int> heap_t;

// Tests

void print_heap(const heap_t& H, bool print_vec = false)
{
	size_t n = H.size();

	if (print_vec)
	{
		const std::vector<int>& elements = H.elements();
		for (size_t i = 0; i < elements.size(); ++i)
		{
			std::printf("%d ", elements[i]);
		}
		std::printf("==> ");
	}

	for (size_t v = 1; v <= n; ++v)
	{
		std::printf("%d ", H.get_by_node(v));
	}

	std::printf("\n");
}



template<typename T>
bool verify_heap_basics(binary_heap<T>& heap)
{
	const std::vector<size_t>& btree = heap.tree_array();
	const std::vector<size_t>& node_map = heap.node_map();
	const std::vector<T>& elements = heap.elements();

	size_t ne = elements.size();
	size_t n = heap.size();

	if (!( n <= ne && n == btree.size() && ne == node_map.size() ))
	{
		return false;
	}

	for (size_t v = 1; v <= n; ++v)
	{
		size_t i = btree[v-1];
		if (node_map[i] != v) return false;
	}

	for (size_t i = 0; i < ne; ++i)
	{
		size_t v = node_map[i];
		if (v > 0)
		{
			if (!heap.is_in_heap(i)) return false;
			if (btree[v-1] != i) return false;
		}
		else
		{
			if (heap.is_in_heap(i)) return false;
		}
	}

	return true;
}



template<typename T>
bool verify_binary_min_heap(binary_heap<T>& heap)
{
	size_t n = heap.size();

	if (n > 0)
	{
		for (size_t v = 1; v <= n; ++v)
		{
			size_t lc = 2 * v;
			size_t rc = 2 * v + 1;

			T cval = heap.get_by_node(v);

			if (v > 1)
			{
				if (cval < heap.get_by_node(v >> 1)) return false;
			}

			if (lc <= n)
			{
				if (cval > heap.get_by_node(lc)) return false;
			}

			if (rc <= n)
			{
				if (cval > heap.get_by_node(rc)) return false;
			}
		}
	}

	return true;
}



TEST( BinaryHeap, MakeHeap )
{
	// empty heap

	vec_t V0;
	heap_t H0(V0);

	ASSERT_EQ( H0.size(), 0 );
	ASSERT_TRUE( verify_heap_basics(H0) );
	ASSERT_TRUE( verify_binary_min_heap(H0) );


	// single element heap

	vec_t V1;
	V1.push_back(101);
	heap_t H1(V1);

	ASSERT_EQ( H1.size(), 1 );
	ASSERT_TRUE( verify_heap_basics(H1) );
	ASSERT_TRUE( verify_binary_min_heap(H1) );

	// five-element heap

	int nr = 100;
	for (int i = 0; i < nr; ++i)
	{
		vec_t V5;
		for (int j = 0; j < 5; ++j)
		{
			V5.push_back(std::rand() % 100);
		}

		heap_t H5(V5);

		ASSERT_EQ(H5.size(), 5);
		ASSERT_TRUE( verify_heap_basics(H5) );
		ASSERT_TRUE( verify_binary_min_heap(H5) );

	}

	// six-element heap

	for (int i = 0; i < nr; ++i)
	{
		vec_t V6;
		for (int j = 0; j < 6; ++j)
		{
			V6.push_back(std::rand() % 100);
		}

		heap_t H6(V6);

		ASSERT_EQ(H6.size(), 6);
		ASSERT_TRUE( verify_heap_basics(H6) );
		ASSERT_TRUE( verify_binary_min_heap(H6) );
	}

	// seven-element heap

	for (int i = 0; i < nr; ++i)
	{
		vec_t V7;
		for (int j = 0; j < 7; ++j)
		{
			V7.push_back(std::rand() % 100);
		}

		heap_t H7(V7);

		ASSERT_EQ(H7.size(), 7);
		ASSERT_TRUE( verify_heap_basics(H7) );
		ASSERT_TRUE( verify_binary_min_heap(H7) );

	}


	// nine-element heap

	for (int i = 0; i < nr; ++i)
	{
		vec_t V9;
		for (int j = 0; j < 9; ++j)
		{
			V9.push_back(std::rand() % 100);
		}

		heap_t H9(V9);

		ASSERT_EQ(H9.size(), 9);
		ASSERT_TRUE( verify_heap_basics(H9) );
		ASSERT_TRUE( verify_binary_min_heap(H9) );
	}

}



TEST( BinaryHeap, PushAndPop )
{
	size_t nr = 100;
	size_t N = 50;

	for (size_t i = 0; i < nr; ++i)
	{
		vec_t V;
		heap_t H(V);

		for (size_t j = 0; j < N; ++j)
		{
			H.push(std::rand() % 100);

			ASSERT_EQ(H.size(), j+1);
			ASSERT_TRUE( verify_heap_basics(H) );
			ASSERT_TRUE( verify_binary_min_heap(H) );
		}

		int prev_v = 0;
		for (size_t j = 0; j < N; ++j)
		{
			if (j > 0)
			{
				ASSERT_TRUE( H.top() >= prev_v );
			}
			prev_v = H.top();

			H.pop();
			ASSERT_EQ( H.size(), N-(j+1) );
			ASSERT_TRUE( verify_heap_basics(H) );
			ASSERT_TRUE( verify_binary_min_heap(H) );
		}
	}
}


TEST( BinaryHeap, Update )
{
	size_t nr = 100;
	size_t N = 25;
	size_t nc = 200;

	for (size_t i = 0; i < nr; ++i)
	{
		vec_t V;

		for (size_t j = 0; j < N; ++j)
		{
			V.push_back(std::rand() % 100);
		}
		heap_t H(V);

		ASSERT_EQ(H.size(), N);
		ASSERT_TRUE( verify_heap_basics(H) );
		ASSERT_TRUE( verify_binary_min_heap(H) );

		for (size_t j = 0; j < nc; ++j)
		{
			size_t idx = (size_t)std::rand() % N;
			int v = std::rand() % 100;

			H.set(idx, v);
			ASSERT_EQ(H.get(idx), v);

			ASSERT_EQ(H.size(), N);
			ASSERT_TRUE( verify_heap_basics(H) );
			ASSERT_TRUE( verify_binary_min_heap(H) );
		}
	}
}






