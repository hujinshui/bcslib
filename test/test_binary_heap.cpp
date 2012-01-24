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

template class bcs::consecutive_binary_tree<int>;
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
int verify_heap_basics(binary_heap<T>& heap)
{
	typedef typename binary_heap<T>::tree_type tree_t;
	const tree_t& btree = heap.tree();
	const typename binary_heap<T>::node_map_type& node_map = heap.node_map();
	const std::vector<T>& elements = heap.elements();

	size_t ne = elements.size();
	size_t n = heap.size();

	if (!( n <= ne && n == btree.size() && ne == node_map.size() ))
	{
		std::printf("btree size = %d, node_map size = %d\n", (int)btree.size(), (int)node_map.size());
		return 1;
	}

	if (n > 0)
	{
		if (elements[heap.top_index()] != heap.top_value())
			return false;
	}

	for (size_t v = 1; v <= n; ++v)
	{
		typename tree_t::handle nd(v);
		size_t i = btree(v);
		if (node_map[i] != nd) return 2;
	}

	size_t c = 0;
	for (size_t i = 0; i < ne; ++i)
	{
		typename tree_t::handle v = node_map[i];
		if (v.non_nil())
		{
			if (!heap.in_heap(i)) return 3;
			if (btree(v) != i) return 4;
			++ c;
		}
		else
		{
			if (heap.in_heap(i)) return 5;
		}
	}
	if (c != n) return 6;

	return 0;
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
	H0.make_heap();

	ASSERT_EQ( 0, H0.size() );
	ASSERT_EQ( 0, verify_heap_basics(H0) );
	ASSERT_TRUE( verify_binary_min_heap(H0) );


	// single element heap

	vec_t V1;
	V1.push_back(101);
	heap_t H1(V1);
	H1.make_heap();

	ASSERT_EQ( 1, H1.size() );
	ASSERT_EQ( 0, verify_heap_basics(H1) );
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
		H5.make_heap();

		ASSERT_EQ( 5, H5.size() );
		ASSERT_EQ( 0, verify_heap_basics(H5) );
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
		H6.make_heap();

		ASSERT_EQ( 6, H6.size());
		ASSERT_EQ( 0, verify_heap_basics(H6) );
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
		H7.make_heap();

		ASSERT_EQ( 7, H7.size());
		ASSERT_EQ( 0, verify_heap_basics(H7) );
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
		H9.make_heap();

		ASSERT_EQ( 9, H9.size());
		ASSERT_EQ( 0, verify_heap_basics(H9) );
		ASSERT_TRUE( verify_binary_min_heap(H9) );
	}


	// nine-element partial heap

	for (int i = 0; i < nr; ++i)
	{
		vec_t V9;
		for (int j = 0; j < 9; ++j)
		{
			V9.push_back(std::rand() % 100);
		}

		heap_t H9(V9);
		ASSERT_EQ( 0, H9.size());
		ASSERT_EQ( 0, verify_heap_basics(H9) );
		ASSERT_TRUE( verify_binary_min_heap(H9) );

		size_t inds[5] = {1, 3, 4, 6, 7};
		H9.make_heap(inds, inds+5);

		ASSERT_FALSE( H9.in_heap(0) );
		ASSERT_TRUE ( H9.in_heap(1) );
		ASSERT_FALSE( H9.in_heap(2) );
		ASSERT_TRUE ( H9.in_heap(3) );
		ASSERT_TRUE ( H9.in_heap(4) );
		ASSERT_FALSE( H9.in_heap(5) );
		ASSERT_TRUE ( H9.in_heap(6) );
		ASSERT_TRUE ( H9.in_heap(7) );
		ASSERT_FALSE( H9.in_heap(8) );

		ASSERT_EQ( 5, H9.size() );
		ASSERT_EQ( 0, verify_heap_basics(H9) );
		ASSERT_TRUE( verify_binary_min_heap(H9) );
	}

}


TEST( BinaryHeap, InsertAndDelete )
{
	size_t nr = 100;
	size_t N = 50;

	for (size_t i = 0; i < nr; ++i)
	{
		// prepare elements

		vec_t V;
		for (size_t j = 0; j < N; ++j)
		{
			V.push_back(std::rand() % 100);
		}

		// gradual enroll

		heap_t H(V);
		for (size_t j = 0; j < N; ++j)
		{
			H.insert(j);

			ASSERT_EQ( j+1, H.size());
			ASSERT_EQ( 0, verify_heap_basics(H) );
			ASSERT_TRUE( verify_binary_min_heap(H) );
		}

		// gradual pop

		int prev_v = 0;
		for (size_t j = 0; j < N; ++j)
		{
			if (j > 0)
			{
				ASSERT_TRUE( H.top_value() >= prev_v );
			}
			prev_v = H.top_value();
			ASSERT_EQ( prev_v, V[H.top_index()] );

			H.delete_top();
			ASSERT_EQ( N-(j+1), H.size() );
			ASSERT_EQ( 0, verify_heap_basics(H) );
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
		H.make_heap();

		ASSERT_EQ( N, H.size() );
		ASSERT_EQ( 0, verify_heap_basics(H) );
		ASSERT_TRUE( verify_binary_min_heap(H) );

		for (size_t j = 0; j < nc; ++j)
		{
			size_t idx = (size_t)std::rand() % N;
			int v = std::rand() % 100;

			update_element(V, H, idx, v);
			ASSERT_EQ( v, V[idx] );
			ASSERT_EQ( v, H.get_by_node(H.node(idx)) );

			ASSERT_EQ( N, H.size() );
			ASSERT_EQ( 0, verify_heap_basics(H) );
			ASSERT_TRUE( verify_binary_min_heap(H) );
		}
	}
}






