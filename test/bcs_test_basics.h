/**
 * @file bcs_test_basics.h
 *
 * Basic facilities for BCSLib testing
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_TEST_BASICS_H_
#define BCSLIB_TEST_BASICS_H_

#include <gtest/gtest.h>

#include <bcslib/core/basic_defs.h>
#include <cstdio>
#include <cmath>

namespace bcs { namespace test {

	template<typename TIter0, typename TIter>
	bool collection_equal(TIter0 begin, TIter0 end, TIter src, size_t n)
	{
		TIter0 it = begin;
		for (size_t i = 0; i < n; ++i)
		{
			if (it == end)
			{
				return false;
			}
			if (*it != *src)
			{
				return false;
			}

			++it;
			++src;
		}

		return it == end;
	}


	template<class ArrayClass1, class ArrayClass2>
	bool array_equal(const ArrayClass1& a, const ArrayClass2& b, index_t n)
	{
		for (index_t i = 0; i < n; ++i)
		{
			if (!(a[i] == b[i])) return false;
		}
		return true;
	}


	template<class ArrayClass1, typename T>
	bool array_equal_scalar(const ArrayClass1& a, const T& v, index_t n)
	{
		for (index_t i = 0; i < n; ++i)
		{
			if (!(a[i] == v)) return false;
		}
		return true;
	}

	template<class ArrayClass1, class ArrayClass2>
	bool array1d_equal(const ArrayClass1& a, const ArrayClass2& b)
	{
		if (a.nelems() != b.nelems()) return false;

		index_t n = a.nelems();
		for (index_t i = 0; i < n; ++i)
		{
			if (!(a(i) == b(i))) return false;
		}
		return true;
	}

	template<class ArrayClass1, class ArrayClass2>
	bool array2d_equal(const ArrayClass1& a, const ArrayClass2& b)
	{
		if (a.nrows() != b.nrows()) return false;
		if (a.ncolumns() != b.ncolumns()) return false;

		index_t m = a.nrows();
		index_t n = a.ncolumns();
		for (index_t i = 0; i < m; ++i)
		{
			for (index_t j = 0; j < n; ++j)
			{
				if (!(a(i, j) == b(i, j))) return false;
			}
		}
		return true;
	}


	template<class ArrayClass1, class ArrayClass2>
	bool array_approx(const ArrayClass1& a, const ArrayClass2& b, index_t n, double eps)
	{
		for (index_t i = 0; i < n; ++i)
		{
			double d = abs((double)(a[i] - b[i]));
			if (d > eps) return false;
		}
		return true;
	}


	template<class ArrayClass>
	void print_array1d(const ArrayClass& a, const char *fmt)
	{
		index_t n = a.nelems();
		for (index_t i = 0; i < n; ++i)
		{
			std::printf(fmt, a(i));
		}
		std::printf("\n");
	}


	template<class ArrayClass>
	void print_array2d(const ArrayClass& a, const char *fmt)
	{
		index_t m = a.nrows();
		index_t n = a.ncolumns();

		for (index_t i = 0; i < m; ++i)
		{
			for(index_t j = 0; j < n; ++j)
			{
				std::printf(fmt, a(i, j));
			}
			std::printf("\n");
		}
	}


} }



#endif 
