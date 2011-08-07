/**
 * @file bcs_test_basics.h
 *
 * Basic facilities for BCSLib testing
 * 
 * @author Dahua Lin
 */

#ifndef BCS_TEST_BASICS_H_
#define BCS_TEST_BASICS_H_

#include <gtest/gtest.h>
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
	bool array_equal(const ArrayClass1& a, const ArrayClass2& b, size_t n)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (!(a[i] == b[i])) return false;
		}
		return true;
	}


	template<class ArrayClass1, class ArrayClass2>
	bool array_approx(const ArrayClass1& a, const ArrayClass2& b, size_t n, double eps)
	{
		for (size_t i = 0; i < n; ++i)
		{
			double d = std::abs((double)(a[i] - b[i]));
			if (d > eps) return false;
		}
		return true;
	}


} }



#endif 
