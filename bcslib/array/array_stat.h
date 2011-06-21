/**
 * @file array_stat.h
 *
 * Evaluation of array stats for array classes
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_STAT_H_
#define BCSLIB_ARRAY_STAT_H_

#include <bcslib/array/generic_array_stat.h>

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/base/arg_check.h>
#include <type_traits>

namespace bcs
{

	template<typename T, class TIndexer>
	T sum(const aview1d<T, TIndexer>& a)
	{
		return sum_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer1, class TIndexer2>
	T sum(const aview2d<T, TOrd, TIndexer1, TIndexer2>& a)
	{
		return sum_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer1, class TIndexer2, typename Slicing>
	array1d<T> sum(const aview2d<T, TOrd, TIndexer1, TIndexer2>& a, Slicing)
	{
		return sum_arr(a, Slicing());
	}


}

#endif 
