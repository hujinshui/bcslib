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

	// sum

	template<typename T, class TIndexer>
	T sum(const aview1d<T, TIndexer>& a)
	{
		return sum_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T sum(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return sum_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> sum(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return sum_arr(a, Slicing());
	}


	// vdot

	template<typename T, class LIndexer, class RIndexer>
	T vdot(const aview1d<T, LIndexer>& a, const aview1d<T, RIndexer>& b)
	{
		return vdot_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T vdot(const aview2d<T, TOrd, LIndexer0, LIndexer1>& a, const aview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return vdot_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> vdot(const aview2d<T, TOrd, LIndexer0, LIndexer1>& a, const aview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return vdot_arr(a, b, Slicing());
	}


	// sum_log

	template<typename T, class TIndexer>
	T sum_log(const aview1d<T, TIndexer>& a)
	{
		return sum_log_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T sum_log(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return sum_log_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> sum_log(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return sum_log_arr(a, Slicing());
	}


	// sum_xlogy

	template<typename T, class LIndexer, class RIndexer>
	T sum_xlogy(const aview1d<T, LIndexer>& a, const aview1d<T, RIndexer>& b)
	{
		return sum_xlogy_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T sum_xlogy(const aview2d<T, TOrd, LIndexer0, LIndexer1>& a, const aview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return sum_xlogy_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> sum_xlogy(const aview2d<T, TOrd, LIndexer0, LIndexer1>& a, const aview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return sum_xlogy_arr(a, b, Slicing());
	}


	// mean

	template<typename T, class TIndexer>
	T mean(const aview1d<T, TIndexer>& a)
	{
		return mean_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T mean(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return mean_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> mean(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return mean_arr(a, Slicing());
	}


	// min

	template<typename T, class TIndexer>
	T min(const aview1d<T, TIndexer>& a)
	{
		return min_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T min(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return min_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> min(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return min_arr(a, Slicing());
	}


	// max

	template<typename T, class TIndexer>
	T max(const aview1d<T, TIndexer>& a)
	{
		return max_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T max(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return max_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> max(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return max_arr(a, Slicing());
	}


	// minmax

	template<typename T, class TIndexer>
	std::pair<T, T> minmax(const aview1d<T, TIndexer>& a)
	{
		return minmax_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::pair<T, T> minmax(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return minmax_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<std::pair<T, T> > minmax(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return minmax_arr(a, Slicing());
	}


	// min_index

	template<typename T, class TIndexer>
	index_t min_index(const aview1d<T, TIndexer>& a)
	{
		return min_index_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::array<index_t, 2> min_index2d(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return min_index_arr2d(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<index_t> min_index(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return min_index_arr(a, Slicing());
	}


	// max_index

	template<typename T, class TIndexer>
	index_t max_index(const aview1d<T, TIndexer>& a)
	{
		return max_index_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::array<index_t, 2> max_index2d(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return max_index_arr2d(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<index_t> max_index(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return max_index_arr(a, Slicing());
	}

}

#endif 
