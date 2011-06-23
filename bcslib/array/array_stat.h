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
	T sum(const caview1d<T, TIndexer>& a)
	{
		return sum_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T sum(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return sum_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> sum(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return sum_arr(a, Slicing());
	}


	// vdot

	template<typename T, class LIndexer, class RIndexer>
	T vdot(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return vdot_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T vdot(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return vdot_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> vdot(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return vdot_arr(a, b, Slicing());
	}


	// sum_log

	template<typename T, class TIndexer>
	T sum_log(const caview1d<T, TIndexer>& a)
	{
		return sum_log_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T sum_log(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return sum_log_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> sum_log(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return sum_log_arr(a, Slicing());
	}


	// sum_xlogy

	template<typename T, class LIndexer, class RIndexer>
	T sum_xlogy(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return sum_xlogy_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T sum_xlogy(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return sum_xlogy_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> sum_xlogy(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return sum_xlogy_arr(a, b, Slicing());
	}


	// mean

	template<typename T, class TIndexer>
	T mean(const caview1d<T, TIndexer>& a)
	{
		return mean_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T mean(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return mean_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> mean(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return mean_arr(a, Slicing());
	}


	// min

	template<typename T, class TIndexer>
	T min(const caview1d<T, TIndexer>& a)
	{
		return min_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T min(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return min_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> min(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return min_arr(a, Slicing());
	}


	// max

	template<typename T, class TIndexer>
	T max(const caview1d<T, TIndexer>& a)
	{
		return max_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T max(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return max_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> max(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return max_arr(a, Slicing());
	}


	// minmax

	template<typename T, class TIndexer>
	std::pair<T, T> minmax(const caview1d<T, TIndexer>& a)
	{
		return minmax_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::pair<T, T> minmax(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return minmax_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<std::pair<T, T> > minmax(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return minmax_arr(a, Slicing());
	}


	// min_index

	template<typename T, class TIndexer>
	index_t min_index(const caview1d<T, TIndexer>& a)
	{
		return min_index_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::array<index_t, 2> min_index2d(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return min_index_arr2d(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<index_t> min_index(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return min_index_arr(a, Slicing());
	}


	// max_index

	template<typename T, class TIndexer>
	index_t max_index(const caview1d<T, TIndexer>& a)
	{
		return max_index_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::array<index_t, 2> max_index2d(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return max_index_arr2d(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<index_t> max_index(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return max_index_arr(a, Slicing());
	}


	// norm_L0

	template<typename T, class TIndexer>
	T norm_L0(const caview1d<T, TIndexer>& a)
	{
		return norm_L0_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T norm_L0(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return norm_L0_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> norm_L0(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return norm_L0_arr(a, Slicing());
	}

	// diff_norm_L0

	template<typename T, class LIndexer, class RIndexer>
	T diff_norm_L0(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return diff_norm_L0_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_norm_L0(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return diff_norm_L0_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_norm_L0(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return diff_norm_L0_arr(a, b, Slicing());
	}


	// norm_L1

	template<typename T, class TIndexer>
	T norm_L1(const caview1d<T, TIndexer>& a)
	{
		return norm_L1_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T norm_L1(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return norm_L1_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> norm_L1(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return norm_L1_arr(a, Slicing());
	}

	// diff_norm_L1

	template<typename T, class LIndexer, class RIndexer>
	T diff_norm_L1(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return diff_norm_L1_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_norm_L1(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return diff_norm_L1_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_norm_L1(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return diff_norm_L1_arr(a, b, Slicing());
	}


	// sqrsum

	template<typename T, class TIndexer>
	T sqrsum(const caview1d<T, TIndexer>& a)
	{
		return sqrsum_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T sqrsum(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return sqrsum_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> sqrsum(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return sqrsum_arr(a, Slicing());
	}

	// diff_sqrsum

	template<typename T, class LIndexer, class RIndexer>
	T diff_sqrsum(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return diff_sqrsum_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_sqrsum(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return diff_sqrsum_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_sqrsum(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return diff_sqrsum_arr(a, b, Slicing());
	}


	// norm_L2

	template<typename T, class TIndexer>
	T norm_L2(const caview1d<T, TIndexer>& a)
	{
		return norm_L2_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T norm_L2(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return norm_L2_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> norm_L2(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return norm_L2_arr(a, Slicing());
	}

	// diff_norm_L2

	template<typename T, class LIndexer, class RIndexer>
	T diff_norm_L2(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return diff_norm_L2_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_norm_L2(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return diff_norm_L2_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_norm_L2(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return diff_norm_L2_arr(a, b, Slicing());
	}


	// norm_Linf

	template<typename T, class TIndexer>
	T norm_Linf(const caview1d<T, TIndexer>& a)
	{
		return norm_Linf_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T norm_Linf(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return norm_Linf_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> norm_Linf(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return norm_Linf_arr(a, Slicing());
	}

	// diff_norm_Linf

	template<typename T, class LIndexer, class RIndexer>
	T diff_norm_Linf(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return diff_norm_Linf_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_norm_Linf(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return diff_norm_Linf_arr(a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_norm_Linf(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return diff_norm_Linf_arr(a, b, Slicing());
	}

}

#endif 
