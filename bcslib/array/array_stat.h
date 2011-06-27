/**
 * @file array_stat.h
 *
 * Evaluation of array stats for array classes
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARRAY_STAT_H_
#define BCSLIB_ARRAY_STAT_H_

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/base/arg_check.h>

#include <bcslib/array/generic_array_eval.h>
#include <bcslib/veccomp/vecstat.h>


namespace bcs
{

	// sum

	template<typename T, class TIndexer>
	T sum(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_sum_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T sum(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_sum_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> sum(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_sum_ftor<T>(), a, Slicing());
	}


	// vdot

	template<typename T, class LIndexer, class RIndexer>
	T vdot(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return evaluate_arr_stats(vec_dot_prod_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T vdot(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return evaluate_arr_stats(vec_dot_prod_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> vdot(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return evaluate_arr_slice_stats(vec_dot_prod_ftor<T>(), a, b, Slicing());
	}


	// sum_log

	template<typename T, class TIndexer>
	T sum_log(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_sum_log_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T sum_log(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_sum_log_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> sum_log(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_sum_log_ftor<T>(), a, Slicing());
	}


	// sum_xlogy

	template<typename T, class LIndexer, class RIndexer>
	T sum_xlogy(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return evaluate_arr_stats(vec_sum_xlogy_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T sum_xlogy(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return evaluate_arr_stats(vec_sum_xlogy_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> sum_xlogy(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return evaluate_arr_slice_stats(vec_sum_xlogy_ftor<T>(), a, b, Slicing());
	}


	// mean

	template<typename T, class TIndexer>
	T mean(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_mean_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T mean(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_mean_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> mean(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_mean_ftor<T>(), a, Slicing());
	}

	// min

	template<typename T, class TIndexer>
	T min(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_min_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T min(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_min_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> min(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_min_ftor<T>(), a, Slicing());
	}


	// max

	template<typename T, class TIndexer>
	T max(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_max_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T max(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_max_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> max(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_max_ftor<T>(), a, Slicing());
	}


	// minmax

	template<typename T, class TIndexer>
	std::pair<T, T> minmax(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_minmax_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::pair<T, T> minmax(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_minmax_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<std::pair<T, T> > minmax(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_minmax_ftor<T>(), a, Slicing());
	}


	// min_index

	template<typename T, class TIndexer>
	index_t min_index(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_min_index_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::array<index_t, 2> min_index2d(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		index_t idx = evaluate_arr_stats(vec_min_index_ftor<T>(), a);
		return layout_aux2d<TOrd>::ind2sub(a.dim0(), a.dim1(), idx);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<index_t> min_index(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_min_index_ftor<T>(), a, Slicing());
	}


	// max_index

	template<typename T, class TIndexer>
	index_t max_index(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_max_index_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	std::array<index_t, 2> max_index2d(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		index_t idx = evaluate_arr_stats(vec_max_index_ftor<T>(), a);
		return layout_aux2d<TOrd>::ind2sub(a.dim0(), a.dim1(), idx);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<index_t> max_index(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_max_index_ftor<T>(), a, Slicing());
	}


	// norm_L0

	template<typename T, class TIndexer>
	T norm_L0(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_norm_L0_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T norm_L0(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_norm_L0_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> norm_L0(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_norm_L0_ftor<T>(), a, Slicing());
	}

	// diff_norm_L0

	template<typename T, class LIndexer, class RIndexer>
	T diff_norm_L0(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return evaluate_arr_stats(vec_diff_norm_L0_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_norm_L0(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return evaluate_arr_stats(vec_diff_norm_L0_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_norm_L0(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return evaluate_arr_slice_stats(vec_diff_norm_L0_ftor<T>(), a, b, Slicing());
	}


	// norm_L1

	template<typename T, class TIndexer>
	T norm_L1(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_norm_L1_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T norm_L1(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_norm_L1_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> norm_L1(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_norm_L1_ftor<T>(), a, Slicing());
	}

	// diff_norm_L1

	template<typename T, class LIndexer, class RIndexer>
	T diff_norm_L1(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return evaluate_arr_stats(vec_diff_norm_L1_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_norm_L1(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return evaluate_arr_stats(vec_diff_norm_L1_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_norm_L1(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return evaluate_arr_slice_stats(vec_diff_norm_L1_ftor<T>(), a, b, Slicing());
	}


	// sqrsum

	template<typename T, class TIndexer>
	T sqrsum(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_sqrsum_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T sqrsum(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_sqrsum_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> sqrsum(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_sqrsum_ftor<T>(), a, Slicing());
	}

	// diff_sqrsum

	template<typename T, class LIndexer, class RIndexer>
	T diff_sqrsum(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return evaluate_arr_stats(vec_diff_sqrsum_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_sqrsum(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return evaluate_arr_stats(vec_diff_sqrsum_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_sqrsum(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return evaluate_arr_slice_stats(vec_diff_sqrsum_ftor<T>(), a, b, Slicing());
	}


	// norm_L2

	template<typename T, class TIndexer>
	T norm_L2(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_norm_L2_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T norm_L2(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_norm_L2_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> norm_L2(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_norm_L2_ftor<T>(), a, Slicing());
	}

	// diff_norm_L2

	template<typename T, class LIndexer, class RIndexer>
	T diff_norm_L2(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return evaluate_arr_stats(vec_diff_norm_L2_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_norm_L2(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return evaluate_arr_stats(vec_diff_norm_L2_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_norm_L2(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return evaluate_arr_slice_stats(vec_diff_norm_L2_ftor<T>(), a, b, Slicing());
	}


	// norm_Linf

	template<typename T, class TIndexer>
	T norm_Linf(const caview1d<T, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_norm_Linf_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	T norm_Linf(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_norm_Linf_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<T> norm_Linf(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_norm_Linf_ftor<T>(), a, Slicing());
	}

	// diff_norm_Linf

	template<typename T, class LIndexer, class RIndexer>
	T diff_norm_Linf(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& b)
	{
		return evaluate_arr_stats(vec_diff_norm_Linf_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	T diff_norm_Linf(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		return evaluate_arr_stats(vec_diff_norm_Linf_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename Slicing>
	array1d<T> diff_norm_Linf(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const caview2d<T, TOrd, RIndexer0, RIndexer1>& b, Slicing)
	{
		return evaluate_arr_slice_stats(vec_diff_norm_Linf_ftor<T>(), a, b, Slicing());
	}

}

#endif 
