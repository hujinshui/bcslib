/**
 * @file logical_array_ops.h
 *
 * Logical array operations
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_LOGICAL_ARRAY_OPS_H_
#define BCSLIB_LOGICAL_ARRAY_OPS_H_


#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <bcslib/array/generic_array_eval.h>
#include <bcslib/veccomp/logical_vecops.h>

namespace bcs
{
	// not

	template<class TIndexer>
	inline array1d<bool> operator ! (const caview1d<bool, TIndexer>& a)
	{
		return transform_arr(vec_not_ftor(), a);
	}

	template<typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<bool, TOrd> operator ! (const caview2d<bool, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_not_ftor(), a);
	}


	// and

	template<class LIndexer, class RIndexer>
	inline array1d<bool> operator & (const caview1d<bool, LIndexer>& a, const caview1d<bool, RIndexer>& b)
	{
		return transform_arr(vec_and_ftor(), a, b);
	}

	template<typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> operator & (
			const caview2d<bool, TOrd, LIndexer0, LIndexer1>& a,
			const caview2d<bool, TOrd, RIndexer0, RIndexer1>& b)
	{
		return transform_arr(vec_and_ftor(), a, b);
	}


	// or

	template<class LIndexer, class RIndexer>
	inline array1d<bool> operator | (const caview1d<bool, LIndexer>& a, const caview1d<bool, RIndexer>& b)
	{
		return transform_arr(vec_or_ftor(), a, b);
	}

	template<typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> operator | (
			const caview2d<bool, TOrd, LIndexer0, LIndexer1>& a,
			const caview2d<bool, TOrd, RIndexer0, RIndexer1>& b)
	{
		return transform_arr(vec_or_ftor(), a, b);
	}


	// xor

	template<class LIndexer, class RIndexer>
	inline array1d<bool> operator ^ (const caview1d<bool, LIndexer>& a, const caview1d<bool, RIndexer>& b)
	{
		return transform_arr(vec_xor_ftor(), a, b);
	}

	template<typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> operator ^ (
			const caview2d<bool, TOrd, LIndexer0, LIndexer1>& a,
			const caview2d<bool, TOrd, RIndexer0, RIndexer1>& b)
	{
		return transform_arr(vec_xor_ftor(), a, b);
	}


	// all

	template<class TIndexer>
	bool all(const caview1d<bool, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_all_ftor(), a);
	}

	template<typename TOrd, class TIndexer0, class TIndexer1>
	bool all(const caview2d<bool, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_all_ftor(), a);
	}

	template<typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<bool> all(const caview2d<bool, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_all_ftor(), a, Slicing());
	}


	// any

	template<class TIndexer>
	bool any(const caview1d<bool, TIndexer>& a)
	{
		return evaluate_arr_stats(vec_any_ftor(), a);
	}

	template<typename TOrd, class TIndexer0, class TIndexer1>
	bool any(const caview2d<bool, TOrd, TIndexer0, TIndexer1>& a)
	{
		return evaluate_arr_stats(vec_any_ftor(), a);
	}

	template<typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<bool> any(const caview2d<bool, TOrd, TIndexer0, TIndexer1>& a, Slicing)
	{
		return evaluate_arr_slice_stats(vec_any_ftor(), a, Slicing());
	}


	// count_true

	template<typename TCount, class TIndexer>
	TCount count_true(const caview1d<bool, TIndexer>& a, TCount c0)
	{
		return evaluate_arr_stats(vec_true_counter<TCount>(c0), a);
	}

	template<typename TCount, typename TOrd, class TIndexer0, class TIndexer1>
	TCount count_true(const caview2d<bool, TOrd, TIndexer0, TIndexer1>& a, TCount c0)
	{
		return evaluate_arr_stats(vec_true_counter<TCount>(c0), a);
	}

	template<typename TCount, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<TCount> count_true(const caview2d<bool, TOrd, TIndexer0, TIndexer1>& a, TCount c0, Slicing)
	{
		return evaluate_arr_slice_stats(vec_true_counter<TCount>(c0), a, Slicing());
	}


	// count_false

	template<typename TCount, class TIndexer>
	TCount count_false(const caview1d<bool, TIndexer>& a, TCount c0)
	{
		return evaluate_arr_stats(vec_false_counter<TCount>(c0), a);
	}

	template<typename TCount, typename TOrd, class TIndexer0, class TIndexer1>
	TCount count_false(const caview2d<bool, TOrd, TIndexer0, TIndexer1>& a, TCount c0)
	{
		return evaluate_arr_stats(vec_false_counter<TCount>(c0), a);
	}

	template<typename TCount, typename TOrd, class TIndexer0, class TIndexer1, typename Slicing>
	array1d<TCount> count_false(const caview2d<bool, TOrd, TIndexer0, TIndexer1>& a, TCount c0, Slicing)
	{
		return evaluate_arr_slice_stats(vec_false_counter<TCount>(c0), a, Slicing());
	}

}

#endif 
