/**
 * @file generic_array_stat.h
 *
 * Generic functions for evaluating array stats
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_GENERIC_ARRAY_STAT_H_
#define BCSLIB_GENERIC_ARRAY_STAT_H_

#include <bcslib/array/array_base.h>
#include <bcslib/array/array_expr_base.h>

#include <bcslib/veccomp/vecstat_functors.h>


namespace bcs
{
	// sum, dot & mean

	// sum

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_uniarr_stat<Arr, vec_sum_ftor> >::type
	sum_arr(const Arr& a)
	{
		return _uniarr_stat<Arr, vec_sum_ftor>::default_evaluate(a);
	}

	template<class Arr, typename Slicing>
	inline typename lazy_enable_if<is_array_view_ndim<Arr, 2>::value,
	_uniarr_slice_stat<Arr, Slicing, vec_sum_ftor> >::type
	sum_arr(const Arr& a, Slicing)
	{
		return _uniarr_slice_stat<Arr, Slicing, vec_sum_ftor>::default_evaluate(a);
	}


	// vdot

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_binarr_stat<Arr1, Arr2, vec_dot_prod_ftor> >::type
	vdot_arr(const Arr1& a, const Arr2& b)
	{
		return _binarr_stat<Arr1, Arr2, vec_dot_prod_ftor>::default_evaluate(a, b);
	}

	template<class Arr1, class Arr2, typename Slicing>
	inline typename lazy_enable_if<is_compatible_aviews_ndim<Arr1, Arr2, 2>::value,
	_binarr_slice_stat<Arr1, Arr2, Slicing, vec_dot_prod_ftor> >::type
	vdot_arr(const Arr1& a, const Arr2& b, Slicing)
	{
		return _binarr_slice_stat<Arr1, Arr2, Slicing, vec_dot_prod_ftor>::default_evaluate(a, b);
	}

	// sum_log

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_uniarr_stat<Arr, vec_sum_log_ftor> >::type
	sum_log_arr(const Arr& a)
	{
		return _uniarr_stat<Arr, vec_sum_log_ftor>::default_evaluate(a);
	}

	template<class Arr, typename Slicing>
	inline typename lazy_enable_if<is_array_view_ndim<Arr, 2>::value,
	_uniarr_slice_stat<Arr, Slicing, vec_sum_log_ftor> >::type
	sum_log_arr(const Arr& a, Slicing)
	{
		return _uniarr_slice_stat<Arr, Slicing, vec_sum_log_ftor>::default_evaluate(a);
	}

	// sum_xlogy

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_binarr_stat<Arr1, Arr2, vec_sum_xlogy_ftor> >::type
	sum_xlogy_arr(const Arr1& a, const Arr2& b)
	{
		return _binarr_stat<Arr1, Arr2, vec_sum_xlogy_ftor>::default_evaluate(a, b);
	}

	template<class Arr1, class Arr2, typename Slicing>
	inline typename lazy_enable_if<is_compatible_aviews_ndim<Arr1, Arr2, 2>::value,
	_binarr_slice_stat<Arr1, Arr2, Slicing, vec_sum_xlogy_ftor> >::type
	sum_xlogy_arr(const Arr1& a, const Arr2& b, Slicing)
	{
		return _binarr_slice_stat<Arr1, Arr2, Slicing, vec_sum_xlogy_ftor>::default_evaluate(a, b);
	}

	// mean

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_uniarr_stat<Arr, vec_mean_ftor> >::type
	mean_arr(const Arr& a)
	{
		return _uniarr_stat<Arr, vec_mean_ftor>::default_evaluate(a);
	}

	template<class Arr, typename Slicing>
	inline typename lazy_enable_if<is_array_view_ndim<Arr, 2>::value,
	_uniarr_slice_stat<Arr, Slicing, vec_mean_ftor> >::type
	mean_arr(const Arr& a, Slicing)
	{
		return _uniarr_slice_stat<Arr, Slicing, vec_mean_ftor>::default_evaluate(a);
	}

}


#endif 
