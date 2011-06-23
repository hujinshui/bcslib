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
#include <bcslib/array/generic_array_eval.h>

#include <bcslib/veccomp/vecstat_functors.h>


namespace bcs
{
	// sum, dot & mean

	// sum

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_stats_resultT<vec_sum_ftor, Arr> >::type
	sum_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_stats(vec_sum_ftor<value_type>(), a);
	}

	template<class Arr, typename Slicing>
	inline typename lazy_enable_if<is_array_view_ndim<Arr, 2>::value,
	array_slice_stats_resultT<vec_sum_ftor, Slicing, Arr> >::type
	sum_arr(const Arr& a, Slicing)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_slice_stats(vec_sum_ftor<value_type>(), a, Slicing());
	}


	// vdot

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_stats_resultT<vec_dot_prod_ftor, Arr1, Arr2> >::type
	vdot_arr(const Arr1& a, const Arr2& b)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return evaluate_arr_stats(vec_dot_prod_ftor<value_type>(), a, b);
	}

	template<class Arr1, class Arr2, typename Slicing>
	inline typename lazy_enable_if<is_compatible_aviews_ndim<Arr1, Arr2, 2>::value,
	array_slice_stats_resultT<vec_dot_prod_ftor, Slicing, Arr1, Arr2> >::type
	vdot_arr(const Arr1& a, const Arr2& b, Slicing)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return evaluate_arr_slice_stats(vec_dot_prod_ftor<value_type>(), a, b, Slicing());
	}

	// sum_log

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_stats_resultT<vec_sum_log_ftor, Arr> >::type
	sum_log_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_stats(vec_sum_log_ftor<value_type>(), a);
	}

	template<class Arr, typename Slicing>
	inline typename lazy_enable_if<is_array_view_ndim<Arr, 2>::value,
	array_slice_stats_resultT<vec_sum_log_ftor, Slicing, Arr> >::type
	sum_log_arr(const Arr& a, Slicing)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_slice_stats(vec_sum_log_ftor<value_type>(), a, Slicing());
	}

	// sum_xlogy

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_stats_resultT<vec_sum_xlogy_ftor, Arr1, Arr2> >::type
	sum_xlogy_arr(const Arr1& a, const Arr2& b)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return evaluate_arr_stats(vec_sum_xlogy_ftor<value_type>(), a, b);
	}

	template<class Arr1, class Arr2, typename Slicing>
	inline typename lazy_enable_if<is_compatible_aviews_ndim<Arr1, Arr2, 2>::value,
	array_slice_stats_resultT<vec_sum_xlogy_ftor, Slicing, Arr1, Arr2> >::type
	sum_xlogy_arr(const Arr1& a, const Arr2& b, Slicing)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return evaluate_arr_slice_stats(vec_sum_xlogy_ftor<value_type>(), a, b, Slicing());
	}

	// mean

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_stats_resultT<vec_mean_ftor, Arr> >::type
	mean_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_stats(vec_mean_ftor<value_type>(), a);
	}

	template<class Arr, typename Slicing>
	inline typename lazy_enable_if<is_array_view_ndim<Arr, 2>::value,
	array_slice_stats_resultT<vec_mean_ftor, Slicing, Arr> >::type
	mean_arr(const Arr& a, Slicing)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_slice_stats(vec_mean_ftor<value_type>(), a, Slicing());
	}


	// min, max & median

	// min

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_stats_resultT<vec_min_ftor, Arr> >::type
	min_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_stats(vec_min_ftor<value_type>(), a);
	}

	template<class Arr, typename Slicing>
	inline typename lazy_enable_if<is_array_view_ndim<Arr, 2>::value,
	array_slice_stats_resultT<vec_min_ftor, Slicing, Arr> >::type
	min_arr(const Arr& a, Slicing)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_slice_stats(vec_min_ftor<value_type>(), a, Slicing());
	}

	// max

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_stats_resultT<vec_max_ftor, Arr> >::type
	max_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_stats(vec_max_ftor<value_type>(), a);
	}

	template<class Arr, typename Slicing>
	inline typename lazy_enable_if<is_array_view_ndim<Arr, 2>::value,
	array_slice_stats_resultT<vec_max_ftor, Slicing, Arr> >::type
	max_arr(const Arr& a, Slicing)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_slice_stats(vec_max_ftor<value_type>(), a, Slicing());
	}

	// minmax

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_stats_resultT<vec_minmax_ftor, Arr> >::type
	minmax_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_stats(vec_minmax_ftor<value_type>(), a);
	}

	template<class Arr, typename Slicing>
	inline typename lazy_enable_if<is_array_view_ndim<Arr, 2>::value,
	array_slice_stats_resultT<vec_minmax_ftor, Slicing, Arr> >::type
	minmax_arr(const Arr& a, Slicing)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return evaluate_arr_slice_stats(vec_minmax_ftor<value_type>(), a, Slicing());
	}

}


#endif 
