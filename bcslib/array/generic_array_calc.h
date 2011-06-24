/**
 * @file generic_array_calc.h
 *
 * Generic functions for array calculation
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_GENERIC_ARRAY_CALC_H
#define BCSLIB_GENERIC_ARRAY_CALC_H

#include <bcslib/array/array_base.h>
#include <bcslib/veccomp/veccalc.h>

#include <bcslib/array/generic_array_eval.h>

namespace bcs
{
	/********************************************
	 *
	 *  Comparison
	 *
	 *******************************************/

	// eq

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_eq_ftor, Arr1, Arr2>>::type
	eq_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_eq_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_eq_ftor, Arr>>::type
	eq_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_eq_ftor<value_type>(x), a);
	}

	// ne

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_ne_ftor, Arr1, Arr2>>::type
	ne_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_ne_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_ne_ftor, Arr>>::type
	ne_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_ne_ftor<value_type>(x), a);
	}

	// gt

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_gt_ftor, Arr1, Arr2>>::type
	gt_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_gt_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_gt_ftor, Arr>>::type
	gt_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_gt_ftor<value_type>(x), a);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<sca_vec_gt_ftor, Arr>>::type
	gt_sca_arr(const T& x, const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(sca_vec_gt_ftor<value_type>(x), a);
	}

	// ge

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_ge_ftor, Arr1, Arr2>>::type
	ge_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_ge_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_ge_ftor, Arr>>::type
	ge_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_ge_ftor<value_type>(x), a);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<sca_vec_ge_ftor, Arr>>::type
	ge_sca_arr(const T& x, const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(sca_vec_ge_ftor<value_type>(x), a);
	}

	// lt

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_lt_ftor, Arr1, Arr2>>::type
	lt_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_lt_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_lt_ftor, Arr>>::type
	lt_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_lt_ftor<value_type>(x), a);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<sca_vec_lt_ftor, Arr>>::type
	lt_sca_arr(const T& x, const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(sca_vec_lt_ftor<value_type>(x), a);
	}

	// le

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_le_ftor, Arr1, Arr2>>::type
	le_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_le_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_le_ftor, Arr>>::type
	le_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_le_ftor<value_type>(x), a);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<sca_vec_le_ftor, Arr>>::type
	le_sca_arr(const T& x, const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(sca_vec_le_ftor<value_type>(x), a);
	}


	// max_each

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_max_each_ftor, Arr1, Arr2>>::type
	max_each_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_max_each_ftor<value_type>(), a1, a2);
	}

	// min_each

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_min_each_ftor, Arr1, Arr2>>::type
	min_each_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_min_each_ftor<value_type>(), a1, a2);
	}

	/********************************************
	 *
	 *  Bounding
	 *
	 *******************************************/

	// lbound

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_lbound_ftor, Arr>>::type
	lbound_arr(const Arr& a, const T& lb)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_lbound_ftor<value_type>(lb), a);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	lbound_arr_inplace(Arr& a, const T& lb)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_lbound_ftor<value_type>(lb), a);
	}

	// ubound

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_ubound_ftor, Arr>>::type
	ubound_arr(const Arr& a, const T& ub)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_ubound_ftor<value_type>(ub), a);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	ubound_arr_inplace(Arr& a, const T& ub)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_ubound_ftor<value_type>(ub), a);
	}

	// rgn_bound

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_rgn_bound_ftor, Arr>>::type
	rgn_bound_arr(const Arr& a, const T& lb, const T& ub)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_rgn_bound_ftor<value_type>(lb, ub), a);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	rgn_bound_arr_inplace(Arr& a, const T& lb, const T& ub)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_rgn_bound_ftor<value_type>(lb, ub), a);
	}

	// abound

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_abound_ftor, Arr>>::type
	abound_arr(const Arr& a, const T& ab)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_abound_ftor<value_type>(ab), a);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	abound_arr_inplace(Arr& a, const T& ab)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_abound_ftor<value_type>(ab), a);
	}


	/********************************************
	 *
	 *  Arithmetic Calculation
	 *
	 *******************************************/

	// addition

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_add_ftor, Arr1, Arr2>>::type
	add_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_add_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_add_ftor, Arr>>::type
	add_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_add_ftor<value_type>(x), a);
	}

	template<class Arr1, typename Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value, void>::type
	add_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		inplace_transform_arr(vec_vec_add_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	add_arr_sca_inplace(Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_sca_add_ftor<value_type>(x), a);
	}


	// subtraction

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_sub_ftor, Arr1, Arr2>>::type
	sub_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_sub_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_sub_ftor, Arr>>::type
	sub_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_sub_ftor<value_type>(x), a);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<sca_vec_sub_ftor, Arr>>::type
	sub_sca_arr(const T& x, const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(sca_vec_sub_ftor<value_type>(x), a);
	}

	template<class Arr1, typename Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value, void>::type
	sub_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		inplace_transform_arr(vec_vec_sub_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	sub_arr_sca_inplace(Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_sca_sub_ftor<value_type>(x), a);
	}


	// multiplication

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_mul_ftor, Arr1, Arr2>>::type
	mul_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_mul_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_mul_ftor, Arr>>::type
	mul_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_mul_ftor<value_type>(x), a);
	}

	template<class Arr1, typename Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value, void>::type
	mul_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		inplace_transform_arr(vec_vec_mul_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	mul_arr_sca_inplace(Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_sca_mul_ftor<value_type>(x), a);
	}


	// division

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_vec_div_ftor, Arr1, Arr2>>::type
	div_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_vec_div_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_div_ftor, Arr>>::type
	div_arr_sca(const Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_div_ftor<value_type>(x), a);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<sca_vec_div_ftor, Arr>>::type
	div_sca_arr(const T& x, const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(sca_vec_div_ftor<value_type>(x), a);
	}

	template<class Arr1, typename Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value, void>::type
	div_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		inplace_transform_arr(vec_vec_div_ftor<value_type>(), a1, a2);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	div_arr_sca_inplace(Arr& a, const T& x)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_sca_div_ftor<value_type>(x), a);
	}


	// negation

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_neg_ftor, Arr>>::type
	neg_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_neg_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	neg_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_neg_ftor<value_type>(), a);
	}


	// absolute value

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_abs_ftor, Arr>>::type
	abs_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_abs_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	abs_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_abs_ftor<value_type>(), a);
	}



	/********************************************
	 *
	 *  Other Elementary functions
	 *
	 *******************************************/

	// power and root functions

	// sqr

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_sqr_ftor, Arr>>::type
	sqr_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sqr_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	sqr_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_sqr_ftor<value_type>(), a);
	}

	// sqrt

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_sqrt_ftor, Arr>>::type
	sqrt_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sqrt_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	sqrt_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_sqrt_ftor<value_type>(), a);
	}


	// rcp

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_rcp_ftor, Arr>>::type
	rcp_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_rcp_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	rcp_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_rcp_ftor<value_type>(), a);
	}

	// rsqrt

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_rsqrt_ftor, Arr>>::type
	rsqrt_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_rsqrt_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	rsqrt_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_rsqrt_ftor<value_type>(), a);
	}


	// pow

	template<class Arr, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr, Arr2>::value,
	array_transform_resultT<vec_pow_ftor, Arr, Arr2>>::type
	pow_arr(const Arr& a, const Arr2& e)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_pow_ftor<value_type>(), a, e);
	}

	template<class Arr, class Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr, Arr2>::value, void>::type
	pow_arr_inplace(Arr& a, const Arr2& e)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_pow_ftor<value_type>(), a, e);
	}


	// pow with constant exponent

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	array_transform_resultT<vec_sca_pow_ftor, Arr>>::type
	pow_arr_sca(const Arr& a, const T& e)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sca_pow_ftor<value_type>(e), a);
	}


	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	pow_arr_sca_inplace(Arr& a, const T& e)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_sca_pow_ftor<value_type>(e), a);
	}


	// exponential and logarithm functions

	// exp

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_exp_ftor, Arr>>::type
	exp_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_exp_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	exp_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_exp_ftor<value_type>(), a);
	}

	// log

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_log_ftor, Arr>>::type
	log_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_log_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	log_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_log_ftor<value_type>(), a);
	}

	// log10

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_log10_ftor, Arr>>::type
	log10_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_log10_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	log10_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_log10_ftor<value_type>(), a);
	}


	// rounding functions

	// floor

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_floor_ftor, Arr>>::type
	floor_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_floor_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	floor_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_floor_ftor<value_type>(), a);
	}


	// ceil

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_ceil_ftor, Arr>>::type
	ceil_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_ceil_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	ceil_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_ceil_ftor<value_type>(), a);
	}


	// trigonometric functions

	// sin

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_sin_ftor, Arr>>::type
	sin_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sin_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	sin_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_sin_ftor<value_type>(), a);
	}

	// cos

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_cos_ftor, Arr>>::type
	cos_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_cos_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	cos_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_cos_ftor<value_type>(), a);
	}

	// tan

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_tan_ftor, Arr>>::type
	tan_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_tan_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	tan_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_tan_ftor<value_type>(), a);
	}

	// asin

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_asin_ftor, Arr>>::type
	asin_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_asin_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	asin_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_asin_ftor<value_type>(), a);
	}

	// acos

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_acos_ftor, Arr>>::type
	acos_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_acos_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	acos_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_acos_ftor<value_type>(), a);
	}

	// atan

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_atan_ftor, Arr>>::type
	atan_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_atan_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	atan_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_atan_ftor<value_type>(), a);
	}

	// atan2

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	array_transform_resultT<vec_atan2_ftor, Arr1, Arr2>>::type
	atan2_arr(const Arr1& a1, const Arr2& a2)
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		return transform_arr(vec_atan2_ftor<value_type>(), a1, a2);
	}


	// sinh

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_sinh_ftor, Arr>>::type
	sinh_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_sinh_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	sinh_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_sinh_ftor<value_type>(), a);
	}

	// cosh

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_cosh_ftor, Arr>>::type
	cosh_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_cosh_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	cosh_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_cosh_ftor<value_type>(), a);
	}

	// tanh

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	array_transform_resultT<vec_tanh_ftor, Arr>>::type
	tanh_arr(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		return transform_arr(vec_tanh_ftor<value_type>(), a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	tanh_arr_inplace(Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		inplace_transform_arr(vec_tanh_ftor<value_type>(), a);
	}

}

#endif 
