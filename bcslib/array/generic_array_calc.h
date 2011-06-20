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
#include <bcslib/array/array_expr_base.h>

#include <bcslib/veccomp/veccalc_functors.h>

namespace bcs
{
	/********************************************
	 *
	 *  Arithmetic Calculation
	 *
	 *******************************************/

	// addition

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_binop<Arr1, Arr2, vec_vec_add_ftor>>::type
	add_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		return _arr_binop<Arr1, Arr2, vec_vec_add_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_uniop<Arr, vec_sca_add_ftor>>::type
	add_arr_sca(const Arr& a, const T& x)
	{
		return _arr_uniop<Arr, vec_sca_add_ftor>::evaluate_with_scalar(a, x);
	}

	template<class Arr1, typename Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value, void>::type
	add_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		_arr_ipop_R1<Arr1, Arr2, vec_vec_add_ip_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	add_arr_sca_inplace(Arr& a, const T& x)
	{
		_arr_ipop<Arr, vec_sca_add_ip_ftor>::evaluate_with_scalar(a, x);
	}


	// subtraction

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_binop<Arr1, Arr2, vec_vec_sub_ftor>>::type
	sub_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		return _arr_binop<Arr1, Arr2, vec_vec_sub_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_uniop<Arr, vec_sca_sub_ftor>>::type
	sub_arr_sca(const Arr& a, const T& x)
	{
		return _arr_uniop<Arr, vec_sca_sub_ftor>::evaluate_with_scalar(a, x);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_uniop<Arr, sca_vec_sub_ftor>>::type
	sub_sca_arr(const T& x, const Arr& a)
	{
		return _arr_uniop<Arr, sca_vec_sub_ftor>::evaluate_with_scalar(a, x);
	}

	template<class Arr1, typename Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value, void>::type
	sub_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		_arr_ipop_R1<Arr1, Arr2, vec_vec_sub_ip_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	sub_arr_sca_inplace(Arr& a, const T& x)
	{
		_arr_ipop<Arr, vec_sca_sub_ip_ftor>::evaluate_with_scalar(a, x);
	}


	// multiplication

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_binop<Arr1, Arr2, vec_vec_mul_ftor>>::type
	mul_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		return _arr_binop<Arr1, Arr2, vec_vec_mul_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_uniop<Arr, vec_sca_mul_ftor>>::type
	mul_arr_sca(const Arr& a, const T& x)
	{
		return _arr_uniop<Arr, vec_sca_mul_ftor>::evaluate_with_scalar(a, x);
	}

	template<class Arr1, typename Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value, void>::type
	mul_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		_arr_ipop_R1<Arr1, Arr2, vec_vec_mul_ip_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	mul_arr_sca_inplace(Arr& a, const T& x)
	{
		_arr_ipop<Arr, vec_sca_mul_ip_ftor>::evaluate_with_scalar(a, x);
	}


	// division

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_binop<Arr1, Arr2, vec_vec_div_ftor>>::type
	div_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		return _arr_binop<Arr1, Arr2, vec_vec_div_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_uniop<Arr, vec_sca_div_ftor>>::type
	div_arr_sca(const Arr& a, const T& x)
	{
		return _arr_uniop<Arr, vec_sca_div_ftor>::evaluate_with_scalar(a, x);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_uniop<Arr, sca_vec_div_ftor>>::type
	div_sca_arr(const T& x, const Arr& a)
	{
		return _arr_uniop<Arr, sca_vec_div_ftor>::evaluate_with_scalar(a, x);
	}

	template<class Arr1, typename Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value, void>::type
	div_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		_arr_ipop_R1<Arr1, Arr2, vec_vec_div_ip_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	div_arr_sca_inplace(Arr& a, const T& x)
	{
		_arr_ipop<Arr, vec_sca_div_ip_ftor>::evaluate_with_scalar(a, x);
	}


	// negation

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_neg_ftor>>::type
	neg_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_neg_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	neg_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_neg_ip_ftor>::default_evaluate(a);
	}


	// absolute value

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_abs_ftor>>::type
	abs_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_abs_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	abs_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_abs_ftor>::default_evaluate(a);
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
	_arr_uniop<Arr, vec_sqr_ftor>>::type
	sqr_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_sqr_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	sqr_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_sqr_ftor>::default_evaluate(a);
	}

	// sqrt

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_sqrt_ftor>>::type
	sqrt_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_sqrt_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	sqrt_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_sqrt_ftor>::default_evaluate(a);
	}


	// rcp

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_rcp_ftor>>::type
	rcp_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_rcp_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	rcp_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_rcp_ftor>::default_evaluate(a);
	}

	// rsqrt

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_rsqrt_ftor>>::type
	rsqrt_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_rsqrt_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	rsqrt_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_rsqrt_ftor>::default_evaluate(a);
	}


	// pow

	template<class Arr, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr, Arr2>::value,
	_arr_binop<Arr, Arr2, vec_pow_ftor>>::type
	pow_arr(const Arr& a, const Arr2& e)
	{
		return _arr_binop<Arr, Arr2, vec_pow_ftor>::default_evaluate(a, e);
	}

	template<class Arr, class Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr, Arr2>::value, void>::type
	pow_arr_inplace(Arr& a, const Arr2& e)
	{
		_arr_ipop_R1<Arr, Arr2, vec_pow_ftor>::default_evaluate(a, e);
	}


	// pow with constant exponent

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_uniop<Arr, vec_sca_pow_ftor>>::type
	pow_arr_sca(const Arr& a, const T& e)
	{
		return _arr_uniop<Arr, vec_sca_pow_ftor>::evaluate_with_scalar(a, e);
	}


	template<class Arr, typename T>
	inline typename std::enable_if<is_compatible_aview_v<Arr, T>::value, void>::type
	pow_arr_sca_inplace(Arr& a, const T& x)
	{
		_arr_ipop<Arr, vec_sca_pow_ftor>::evaluate_with_scalar(a, x);
	}


	// exponential and logarithm functions

	// exp

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_exp_ftor>>::type
	exp_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_exp_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	exp_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_exp_ftor>::default_evaluate(a);
	}

	// log

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_log_ftor>>::type
	log_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_log_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	log_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_log_ftor>::default_evaluate(a);
	}

	// log10

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_log10_ftor>>::type
	log10_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_log10_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	log10_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_log10_ftor>::default_evaluate(a);
	}


	// rounding functions

	// floor

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_floor_ftor>>::type
	floor_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_floor_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	floor_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_floor_ftor>::default_evaluate(a);
	}


	// ceil

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_ceil_ftor>>::type
	ceil_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_ceil_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	ceil_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_ceil_ftor>::default_evaluate(a);
	}


	// trigonometric functions

	// sin

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_sin_ftor>>::type
	sin_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_sin_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	sin_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_sin_ftor>::default_evaluate(a);
	}

	// cos

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_cos_ftor>>::type
	cos_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_cos_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	cos_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_cos_ftor>::default_evaluate(a);
	}

	// tan

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_tan_ftor>>::type
	tan_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_tan_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	tan_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_tan_ftor>::default_evaluate(a);
	}

	// asin

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_asin_ftor>>::type
	asin_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_asin_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	asin_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_asin_ftor>::default_evaluate(a);
	}

	// acos

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_acos_ftor>>::type
	acos_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_acos_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	acos_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_acos_ftor>::default_evaluate(a);
	}

	// atan

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_atan_ftor>>::type
	atan_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_atan_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	atan_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_atan_ftor>::default_evaluate(a);
	}

	// atan2

	template<class Arr, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr, Arr2>::value,
	_arr_binop<Arr, Arr2, vec_atan2_ftor>>::type
	atan2_arr(const Arr& a, const Arr2& b)
	{
		return _arr_binop<Arr, Arr2, vec_atan2_ftor>::default_evaluate(a, b);
	}


	// sinh

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_sinh_ftor>>::type
	sinh_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_sinh_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	sinh_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_sinh_ftor>::default_evaluate(a);
	}

	// cosh

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_cosh_ftor>>::type
	cosh_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_cosh_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	cosh_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_cosh_ftor>::default_evaluate(a);
	}

	// tanh

	template<class Arr>
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_uniop<Arr, vec_tanh_ftor>>::type
	tanh_arr(const Arr& a)
	{
		return _arr_uniop<Arr, vec_tanh_ftor>::default_evaluate(a);
	}

	template<class Arr>
	inline typename std::enable_if<is_array_view<Arr>::value, void>::type
	tanh_arr_inplace(Arr& a)
	{
		_arr_ipop<Arr, vec_tanh_ftor>::default_evaluate(a);
	}

}

#endif 
