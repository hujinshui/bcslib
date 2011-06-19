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
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_ipop_R1<Arr1, Arr2, vec_vec_add_ip_ftor>>::type
	add_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		return _arr_ipop_R1<Arr1, Arr2, vec_vec_add_ip_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_ipop<Arr, vec_sca_add_ip_ftor>>::type
	add_arr_sca_inplace(Arr& a, const T& x)
	{
		return _arr_ipop<Arr, vec_sca_add_ip_ftor>::evaluate_with_scalar(a, x);
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
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_ipop_R1<Arr1, Arr2, vec_vec_sub_ip_ftor>>::type
	sub_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		return _arr_ipop_R1<Arr1, Arr2, vec_vec_sub_ip_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_ipop<Arr, vec_sca_sub_ip_ftor>>::type
	sub_arr_sca_inplace(Arr& a, const T& x)
	{
		return _arr_ipop<Arr, vec_sca_sub_ip_ftor>::evaluate_with_scalar(a, x);
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
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_ipop_R1<Arr1, Arr2, vec_vec_mul_ip_ftor>>::type
	mul_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		return _arr_ipop_R1<Arr1, Arr2, vec_vec_mul_ip_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_ipop<Arr, vec_sca_mul_ip_ftor>>::type
	mul_arr_sca_inplace(Arr& a, const T& x)
	{
		return _arr_ipop<Arr, vec_sca_mul_ip_ftor>::evaluate_with_scalar(a, x);
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
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_ipop_R1<Arr1, Arr2, vec_vec_div_ip_ftor>>::type
	div_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		return _arr_ipop_R1<Arr1, Arr2, vec_vec_div_ip_ftor>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_ipop<Arr, vec_sca_div_ip_ftor>>::type
	div_arr_sca_inplace(Arr& a, const T& x)
	{
		return _arr_ipop<Arr, vec_sca_div_ip_ftor>::evaluate_with_scalar(a, x);
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
	inline typename lazy_enable_if<is_array_view<Arr>::value,
	_arr_ipop<Arr, vec_neg_ip_ftor>>::type
	neg_arr_inplace(Arr& a)
	{
		return _arr_ipop<Arr, vec_neg_ip_ftor>::default_evaluate(a);
	}

}

#endif 
