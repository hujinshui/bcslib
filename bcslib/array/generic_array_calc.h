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
	_arr_binop<Arr1, Arr2, vec_vec_add>>::type
	add_arr_arr(const Arr1& a1, const Arr2& a2)
	{
		return _arr_binop<Arr1, Arr2, vec_vec_add>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_uniop<Arr, vec_sca_add>>::type
	add_arr_sca(const Arr& a, const T& x)
	{
		return _arr_uniop<Arr, vec_sca_add>::evaluate_with_scalar(a, x);
	}

	template<class Arr1, typename Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_ipop_R1<Arr1, Arr2, vec_vec_add_ip>>::type
	add_arr_arr_inplace(Arr1& a1, const Arr2& a2)
	{
		return _arr_ipop_R1<Arr1, Arr2, vec_vec_add_ip>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_ipop<Arr, vec_sca_add_ip>>::type
	add_arr_sca_inplace(Arr& a, const T& x)
	{
		return _arr_ipop<Arr, vec_sca_add_ip>::evaluate_with_scalar(a, x);
	}



}

#endif 
