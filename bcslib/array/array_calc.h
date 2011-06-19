/**
 * @file array_calc.h
 *
 * Vectorized calculation on arrays
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_CALC_H
#define BCSLIB_ARRAY_CALC_H

#include <bcslib/array/array_expr_base.h>
#include <bcslib/veccomp/veccalc_functors.h>

#include <bcslib/base/arg_check.h>
#include <type_traits>

namespace bcs
{

	/******************************************************
	 *
	 *  add, sub, mul, div, negate
	 *
	 ******************************************************/

	// addition

	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	_arr_binop<Arr1, Arr2, vec_vec_add>>::type
	add_arr_arr (const Arr1& a1, const Arr2& a2)
	{
		return _arr_binop<Arr1, Arr2, vec_vec_add>::default_evaluate(a1, a2);
	}

	template<class Arr, typename T>
	inline typename lazy_enable_if<is_compatible_aview_v<Arr, T>::value,
	_arr_uniop<Arr, vec_sca_add>>::type
	add_arr_sca (const Arr& a1, const T& x2)
	{
		return _arr_uniop<Arr, vec_sca_add>::evaluate_with_scalar(a1, x2);
	}

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> operator + (const aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return add_arr_arr(lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator + (const aview1d<T, LIndexer>& lhs, const T& v2)
	{
		return add_arr_sca(lhs, v2);
	}

	template<typename T, class RIndexer>
	inline array1d<T> operator + (const T& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return add_arr_sca(rhs, lhs);
	}


}

#endif 
