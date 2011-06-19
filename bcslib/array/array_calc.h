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
#include <bcslib/veccomp/veccalc.h>

#include <bcslib/base/arg_check.h>
#include <type_traits>

namespace bcs
{

	/******************************************************
	 *
	 *  Addition
	 *
	 ******************************************************/


	template<class Arr1, class Arr2>
	inline typename lazy_enable_if<is_compatible_array_views<Arr1, Arr2>::value,
	_arr_op_aa<Arr1, Arr2, vec_add_functor>>::type
	operator + (const Arr1& a1, const Arr2& a2)
	{
		return _arr_op_aa<Arr1, Arr2, vec_add_functor>::default_evaluate(a1, a2);
	}


}

#endif 
