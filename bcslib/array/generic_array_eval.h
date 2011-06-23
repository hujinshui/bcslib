/**
 * @file generic_array_eval.h
 *
 * Generic functions for array-based evaluation
 * 
 * @author Dahua Lin
 */

#ifndef GENERIC_ARRAY_EVAL_H_
#define GENERIC_ARRAY_EVAL_H_

#include <bcslib/base/arg_check.h>
#include <bcslib/array/array_base.h>
#include <bcslib/array/generic_array_functions.h>

#include <type_traits>

namespace bcs
{

	/******************************************************
	 *
	 *  Array Transform
	 *
	 ******************************************************/

	template<class VecFunc, class Arr1, class Arr2=nil_type> struct array_transform_result;

	template<class VecFunc, class Arr1>
	struct array_transform_result<VecFunc, Arr1, nil_type>
	{
		static_assert(is_array_view<Arr1>::value, "Arr1 should be an array view type");

		typedef typename array_view_traits<Arr1>::value_type arg1_value_type;
		typedef typename std::result_of<VecFunc(arg1_value_type)>::type result_value_type;

		typedef typename array_creater<Arr1>::template remap<result_value_type>::type _rcreater;
		typedef typename _rcreater::result_type type;

		typedef typename array_view_traits<Arr1>::shape_type shape_type;

		static type create(const shape_type& shape)
		{
			return _rcreater::create(shape);
		}
	};

	template<class VecFunc, class Arr1, class Arr2>
	struct array_transform_result
	{
		static_assert(is_compatible_aviews<Arr1, Arr2>::value,
				"Arr1 and Arr2 should be compatible array view types.");

		typedef typename array_view_traits<Arr1>::value_type arg1_value_type;
		typedef typename array_view_traits<Arr2>::value_type arg2_value_type;
		typedef typename std::result_of<VecFunc(arg1_value_type, arg2_value_type)>::type result_value_type;

		typedef typename array_creater<Arr1>::template remap<result_value_type>::type _rcreater;
		typedef typename _rcreater::result_type type;

		typedef typename array_view_traits<Arr1>::shape_type shape_type;

		static type create(const shape_type& shape)
		{
			return _rcreater::create(shape);
		}
	};

	template<template<typename U> class VecFuncTemplate, class Arr1, class Arr2=nil_type>
	struct array_transform_resultT
	{
		typedef typename array_view_traits<Arr1>::value_type arg1_value_type;
		typedef VecFuncTemplate<arg1_value_type> vecfunc_type;
		typedef typename array_transform_result<vecfunc_type, Arr1, Arr2>::type type;
	};


	template<class VecFunc, class Arr1>
	typename array_transform_result<VecFunc, Arr1>::type
	transform_arr(VecFunc vfunc, const Arr1& a1)
	{
		typedef typename array_transform_result<VecFunc, Arr1>::type result_t;

		scoped_aview_read_proxy<Arr1> a1p(a1);
		result_t r = array_transform_result<VecFunc, Arr1>::create(get_array_shape(a1));
		vfunc(get_num_elems(a1), a1p.pbase(), ptr_base(r));

		return r;
	}

	template<class VecFunc, class Arr1, class Arr2>
	typename array_transform_result<VecFunc, Arr1, Arr2>::type
	transform_arr(VecFunc vfunc, const Arr1& a1, const Arr2& a2)
	{
		check_arg(get_array_shape(a1) == get_array_shape(a2),
				"The shapes of operand arrays are inconsistent.");

		typedef typename array_transform_result<VecFunc, Arr1, Arr2>::type result_t;

		scoped_aview_read_proxy<Arr1> a1p(a1);
		scoped_aview_read_proxy<Arr2> a2p(a2);
		result_t r = array_transform_result<VecFunc, Arr1, Arr2>::create(get_array_shape(a1));
		vfunc(get_num_elems(a1), a1p.pbase(), a2p.pbase(), ptr_base(r));

		return r;
	}

	template<class VecFunc, class Arr1>
	void inplace_transform_arr(VecFunc vfunc, Arr1& a1)
	{
		scoped_aview_write_proxy<Arr1> a1p(a1, true);
		vfunc(get_num_elems(a1), a1p.pbase());
		a1p.commit();
	}

	template<class VecFunc, class Arr1, class Arr2>
	void inplace_transform_arr(VecFunc vfunc, Arr1& a1, const Arr2& a2)
	{
		check_arg(get_array_shape(a1) == get_array_shape(a2),
				"The shapes of operand arrays are inconsistent.");

		scoped_aview_write_proxy<Arr1> a1p(a1, true);
		scoped_aview_read_proxy<Arr2> a2p(a2);

		vfunc(get_num_elems(a1), a1p.pbase(), a2p.pbase());
		a1p.commit();
	}

}

#endif 
