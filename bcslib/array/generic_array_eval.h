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
	// slicing types

	struct per_row
	{
		template<class Arr>
		struct is_slice_major
		{
			static const bool value = is_row_major<Arr>::value;
		};

		static index_t get_num_slices(index_t m, index_t n) { return m; }
		static index_t get_slice_length(index_t m, index_t n) { return n; }
	};

	struct per_col
	{
		template<class Arr>
		struct is_slice_major
		{
			static const bool value = is_column_major<Arr>::value;
		};

		static index_t get_num_slices(index_t m, index_t n) { return n; }
		static index_t get_slice_length(index_t m, index_t n) { return m; }
	};

	template<typename T> struct is_slicing_type : public std::false_type { };
	template<> struct is_slicing_type<per_row> : public std::true_type { };
	template<> struct is_slicing_type<per_col> : public std::true_type { };


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
	inline typename std::enable_if<is_array_view<Arr1>::value,
	typename array_transform_result<VecFunc, Arr1>::type>::type
	transform_arr(VecFunc vfunc, const Arr1& a1)
	{
		typedef typename array_transform_result<VecFunc, Arr1>::type result_t;

		scoped_aview_read_proxy<Arr1> a1p(a1);
		result_t r = array_transform_result<VecFunc, Arr1>::create(get_array_shape(a1));
		vfunc(get_num_elems(a1), a1p.pbase(), ptr_base(r));

		return r;
	}

	template<class VecFunc, class Arr1, class Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	typename array_transform_result<VecFunc, Arr1, Arr2>::type>::type
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
	inline typename std::enable_if<is_array_view<Arr1>::value, void>::type
	inplace_transform_arr(VecFunc vfunc, Arr1& a1)
	{
		scoped_aview_write_proxy<Arr1> a1p(a1, true);
		vfunc(get_num_elems(a1), a1p.pbase());
		a1p.commit();
	}

	template<class VecFunc, class Arr1, class Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value, void>::type
	inplace_transform_arr(VecFunc vfunc, Arr1& a1, const Arr2& a2)
	{
		check_arg(get_array_shape(a1) == get_array_shape(a2),
				"The shapes of operand arrays are inconsistent.");

		scoped_aview_write_proxy<Arr1> a1p(a1, true);
		scoped_aview_read_proxy<Arr2> a2p(a2);

		vfunc(get_num_elems(a1), a1p.pbase(), a2p.pbase());
		a1p.commit();
	}


	/******************************************************
	 *
	 *  Array Statistics
	 *
	 ******************************************************/

	// stats types

	template<class VecFunc, class Arr1, class Arr2=nil_type> struct array_stats_result;

	template<class VecFunc, class Arr1>
	struct array_stats_result<VecFunc, Arr1, nil_type>
	{
		static_assert(is_array_view<Arr1>::value, "Arr1 should be an array view type");
		typedef typename array_view_traits<Arr1>::value_type arg1_value_type;
		typedef typename std::result_of<VecFunc(size_t, const arg1_value_type*)>::type type;
	};

	template<class VecFunc, class Arr1, class Arr2>
	struct array_stats_result
	{
		static_assert(is_compatible_aviews<Arr1, Arr2>::value, "Arr1 and Arr2 should be compatible array view types");
		typedef typename array_view_traits<Arr1>::value_type arg1_value_type;
		typedef typename array_view_traits<Arr2>::value_type arg2_value_type;
		typedef typename std::result_of<VecFunc(size_t, const arg1_value_type*, const arg2_value_type*)>::type type;
	};

	template<template<typename U> class VecFuncTemplate, class Arr1, class Arr2=nil_type>
	struct array_stats_resultT
	{
		typedef typename array_view_traits<Arr1>::value_type arg1_value_type;
		typedef VecFuncTemplate<arg1_value_type> vecfunc_type;
		typedef typename array_stats_result<vecfunc_type, Arr1, Arr2>::type type;
	};

	// slice stats types

	template<class VecFunc, typename Slicing, class Arr1, class Arr2=nil_type> struct array_slice_stats_result;

	template<class VecFunc, typename Slicing, class Arr1>
	struct array_slice_stats_result<VecFunc, Slicing, Arr1, nil_type>
	{
		static_assert(is_array_view_ndim<Arr1, 2>::value, "Arr1 should be a 2D array view type");

		typedef typename array_view_traits<Arr1>::value_type arg1_value_type;
		typedef typename std::result_of<VecFunc(size_t, const arg1_value_type*)>::type result_value_type;

		typedef typename array_creater<Arr1>::template remap<result_value_type, 1>::type _rcreater;
		typedef typename _rcreater::result_type type;
	};

	template<class VecFunc, typename Slicing, class Arr1, class Arr2>
	struct array_slice_stats_result
	{
		static_assert(is_compatible_aviews_ndim<Arr1, Arr2, 2>::value,
				"Arr1 and Arr2 should be compatible 2D array view types");

		typedef typename array_view_traits<Arr1>::value_type arg1_value_type;
		typedef typename array_view_traits<Arr2>::value_type arg2_value_type;
		typedef typename std::result_of<VecFunc(size_t, const arg1_value_type*, const arg2_value_type*)>::type result_value_type;

		typedef typename array_creater<Arr1>::template remap<result_value_type, 1>::type _rcreater;
		typedef typename _rcreater::result_type type;
	};

	template<template<typename U> class VecFuncTemplate, typename Slicing, class Arr1, class Arr2=nil_type>
	struct array_slice_stats_resultT
	{
		typedef typename array_view_traits<Arr1>::value_type arg1_value_type;
		typedef VecFuncTemplate<arg1_value_type> vecfunc_type;
		typedef typename array_slice_stats_result<vecfunc_type, Slicing, Arr1, Arr2>::type type;
	};

	template<typename VStatFunc, class Arr1>
	inline typename std::enable_if<is_array_view<Arr1>::value,
	typename array_stats_result<VStatFunc, Arr1>::type>::type
	evaluate_arr_stats(VStatFunc f, const Arr1& a1)
	{
		scoped_aview_read_proxy<Arr1> a1p(a1);
		return f(get_num_elems(a1), a1p.pbase());
	}

	template<typename VStatFunc, class Arr1, class Arr2>
	inline typename std::enable_if<is_compatible_aviews<Arr1, Arr2>::value,
	typename array_stats_result<VStatFunc, Arr1, Arr2>::type>::type
	evaluate_arr_stats(VStatFunc f, const Arr1& a1, const Arr2& a2)
	{
		check_arg(get_array_shape(a1) == get_array_shape(a2),
				"The shapes of operand arrays are inconsistent.");

		scoped_aview_read_proxy<Arr1> a1p(a1);
		scoped_aview_read_proxy<Arr2> a2p(a2);
		return f(get_num_elems(a1), a1p.pbase(), a2p.pbase());
	}

	template<typename VStatFunc, typename Slicing, class Arr1>
	inline typename std::enable_if<is_array_view_ndim<Arr1, 2>::value,
	typename array_slice_stats_result<VStatFunc, Slicing, Arr1>::type>::type
	evaluate_arr_slice_stats(VStatFunc f, const Arr1& a1, Slicing)
	{
		typedef typename array_slice_stats_result<VStatFunc, Slicing, Arr1>::type result_t;

		auto shape = get_array_shape(a1);
		index_t ns = Slicing::get_num_slices(shape[0], shape[1]);
		size_t slen = static_cast<size_t>(Slicing::get_slice_length(shape[0], shape[1]));

		typedef typename array_view_traits<Arr1>::value_type a1_value_t;
		typedef typename array_slice_stats_result<VStatFunc, Slicing, Arr1>::result_value_type rvalue_t;

		typedef typename array_creater<Arr1>::template remap<rvalue_t, 1>::type _rcreater;
		result_t r = _rcreater::create(arr_shape(ns));
		rvalue_t *pd = ptr_base(r);

		if (Slicing::template is_slice_major<Arr1>::value)
		{
			scoped_aview_read_proxy<Arr1> a1p(a1);
			const a1_value_t *ps1 = a1p.pbase();

			for (index_t i = 0; i < ns; ++i, ps1 += slen)
			{
				pd[i] = f(slen, ps1);
			}
		}
		else
		{
			auto a1t = transpose(a1);
			const a1_value_t *ps1 = ptr_base(a1t);

			for (index_t i = 0; i < ns; ++i, ps1 += slen)
			{
				pd[i] = f(slen, ps1);
			}
		}

		return r;
	}


	template<typename VStatFunc, typename Slicing, class Arr1, class Arr2>
	inline typename std::enable_if<is_compatible_aviews_ndim<Arr1, Arr2, 2>::value,
	typename array_slice_stats_result<VStatFunc, Slicing, Arr1, Arr2>::type>::type
	evaluate_arr_slice_stats(VStatFunc f, const Arr1& a1, const Arr2& a2, Slicing)
	{
		check_arg(get_array_shape(a1) == get_array_shape(a2),
				"The shapes of operand arrays are inconsistent.");

		typedef typename array_slice_stats_result<VStatFunc, Slicing, Arr1, Arr2>::type result_t;

		auto shape = get_array_shape(a1);
		index_t ns = Slicing::get_num_slices(shape[0], shape[1]);
		size_t slen = static_cast<size_t>(Slicing::get_slice_length(shape[0], shape[1]));

		typedef typename array_view_traits<Arr1>::value_type a1_value_t;
		typedef typename array_view_traits<Arr2>::value_type a2_value_t;
		typedef typename array_slice_stats_result<VStatFunc, Slicing, Arr1, Arr2>::result_value_type rvalue_t;

		typedef typename array_creater<Arr1>::template remap<rvalue_t, 1>::type _rcreater;
		result_t r = _rcreater::create(arr_shape(ns));
		rvalue_t *pd = ptr_base(r);

		if (Slicing::template is_slice_major<Arr1>::value)
		{
			scoped_aview_read_proxy<Arr1> a1p(a1);
			scoped_aview_read_proxy<Arr2> a2p(a2);

			const a1_value_t *ps1 = a1p.pbase();
			const a2_value_t *ps2 = a2p.pbase();

			for (index_t i = 0; i < ns; ++i, ps1 += slen, ps2 += slen)
			{
				pd[i] = f(slen, ps1, ps2);
			}
		}
		else
		{
			auto a1t = transpose(a1);
			auto a2t = transpose(a2);

			const a1_value_t *ps1 = ptr_base(a1t);
			const a2_value_t *ps2 = ptr_base(a2t);

			for (index_t i = 0; i < ns; ++i, ps1 += slen, ps2 += slen)
			{
				pd[i] = f(slen, ps1, ps2);
			}
		}

		return r;
	}


}

#endif 
