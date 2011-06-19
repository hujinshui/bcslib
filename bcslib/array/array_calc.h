/**
 * @file array_calc.h
 *
 * Vectorized calculation on arrays
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_CALC_H
#define BCSLIB_ARRAY_CALC_H

#include <bcslib/base/arg_check.h>
#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <bcslib/veccomp/veccalc.h>


namespace bcs
{

	/******************************************************
	 *
	 *  Generic functions
	 *
	 ******************************************************/

	template<class Arr, class VecFunc>
	inline typename std::enable_if<
		is_array_view_ndim<Arr>::value,
	array1d<typename VecFunc::result_type> >::type
	array_calc(const Arr& a, VecFunc f)
	{
		static const dim_num_t num_dims = array_view_traits<Arr>::num_dims;
		typedef typename array_view_traits<Arr>::layout_order layout_order;
		typedef typename VecFunc::result_type result_t;

		typedef array_maker<result_t, num_dims, layout_order> maker_t;
		auto r = maker_t::create(get_shape(a));

		if (is_dense_view(a))
		{
			f(n, ptr_base(a), ptr_base(r));
		}
		else
		{
			auto ac = clone_array(a);
			f(n, ptr_base(ac), ptr_base(r));
		}

		return r;
	}


	template<class Arr1, class Arr2, class VecFunc>
	inline typename std::enable_if<
		is_array_view_ndim<Arr1>::value &&
		is_array_view_ndim<Arr2>::value,
	array1d<typename VecFunc::result_type> >::type
	array_calc(const Arr1& a, const Arr2& b, VecFunc f)
	{
		static const dim_num_t num_dims = array_view_traits<Arr1>::num_dims;
		typedef typename array_view_traits<Arr1>::layout_order layout_order;

		static const dim_num_t num_dims2 = array_view_traits<Arr2>::num_dims;
		typedef typename array_view_traits<Arr2>::layout_order layout_order2;

		typedef typename VecFunc::result_type result_t;

		static_assert(num_dims == num_dims2, "The number of dimensions of input arrays should be the same.");
		static_assert(is_same<layout_order, layout_order2>::value, "The layout orders of input arrays should be the same.");

		check_arg(get_shape(a) == get_shape(b), "The sizes of input arrays mismatch.");

		typedef array_maker<result_t, num_dims, layout_order> maker_t;

		auto r = maker_t::create(get_shape(a));

		if (is_dense_view(a))
		{
			if (is_dense_view(b))
			{
				f(n, ptr_base(a), ptr_base(b), ptr_base(r));
			}
			else
			{
				auto bc = clone_array(b);
				f(n, ptr_base(a), ptr_base(bc), ptr_base(r));
			}
		}
		else
		{
			auto ac = clone_array(a);
			if (is_dense_view(b))
			{
				f(n, ptr_base(ac), ptr_base(b), ptr_base(r));
			}
			else
			{
				array1d<result_t> bc(n, begin(b));
				f(n, ptr_base(ac), ptr_base(bc), ptr_base(r));
			}
		}

		return r;
	}


	template<class Arr, class VecFunc>
	inline typename std::enable_if<
		is_array_view_ndim<Arr, 1>::value,
	void>::type
	array_calc_inplace(Arr& a, VecFunc f)
	{
		typedef typename VecFunc::result_type result_t;

		size_t n = get_num_elems(a);

		if (is_dense_view(a))
		{
			f(n, ptr_base(a));
		}
		else
		{
			array1d<result_t> ac(n, begin(a));
			f(n, ptr_base(ac));
			import_from(a, ptr_base(ac));
		}
	}


	template<class Arr, class Arr2, class VecFunc>
	inline typename std::enable_if<
		is_array_view_ndim<Arr, 1>::value &&
		is_array_view_ndim<Arr2, 1>::value,
	void>::type
	array_calc_inplace(Arr& a, const Arr2& b, VecFunc f)
	{
		typedef typename VecFunc::result_type result_t;

		size_t n = get_num_elems(a);

		if (is_dense_view(a))
		{
			f(n, ptr_base(a));
		}
		else
		{
			array1d<result_t> ac(n, begin(a));
			f(n, ptr_base(ac));
			import_from(a, ptr_base(ac));
		}
	}




}

#endif 
