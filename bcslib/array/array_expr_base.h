/**
 * @file array_expr_base.h
 *
 * The Basic facility for array expression evaluation
 * 
 * @author Dahua Lin
 */

#ifndef ARRAY_EXPR_BASE_H_
#define ARRAY_EXPR_BASE_H_

#include <bcslib/array/array_base.h>
#include <bcslib/array/generic_array_functions.h>
#include <bcslib/base/sexpression.h>

namespace bcs
{

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


	/******************************************************
	 *
	 *  Array Evaluation
	 *
	 ******************************************************/


	template<typename VecFunc, class Arr1>
	class unary_array_operator
	{
	public:
		static_assert(is_array_view<Arr1>::value, "Operands must be of array view types.");

		typedef typename array_view_traits<Arr1>::value_type arr1_value_type;

		typedef typename array_creater<Arr1>::template remap<typename VecFunc::result_value_type>::type _rcreater;
		typedef typename _rcreater::result_type result_type;

		unary_array_operator(VecFunc f) : m_vecfunc(f)
		{
		}

		result_type operator() (const Arr1& a1) const
		{
			return evaluate(m_vecfunc, a1);
		}

	public:
		static result_type evaluate(VecFunc vfunc, const Arr1& a1)
		{
			size_t n = get_num_elems(a1);

			scoped_aview_read_proxy<Arr1> ap1(a1);
			result_type r = _rcreater::create(get_array_shape(a1));

			vfunc(n, ap1.pbase(), ptr_base(r));

			return r;
		}

	private:
		VecFunc m_vecfunc;

	}; // end class unary_array_operator


	template<typename VecFunc, class Arr1, class Arr2>
	class binary_array_operator
	{
	public:
		static_assert(is_array_view<Arr1>::value && is_array_view<Arr2>::value,
				"Operands must be of array view types.");

		static_assert(array_view_traits<Arr1>::num_dims == array_view_traits<Arr2>::num_dims,
				"Operand arrays are required to have the same number of dimensions.");

		static_assert(std::is_same<
				typename array_view_traits<Arr1>::layout_order,
				typename array_view_traits<Arr2>::layout_order>::value,
				"Operand arrays are required to have the same layout order.");

		typedef typename array_view_traits<Arr1>::value_type arr1_value_type;
		typedef typename array_view_traits<Arr2>::value_type arr2_value_type;

		typedef typename array_creater<Arr1>::template remap<typename VecFunc::result_value_type>::type _rcreater;
		typedef typename _rcreater::result_type result_type;

		binary_array_operator(VecFunc f) : m_vecfunc(f)
		{
		}

		result_type operator() (const Arr1& a1, const Arr2& a2) const
		{
			return evaluate(m_vecfunc, a1, a2);
		}

	public:
		static result_type evaluate(VecFunc vfunc, const Arr1& a1, const Arr2& a2)
		{
			check_arg(get_array_shape(a1) == get_array_shape(a2), "The shapes of operand arrays are inconsistent.");

			size_t n = get_num_elems(a1);

			scoped_aview_read_proxy<Arr1> ap1(a1);
			scoped_aview_read_proxy<Arr2> ap2(a2);

			result_type r = _rcreater::create(get_array_shape(a1));

			vfunc(n, ap1.pbase(), ap2.pbase(), ptr_base(r));

			return r;
		}

	private:
		VecFunc m_vecfunc;

	}; // end class binary_array_operator


	template<typename InplaceVecFunc, class Arr>
	class array_inplace_operator
	{
	public:
		static_assert(is_array_view<Arr>::value, "Operands must be of array view types.");

		typedef typename array_view_traits<Arr>::value_type arr_value_type;

		array_inplace_operator(InplaceVecFunc f) : m_vecfunc(f)
		{
		}

		void operator() (Arr& a) const
		{
			return evaluate(m_vecfunc, a);
		}

	public:
		static void evaluate(InplaceVecFunc vfunc, Arr& a)
		{
			size_t n = get_num_elems(a);

			scoped_aview_write_proxy<Arr> ap(a, true);
			vfunc(n, ap.pbase());
			ap.commit();
		}

	private:
		InplaceVecFunc m_vecfunc;

	}; // end class array_inplace_operator


	template<typename InplaceVecFunc, class Arr, class RArr1>
	class array_inplace_operator_R1
	{
	public:
		static_assert(is_array_view<Arr>::value && is_array_view<RArr1>::value,
				"Operands must be of array view types.");

		static_assert(array_view_traits<Arr>::num_dims == array_view_traits<RArr1>::num_dims,
				"Operand arrays are required to have the same number of dimensions.");

		static_assert(std::is_same<
				typename array_view_traits<Arr>::layout_order,
				typename array_view_traits<RArr1>::layout_order>::value,
				"Operand arrays are required to have the same layout order.");

		typedef typename array_view_traits<Arr>::value_type arr_value_type;
		typedef typename array_view_traits<RArr1>::value_type rarr1_value_type;

		array_inplace_operator_R1(InplaceVecFunc f) : m_vecfunc(f)
		{
		}

		void operator() (Arr& a, const RArr1& r1) const
		{
			return evaluate(m_vecfunc, a, r1);
		}

	public:
		static void evaluate(InplaceVecFunc vfunc, Arr& a, const RArr1& r1)
		{
			size_t n = get_num_elems(a);

			scoped_aview_write_proxy<Arr> ap(a, true);
			scoped_aview_read_proxy<RArr1> r1p(r1);
			vfunc(n, ap.pbase(), r1p.pbase());
			ap.commit();
		}

	private:
		InplaceVecFunc m_vecfunc;

	}; // end class array_inplace_operator_R1



	template<typename VecFunc, class Arr>
	class unary_array_stats_evaluator
	{
	public:
		static_assert(is_array_view<Arr>::value, "Operands must be of array view types.");

		typedef typename array_view_traits<Arr>::value_type arr_value_type;

		typedef typename VecFunc::result_type result_type;

	public:
		unary_array_stats_evaluator(VecFunc vfunc) : m_vecfunc(vfunc)
		{
		}

		result_type operator() (const Arr& a) const
		{
			return evaluate(m_vecfunc, a);
		}

	public:
		static result_type evaluate(VecFunc vfunc, const Arr& a)
		{
			size_t n = get_num_elems(a);

			scoped_aview_read_proxy<Arr> ap(a);
			return vfunc(n, ap.pbase());
		}


	private:
		VecFunc m_vecfunc;

	}; // end class unary_array_stats_evaluator


	template<typename VecFunc, class Arr, class Slicing>
	class unary_array_slice_stats_evaluator
	{
	public:
		static_assert(is_array_view_ndim<Arr, 2>::value, "Operands must be of array view types.");

		typedef typename array_view_traits<Arr>::value_type arr_value_type;

		typedef typename VecFunc::result_type result_value_type;

		typedef typename array_creater<Arr>::template remap<result_value_type, 1>::type _rcreater;
		typedef typename _rcreater::result_type result_type;

		typedef Slicing slicing;

	public:
		unary_array_slice_stats_evaluator(VecFunc vfunc) : m_vecfunc(vfunc)
		{
		}

		result_type operator() (const Arr& a) const
		{
			return evaluate(m_vecfunc, a);
		}

	public:
		static result_type evaluate(VecFunc vfunc, const Arr& a)
		{
			auto shape = get_array_shape(a);
			index_t ns = slicing::get_num_slices(shape[0], shape[1]);
			index_t slen = slicing::get_slice_length(shape[0], shape[1]);

			if (slicing::template is_slice_major<Arr>::value)
			{
				scoped_aview_read_proxy<Arr> ap(a);
				return do_eval(vfunc, ns, slen, ap.pbase());
			}
			else
			{
				auto t = transpose(a);
				return do_eval(vfunc, ns, slen, ptr_base(t));
			}
		}

	private:
		static result_type do_eval(VecFunc vfunc, index_t ns, index_t slen, const arr_value_type *x)
		{
			result_type r = _rcreater::create(arr_shape(ns));
			size_t sl = static_cast<size_t>(slen);

			result_value_type *pd = ptr_base(r);
			for (index_t i = 0; i < ns; ++i, x += slen)
			{
				pd[i] = vfunc(sl, x);
			}

			return r;
		}

	private:
		VecFunc m_vecfunc;

	}; // end class unary_array_perrow_stats_evaluator



	template<typename VecFunc, class Arr1, class Arr2>
	class binary_array_stats_evaluator
	{
	public:
		static_assert(is_array_view<Arr1>::value && is_array_view<Arr2>::value,
				"Operands must be of array view types.");

		static_assert(array_view_traits<Arr1>::num_dims == array_view_traits<Arr2>::num_dims,
				"Operand arrays are required to have the same number of dimensions.");

		static_assert(std::is_same<
				typename array_view_traits<Arr1>::layout_order,
				typename array_view_traits<Arr2>::layout_order>::value,
				"Operand arrays are required to have the same layout order.");

		typedef typename array_view_traits<Arr1>::value_type arr1_value_type;
		typedef typename array_view_traits<Arr2>::value_type arr2_value_type;

		typedef typename VecFunc::result_type result_type;

	public:
		binary_array_stats_evaluator(VecFunc vfunc) : m_vecfunc(vfunc)
		{
		}

		result_type operator() (const Arr1& a, const Arr2& b) const
		{
			return evaluate(m_vecfunc, a, b);
		}

	public:
		static result_type evaluate(VecFunc vfunc, const Arr1& a, const Arr2& b)
		{
			check_arg(get_array_shape(a) == get_array_shape(b), "The shapes of operand arrays are inconsistent.");
			size_t n = get_num_elems(a);

			scoped_aview_read_proxy<Arr1> ap(a);
			scoped_aview_read_proxy<Arr2> bp(b);

			return vfunc(n, ap.pbase(), bp.pbase());
		}

	private:
		VecFunc m_vecfunc;

	}; // end class binary_array_stats_evaluator



	template<typename VecFunc, class Arr1, class Arr2, class Slicing>
	class binary_array_slice_stats_evaluator
	{
	public:
		static_assert(is_array_view_ndim<Arr1, 2>::value && is_array_view_ndim<Arr2, 2>::value,
				"Operands must be of 2D array view types.");

		static_assert(array_view_traits<Arr1>::num_dims == array_view_traits<Arr2>::num_dims,
				"Operand arrays are required to have the same number of dimensions.");

		static_assert(std::is_same<
				typename array_view_traits<Arr1>::layout_order,
				typename array_view_traits<Arr2>::layout_order>::value,
				"Operand arrays are required to have the same layout order.");

		typedef typename array_view_traits<Arr1>::value_type arr1_value_type;
		typedef typename array_view_traits<Arr2>::value_type arr2_value_type;

		typedef typename VecFunc::result_type result_value_type;

		typedef typename array_creater<Arr1>::template remap<result_value_type, 1>::type _rcreater;
		typedef typename _rcreater::result_type result_type;

		typedef Slicing slicing;

	public:
		binary_array_slice_stats_evaluator(VecFunc vfunc) : m_vecfunc(vfunc)
		{
		}

		result_type operator() (const Arr1& a, const Arr2& b) const
		{
			return evaluate(m_vecfunc, a, b);
		}

	public:
		static result_type evaluate(VecFunc vfunc, const Arr1& a, const Arr2& b)
		{
			check_arg(get_array_shape(a) == get_array_shape(b), "The shapes of operand arrays are inconsistent.");
			auto shape = get_array_shape(a);
			index_t ns = slicing::get_num_slices(shape[0], shape[1]);
			index_t slen = slicing::get_slice_length(shape[0], shape[1]);

			if (slicing::template is_slice_major<Arr1>::value)
			{
				scoped_aview_read_proxy<Arr1> ap(a);
				scoped_aview_read_proxy<Arr2> bp(b);

				return do_eval(vfunc, ns, slen, ap.pbase(), bp.pbase());
			}
			else
			{
				auto ta = transpose(a);
				auto tb = transpose(b);

				return do_eval(vfunc, ns, slen, ptr_base(ta), ptr_base(tb));
			}
		}

	private:
		static result_type do_eval(VecFunc vfunc, index_t ns, index_t slen, const arr1_value_type *x, const arr2_value_type *y)
		{
			result_type r = _rcreater::create(arr_shape(ns));
			size_t sl = static_cast<size_t>(slen);

			result_value_type *pd = ptr_base(r);
			for (index_t i = 0; i < ns; ++i, x += slen, y += slen)
			{
				pd[i] = vfunc(sl, x, y);
			}

			return r;
		}

	private:
		VecFunc m_vecfunc;

	}; // end class binary_array_perrow_stats_evaluator




	/******************************************************
	 *
	 *  Meta-programming helper
	 *
	 ******************************************************/

	template<typename Arr1, template<typename U> class VecFuncTemplate>
	struct _arr_uniop
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef unary_array_operator<vecfunc_type, Arr1> evaluator_type;
		typedef typename evaluator_type::result_type result_type;

		typedef result_type type;  // serve as host

		static result_type default_evaluate(const Arr1& a1)
		{
			return evaluator_type::evaluate(vecfunc_type(), a1);
		}

		static result_type evaluate_with_scalar(const Arr1& a1, const value_type& v)
		{
			return evaluator_type::evaluate(vecfunc_type(v), a1);
		}
	};


	template<typename Arr1, typename Arr2, template<typename U> class VecFuncTemplate>
	struct _arr_binop
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef binary_array_operator<vecfunc_type, Arr1, Arr2> evaluator_type;
		typedef typename evaluator_type::result_type result_type;

		typedef result_type type;  // serve as host

		static result_type default_evaluate(const Arr1& a1, const Arr2& a2)
		{
			return evaluator_type::evaluate(vecfunc_type(), a1, a2);
		}
	};


	template<typename Arr, template<typename U> class VecFuncTemplate>
	struct _arr_ipop
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef array_inplace_operator<vecfunc_type, Arr> evaluator_type;
		typedef void result_type;

		typedef result_type type;  // serve as host

		static void default_evaluate(Arr& a)
		{
			evaluator_type::evaluate(vecfunc_type(), a);
		}

		static void evaluate_with_scalar(Arr& a, const value_type& v)
		{
			evaluator_type::evaluate(vecfunc_type(v), a);
		}
	};


	template<typename Arr, typename RArr1, template<typename U> class VecFuncTemplate>
	struct _arr_ipop_R1
	{
		typedef typename array_view_traits<RArr1>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef array_inplace_operator_R1<vecfunc_type, Arr, RArr1> evaluator_type;
		typedef void result_type;

		typedef result_type type;  // serve as host

		static void default_evaluate(Arr& a, const RArr1& r1)
		{
			evaluator_type::evaluate(vecfunc_type(), a, r1);
		}
	};


	template<typename Arr, template<typename U> class VecFuncTemplate>
	struct _uniarr_stat
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef unary_array_stats_evaluator<vecfunc_type, Arr> evaluator_type;
		typedef typename evaluator_type::result_type result_type;

		typedef result_type type;

		static result_type default_evaluate(const Arr& a1)
		{
			return evaluator_type::evaluate(vecfunc_type(), a1);
		}

		static result_type evaluate_with_scalar(const Arr& a1, const value_type& v)
		{
			return evaluator_type::evaluate(vecfunc_type(v), a1);
		}
	};

	template<typename Arr, typename Slicing, template<typename U> class VecFuncTemplate>
	struct _uniarr_slice_stat
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef unary_array_slice_stats_evaluator<vecfunc_type, Arr, Slicing> evaluator_type;
		typedef typename evaluator_type::result_type result_type;

		typedef result_type type;

		static result_type default_evaluate(const Arr& a1)
		{
			return evaluator_type::evaluate(vecfunc_type(), a1);
		}

		static result_type evaluate_with_scalar(const Arr& a1, const value_type& v)
		{
			return evaluator_type::evaluate(vecfunc_type(v), a1);
		}
	};

	template<typename Arr1, typename Arr2, template<typename U> class VecFuncTemplate>
	struct _binarr_stat
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef binary_array_stats_evaluator<vecfunc_type, Arr1, Arr2> evaluator_type;
		typedef typename evaluator_type::result_type result_type;

		typedef result_type type;

		static result_type default_evaluate(const Arr1& a1, const Arr2& a2)
		{
			return evaluator_type::evaluate(vecfunc_type(), a1, a2);
		}
	};



}

#endif 
