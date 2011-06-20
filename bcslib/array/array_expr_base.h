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

		typedef typename array_creater<Arr1>::template remap<typename VecFunc::result_value_type> _rcreater;
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
			result_type r = _rcreater::create(get_array_shape(a1));
			size_t n = get_num_elems(a1);

			if (is_dense_view(a1))
			{
				vfunc(n, ptr_base(a1), ptr_base(r));
			}
			else
			{
				scoped_buffer<arr1_value_type> buf1(n);
				export_to(a1, buf1.pbase());

				vfunc(n, buf1.pbase(), ptr_base(r));
			}

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

		typedef typename array_creater<Arr1>::template remap<typename VecFunc::result_value_type> _rcreater;
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
			result_type r = _rcreater::create(get_array_shape(a1));
			size_t n = get_num_elems(a1);

			if (is_dense_view(a1))
			{
				if (is_dense_view(a2))
				{
					vfunc(n, ptr_base(a1), ptr_base(a2), ptr_base(r));
				}
				else
				{
					scoped_buffer<arr2_value_type> buf2(n);
					export_to(a2, buf2.pbase());

					vfunc(n, ptr_base(a1), buf2.pbase(), ptr_base(r));
				}
			}
			else
			{
				scoped_buffer<arr1_value_type> buf1(n);
				export_to(a1, buf1.pbase());

				if (is_dense_view(a2))
				{
					vfunc(n, buf1.pbase(), ptr_base(a2), ptr_base(r));
				}
				else
				{
					scoped_buffer<arr2_value_type> buf2(n);
					export_to(a2, buf2.pbase());

					vfunc(n, buf1.pbase(), buf2.pbase(), ptr_base(r));
				}
			}

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

			if (is_dense_view(a))
			{
				vfunc(n, ptr_base(a));
			}
			else
			{
				scoped_buffer<arr_value_type> buf(n);
				export_to(a, buf.pbase());
				vfunc(n, buf.pbase());
				import_from(a, buf.pbase());
			}
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

			if (is_dense_view(a))
			{
				if (is_dense_view(r1))
				{
					vfunc(n, ptr_base(a), ptr_base(r1));
				}
				else
				{
					scoped_buffer<rarr1_value_type> buf_r1(n);
					export_to(r1, buf_r1.pbase());
					vfunc(n, ptr_base(a), buf_r1.pbase());
				}
			}
			else
			{
				scoped_buffer<arr_value_type> buf(n);
				export_to(a, buf.pbase());

				if (is_dense_view(r1))
				{
					vfunc(n, buf.pbase(), ptr_base(r1));
				}
				else
				{
					scoped_buffer<rarr1_value_type> buf_r1(n);
					export_to(r1, buf_r1.pbase());
					vfunc(n, buf.pbase(), buf_r1.pbase());
				}
				import_from(a, buf.pbase());
			}
		}

	private:
		InplaceVecFunc m_vecfunc;

	}; // end // end class array_inplace_operator





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
		typedef unary_array_operator<vecfunc_type, Arr1> operator_type;
		typedef typename operator_type::result_type result_type;

		typedef result_type type;  // serve as host

		static result_type default_evaluate(const Arr1& a1)
		{
			return operator_type::evaluate(vecfunc_type(), a1);
		}

		static result_type evaluate_with_scalar(const Arr1& a1, const value_type& v)
		{
			return operator_type::evaluate(vecfunc_type(v), a1);
		}
	};


	template<typename Arr1, typename Arr2, template<typename U> class VecFuncTemplate>
	struct _arr_binop
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef binary_array_operator<vecfunc_type, Arr1, Arr2> operator_type;
		typedef typename operator_type::result_type result_type;

		typedef result_type type;  // serve as host

		static result_type default_evaluate(const Arr1& a1, const Arr2& a2)
		{
			return operator_type::evaluate(vecfunc_type(), a1, a2);
		}
	};


	template<typename Arr, template<typename U> class VecFuncTemplate>
	struct _arr_ipop
	{
		typedef typename array_view_traits<Arr>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef array_inplace_operator<vecfunc_type, Arr> operator_type;
		typedef void result_type;

		typedef result_type type;  // serve as host

		static void default_evaluate(Arr& a)
		{
			operator_type::evaluate(vecfunc_type(), a);
		}

		static void evaluate_with_scalar(Arr& a, const value_type& v)
		{
			operator_type::evaluate(vecfunc_type(v), a);
		}
	};


	template<typename Arr, typename RArr1, template<typename U> class VecFuncTemplate>
	struct _arr_ipop_R1
	{
		typedef typename array_view_traits<RArr1>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef array_inplace_operator_R1<vecfunc_type, Arr, RArr1> operator_type;
		typedef void result_type;

		typedef result_type type;  // serve as host

		static void default_evaluate(Arr& a, const RArr1& r1)
		{
			operator_type::evaluate(vecfunc_type(), a, r1);
		}
	};


}

#endif 
