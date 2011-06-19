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
#include <bcslib/base/sexpression.h>

namespace bcs
{

	/******************************************************
	 *
	 *  Array Evaluation
	 *
	 ******************************************************/

	template<typename VecFunc, class Arr1, class Arr2>
	class binary_array_array_operator
	{
	public:
		static_assert(array_view_traits<Arr1>::num_dims == array_view_traits<Arr2>::num_dims,
				"Operand arrays are required to have the same number of dimensions.");

		static_assert(std::is_same<
				typename array_view_traits<Arr1>::layout_order,
				typename array_view_traits<Arr2>::layout_order>::value,
				"Operand arrays are required to have the same layout order.");

		typedef typename array_creater<Arr1>::template remap<typename VecFunc::result_value_type> _rcreater;
		typedef typename _rcreater::result_type result_type;

		binary_array_array_operator(VecFunc f) : m_vecfunc(f)
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
					auto a2c = clone_array(a2);
					vfunc(n, ptr_base(a1), ptr_base(a2c), ptr_base(r));
				}
			}
			else
			{
				auto a1c = clone_array(a1);

				if (is_dense_view(a2))
				{
					vfunc(n, ptr_base(a1c), ptr_base(a2), ptr_base(r));
				}
				else
				{
					auto a2c = clone_array(a2);
					vfunc(n, ptr_base(a1c), ptr_base(a2c), ptr_base(r));
				}
			}

			return r;
		}

	private:
		VecFunc m_vecfunc;

	}; // end class binary_array_array_operator


	/******************************************************
	 *
	 *  Meta-programming helper
	 *
	 ******************************************************/

	template<typename Arr1, typename Arr2, template<typename U> class VecFuncTemplate>
	struct _arr_op_aa
	{
		typedef typename array_view_traits<Arr1>::value_type value_type;
		typedef VecFuncTemplate<value_type> vecfunc_type;
		typedef binary_array_array_operator<vecfunc_type, Arr1, Arr2> operator_type;
		typedef typename operator_type::result_type result_type;

		typedef result_type type;  // serve as host

		static result_type default_evaluate(const Arr1& a1, const Arr2& a2)
		{
			return operator_type::evaluate(vecfunc_type(), a1, a2);
		}
	};


}

#endif 
