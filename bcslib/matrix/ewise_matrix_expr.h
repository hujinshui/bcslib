/**
 * @file ewise_matrix_expr.h
 *
 * Element-wise matrix expressions
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_EWISE_MATRIX_EXPR_H_
#define BCSLIB_EWISE_MATRIX_EXPR_H_

#include <bcslib/matrix/matrix_xpr.h>


namespace bcs
{
	// forward declaration

	template<typename Fun, class Arg> struct unary_ewise_expr;
	template<typename Fun, class LArg, class RArg> struct binary_ewise_expr;


	/********************************************
	 *
	 *  Generic expression classes
	 *
	 ********************************************/

	template<typename T, class Arg>
	struct matrix_traits<unary_ewise_expr<T, Arg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_rows<Arg>::value;
		static const int compile_time_num_cols = ct_cols<Arg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef T value_type;
		typedef index_t index_type;
	};


	template<typename Fun, class Arg>
	struct unary_ewise_expr
	: public IMatrixXpr<unary_ewise_expr<Fun, Arg>, typename Fun::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_unary_ewise_functor<Fun>::value, "Fun must be a unary ewise-functor.");
		static_assert(is_mat_xpr<Arg>::value, "Arg must be an matrix expression.");
#endif

		typedef Fun function_type;
		typedef Arg arg_type;

		typedef typename Fun::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Fun fun;
		const Arg& arg;

		unary_ewise_expr(Fun f, const Arg& a)
		: fun(f), arg(a)
		{
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return arg.nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return arg.size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return arg.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return arg.ncolumns();
		}
	};


	template<typename T, class LArg, class RArg>
	struct matrix_traits<binary_ewise_expr<T, LArg, RArg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = binary_ct_rows<LArg, RArg>::value;
		static const int compile_time_num_cols = binary_ct_cols<LArg, RArg>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef T value_type;
		typedef index_t index_type;
	};


	template<typename Fun, class LArg, class RArg>
	struct binary_ewise_expr
	: public IMatrixXpr<binary_ewise_expr<Fun, LArg, RArg>, typename Fun::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_binary_ewise_functor<Fun>::value, "Fun must be a binary ewise-functor.");
		static_assert(is_mat_xpr<LArg>::value, "LArg must be an matrix expression.");
		static_assert(is_mat_xpr<RArg>::value, "RArg must be an matrix expression.");
#endif

		typedef Fun function_type;
		typedef LArg leftarg_type;
		typedef RArg rightarg_type;

		typedef typename Fun::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Fun fun;
		const LArg& left_arg;
		const RArg& right_arg;

		binary_ewise_expr(Fun f, const LArg& a1, const RArg& a2)
		: fun(f), left_arg(a1), right_arg(a2)
		{
			check_arg( is_same_size(a1, a2),
					"The size of two operand matrices are inconsistent." );
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return left_arg.nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return left_arg.size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return left_arg.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return left_arg.ncolumns();
		}
	};


	// convenient functions

	template<typename Fun, typename T, class Arg>
	BCS_ENSURE_INLINE
	unary_ewise_expr<Fun, Arg>
	map_ewise(const Fun& fun, const IMatrixXpr<Arg, T>& arg)
	{
		return unary_ewise_expr<Fun, Arg>(fun, arg.derived());
	}


	template<typename Fun, typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	binary_ewise_expr<Fun, LArg, RArg>
	map_ewise(const Fun& fun, const IMatrixXpr<LArg, T>& larg, const IMatrixXpr<RArg, T>& rarg)
	{
		return binary_ewise_expr<Fun, LArg, RArg>(fun, larg.derived(), rarg.derived());
	}


}

#endif
