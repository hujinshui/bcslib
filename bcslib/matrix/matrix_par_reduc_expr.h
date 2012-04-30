/**
 * @file matrix_reduction_expr.h
 *
 * Generic matrix reduction expressions
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_PAR_REDUC_EXPR_H_
#define BCSLIB_MATRIX_PAR_REDUC_EXPR_H_

#include <bcslib/matrix/matrix_xpr.h>
#include <bcslib/matrix/bits/matrix_par_reduc_internal.h>

namespace bcs
{

	// forward declaration

	template<typename Reductor, class Arg> struct unary_colwise_reduction_expr;
	template<typename Reductor, class LArg, class RArg> struct binary_colwise_reduction_expr;

	template<typename Reductor, class Arg> struct unary_rowwise_reduction_expr;
	template<typename Reductor, class LArg, class RArg> struct binary_rowwise_reduction_expr;

	/********************************************
	 *
	 *  Generic expression classes
	 *
	 ********************************************/

	template<typename Reductor, class Arg>
	struct matrix_traits<unary_colwise_reduction_expr<Reductor, Arg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = 1;
		static const int compile_time_num_cols = ct_cols<Arg>::value;

		static const bool is_linear_indexable = false;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef typename Reductor::result_type value_type;
		typedef index_t index_type;
	};


	template<typename Reductor, class Arg>
	struct unary_colwise_reduction_expr
	: public IMatrixXpr<unary_colwise_reduction_expr<Reductor, Arg>, typename Reductor::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_unary_reduction_functor<Reductor>::value, "Reductor must be a unary reduction functor.");
		static_assert(has_matrix_interface<Arg, IMatrixXpr>::value, "Arg must be an matrix expression.");
#endif

		typedef Reductor reductor_type;
		typedef Arg arg_type;

		typedef typename Reductor::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Reductor reduc;
		const Arg& arg;

		unary_colwise_reduction_expr(Reductor f, const Arg& a)
		: reduc(f), arg(a)
		{
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return arg.ncolumns();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return 1;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return arg.ncolumns();
		}
	};


	template<typename Reductor, class LArg, class RArg>
	struct matrix_traits<binary_colwise_reduction_expr<Reductor, LArg, RArg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = 1;
		static const int compile_time_num_cols = binary_ct_cols<LArg, RArg>::value;

		static const bool is_linear_indexable = false;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef typename Reductor::result_type value_type;
		typedef index_t index_type;
	};


	template<typename Reductor, class LArg, class RArg>
	struct binary_colwise_reduction_expr
	: public IMatrixXpr<binary_colwise_reduction_expr<Reductor, LArg, RArg>, typename Reductor::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_binary_reduction_functor<Reductor>::value, "Reductor must be a binary reduction functor.");
		static_assert(has_matrix_interface<LArg, IMatrixXpr>::value, "LArg must be an matrix expression.");
		static_assert(has_matrix_interface<RArg, IMatrixXpr>::value, "RArg must be an matrix expression.");
#endif

		typedef Reductor reductor_type;
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		typedef typename Reductor::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Reductor reduc;
		const LArg& left_arg;
		const RArg& right_arg;

		binary_colwise_reduction_expr(Reductor f, const LArg& a, const RArg& b)
		: reduc(f), left_arg(a), right_arg(b)
		{
			check_arg( is_same_size(a, b),
					"The size of two operand matrices are inconsistent." );
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return left_arg.ncolumns();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return 1;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return left_arg.ncolumns();
		}
	};


	template<typename Reductor, class Arg>
	struct matrix_traits<unary_rowwise_reduction_expr<Reductor, Arg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_rows<Arg>::value;
		static const int compile_time_num_cols = 1;

		static const bool is_linear_indexable = false;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef typename Reductor::result_type value_type;
		typedef index_t index_type;
	};


	template<typename Reductor, class Arg>
	struct unary_rowwise_reduction_expr
	: public IMatrixXpr<unary_rowwise_reduction_expr<Reductor, Arg>, typename Reductor::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_unary_reduction_functor<Reductor>::value, "Reductor must be a unary reduction functor.");
		static_assert(has_matrix_interface<Arg, IMatrixXpr>::value, "Arg must be an matrix expression.");
#endif

		typedef Reductor reductor_type;
		typedef Arg arg_type;

		typedef typename Reductor::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Reductor reduc;
		const Arg& arg;

		unary_rowwise_reduction_expr(Reductor f, const Arg& a)
		: reduc(f), arg(a)
		{
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return arg.nrows();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return arg.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return 1;
		}
	};


	template<typename Reductor, class LArg, class RArg>
	struct matrix_traits<binary_rowwise_reduction_expr<Reductor, LArg, RArg> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = binary_ct_rows<LArg, RArg>::value;
		static const int compile_time_num_cols = 1;

		static const bool is_linear_indexable = false;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef typename Reductor::result_type value_type;
		typedef index_t index_type;
	};


	template<typename Reductor, class LArg, class RArg>
	struct binary_rowwise_reduction_expr
	: public IMatrixXpr<binary_rowwise_reduction_expr<Reductor, LArg, RArg>, typename Reductor::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_binary_reduction_functor<Reductor>::value, "Reductor must be a binary reduction functor.");
		static_assert(has_matrix_interface<LArg, IMatrixXpr>::value, "LArg must be an matrix expression.");
		static_assert(has_matrix_interface<RArg, IMatrixXpr>::value, "RArg must be an matrix expression.");
#endif

		typedef Reductor reductor_type;
		typedef LArg left_arg_type;
		typedef RArg right_arg_type;

		typedef typename Reductor::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Reductor reduc;
		const LArg& left_arg;
		const RArg& right_arg;

		binary_rowwise_reduction_expr(Reductor f, const LArg& a, const RArg& b)
		: reduc(f), left_arg(a), right_arg(b)
		{
			check_arg( is_same_size(a, b),
					"The size of two operand matrices are inconsistent." );
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return left_arg.nrows();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return left_arg.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return 1;
		}
	};


	// convenient functions

	template<typename Reductor, typename T, class Arg>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<Reductor, Arg>
	colwise_reduce(const Reductor& fun, const IMatrixXpr<Arg, T>& arg)
	{
		return unary_colwise_reduction_expr<Reductor, Arg>(fun, arg.derived());
	}


	template<typename Reductor, typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<Reductor, LArg, RArg>
	colwise_reduce(const Reductor& fun, const IMatrixXpr<LArg, T>& larg, const IMatrixXpr<RArg, T>& rarg)
	{
		check_arg( is_same_size(larg, rarg), "The sizes of two operands are inconsistent.");
		return binary_colwise_reduction_expr<Reductor, LArg, RArg>(fun, larg.derived(), rarg.derived());
	}

	template<typename Reductor, typename T, class Arg>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<Reductor, Arg>
	rowwise_reduce(const Reductor& fun, const IMatrixXpr<Arg, T>& arg)
	{
		return unary_rowwise_reduction_expr<Reductor, Arg>(fun, arg.derived());
	}


	template<typename Reductor, typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<Reductor, LArg, RArg>
	rowwise_reduce(const Reductor& fun, const IMatrixXpr<LArg, T>& larg, const IMatrixXpr<RArg, T>& rarg)
	{
		check_arg( is_same_size(larg, rarg), "The sizes of two operands are inconsistent.");
		return binary_rowwise_reduction_expr<Reductor, LArg, RArg>(fun, larg.derived(), rarg.derived());
	}


	/********************************************
	 *
	 *  Evaluation
	 *
	 ********************************************/

	template<typename Reductor, class Arg>
	struct expr_evaluator<unary_colwise_reduction_expr<Reductor, Arg> >
	{
		typedef unary_colwise_reduction_expr<Reductor, Arg> expr_type;
		typedef typename matrix_traits<expr_type>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::unary_colwise_reduction_evaluator<Reductor, Arg, DMat>::run(
					expr.reduc, expr.arg, dst.derived());
		}
	};


	template<typename Reductor, class Arg>
	struct expr_evaluator<unary_rowwise_reduction_expr<Reductor, Arg> >
	{
		typedef unary_rowwise_reduction_expr<Reductor, Arg> expr_type;
		typedef typename matrix_traits<expr_type>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::unary_rowwise_reduction_evaluator<Reductor, Arg, DMat>::run(
					expr.reduc, expr.arg, dst.derived());
		}
	};


	template<typename Reductor, class LArg, class RArg>
	struct expr_evaluator<binary_colwise_reduction_expr<Reductor, LArg, RArg> >
	{
		typedef binary_colwise_reduction_expr<Reductor, LArg, RArg> expr_type;
		typedef typename matrix_traits<expr_type>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::binary_colwise_reduction_evaluator<Reductor, LArg, RArg, DMat>::run(
					expr.reduc, expr.left_arg, expr.right_arg, dst.derived());
		}
	};


	template<typename Reductor, class LArg, class RArg>
	struct expr_evaluator<binary_rowwise_reduction_expr<Reductor, LArg, RArg> >
	{
		typedef binary_rowwise_reduction_expr<Reductor, LArg, RArg> expr_type;
		typedef typename matrix_traits<expr_type>::value_type value_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			detail::binary_rowwise_reduction_evaluator<Reductor, LArg, RArg, DMat>::run(
					expr.reduc, expr.left_arg, expr.right_arg, dst.derived());
		}
	};
}

#endif
