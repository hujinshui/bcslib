/**
 * @file matrix_par_reduc.h
 *
 * Partial reduction on matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_PAR_REDUC_H_
#define BCSLIB_MATRIX_PAR_REDUC_H_

#include <bcslib/matrix/matrix_xpr.h>
#include <bcslib/matrix/slicewise_proxy.h>

#include <bcslib/matrix/bits/matrix_par_reduc_internal.h>
#include <bcslib/math/basic_reductors.h>


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

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename Reductor::result_type value_type;
		typedef index_t index_type;
	};


	template<typename Reductor, class Arg>
	struct unary_colwise_reduction_expr
	: public IMatrixXpr<unary_colwise_reduction_expr<Reductor, Arg>, typename Reductor::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 1>::value, "Reductor must be a unary reductor.");
		static_assert(has_matrix_interface<Arg, IMatrixXpr>::value, "Arg must be an matrix expression.");
#endif

		typedef Reductor reductor_type;
		typedef Arg arg_type;

		typedef typename Reductor::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Reductor reduc;
		const Arg& arg;

		BCS_ENSURE_INLINE
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

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename Reductor::result_type value_type;
		typedef index_t index_type;
	};


	template<typename Reductor, class LArg, class RArg>
	struct binary_colwise_reduction_expr
	: public IMatrixXpr<binary_colwise_reduction_expr<Reductor, LArg, RArg>, typename Reductor::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 2>::value, "Reductor must be a binary reductor.");
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

		BCS_ENSURE_INLINE
		binary_colwise_reduction_expr(Reductor f, const LArg& a, const RArg& b)
		: reduc(f), left_arg(a), right_arg(b)
		{
			check_arg( has_same_size(a, b),
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

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename Reductor::result_type value_type;
		typedef index_t index_type;
	};


	template<typename Reductor, class Arg>
	struct unary_rowwise_reduction_expr
	: public IMatrixXpr<unary_rowwise_reduction_expr<Reductor, Arg>, typename Reductor::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 1>::value, "Reductor must be a unary reductor.");
		static_assert(has_matrix_interface<Arg, IMatrixXpr>::value, "Arg must be an matrix expression.");
#endif

		typedef Reductor reductor_type;
		typedef Arg arg_type;

		typedef typename Reductor::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Reductor reduc;
		const Arg& arg;

		BCS_ENSURE_INLINE
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

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename Reductor::result_type value_type;
		typedef index_t index_type;
	};


	template<typename Reductor, class LArg, class RArg>
	struct binary_rowwise_reduction_expr
	: public IMatrixXpr<binary_rowwise_reduction_expr<Reductor, LArg, RArg>, typename Reductor::result_type>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 2>::value, "Reductor must be a binary reductor.");
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

		BCS_ENSURE_INLINE
		binary_rowwise_reduction_expr(Reductor f, const LArg& a, const RArg& b)
		: reduc(f), left_arg(a), right_arg(b)
		{
			check_arg( has_same_size(a, b),
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
		check_arg( has_same_size(larg, rarg), "The sizes of two operands are inconsistent.");
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
		check_arg( has_same_size(larg, rarg), "The sizes of two operands are inconsistent.");
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
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, value_type>& dst)
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
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, value_type>& dst)
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
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, value_type>& dst)
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
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, value_type>& dst)
		{
			detail::binary_rowwise_reduction_evaluator<Reductor, LArg, RArg, DMat>::run(
					expr.reduc, expr.left_arg, expr.right_arg, dst.derived());
		}
	};




	/********************************************
	 *
	 *  Specific partial reduction functions
	 *
	 ********************************************/

	// sum

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<sum_reductor<T>, Mat>
	sum(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(sum_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<sum_reductor<T>, Mat>
	sum(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(sum_reductor<T>(), proxy.ref());
	}


	// mean

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<mean_reductor<T>, Mat>
	mean(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(mean_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<mean_reductor<T>, Mat>
	mean(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(mean_reductor<T>(), proxy.ref());
	}


	// min

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<min_reductor<T>, Mat>
	min_val(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(min_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<min_reductor<T>, Mat>
	min_val(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(min_reductor<T>(), proxy.ref());
	}


	// max

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<max_reductor<T>, Mat>
	max_val(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(max_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<max_reductor<T>, Mat>
	max_val(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(max_reductor<T>(), proxy.ref());
	}

	// L1norm

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<L1norm_reductor<T>, Mat>
	L1norm(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(L1norm_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<L1norm_reductor<T>, Mat>
	L1norm(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(L1norm_reductor<T>(), proxy.ref());
	}


	// sqL2norm

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<sqL2norm_reductor<T>, Mat>
	sqL2norm(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(sqL2norm_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<sqL2norm_reductor<T>, Mat>
	sqL2norm(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(sqL2norm_reductor<T>(), proxy.ref());
	}

	// L2norm

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<L2norm_reductor<T>, Mat>
	L2norm(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(L2norm_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<L2norm_reductor<T>, Mat>
	L2norm(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(L2norm_reductor<T>(), proxy.ref());
	}


	// Linfnorm

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<Linfnorm_reductor<T>, Mat>
	Linfnorm(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(Linfnorm_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<Linfnorm_reductor<T>, Mat>
	Linfnorm(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(Linfnorm_reductor<T>(), proxy.ref());
	}



	// dot

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<dot_reductor<T>, LMat, RMat>
	dot(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(dot_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<dot_reductor<T>, LMat, RMat>
	dot(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(dot_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	// L1norm_diff

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<L1diffnorm_reductor<T>, LMat, RMat>
	L1norm_diff(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(L1diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<L1diffnorm_reductor<T>, LMat, RMat>
	L1norm_diff(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(L1diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	// sqL2norm_diff

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<sqL2diffnorm_reductor<T>, LMat, RMat>
	sqL2norm_diff(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(sqL2diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<sqL2diffnorm_reductor<T>, LMat, RMat>
	sqL2norm_diff(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(sqL2diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	// L2norm_diff

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<L2diffnorm_reductor<T>, LMat, RMat>
	L2norm_diff(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(L2diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<L2diffnorm_reductor<T>, LMat, RMat>
	L2norm_diff(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(L2diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}


	// Linfnorm_diff

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<Linfdiffnorm_reductor<T>, LMat, RMat>
	Linfnorm_diff(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(Linfdiffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<Linfdiffnorm_reductor<T>, LMat, RMat>
	Linfnorm_diff(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(Linfdiffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

}


#endif /* MATRIX_PAR_REDUC_H_ */


