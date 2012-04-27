/**
 * @file matrix_xpr.h
 *
 * The basis for matrix expression evaluation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_XPR_H_
#define BCSLIB_MATRIX_XPR_H_

#include <bcslib/matrix/matrix_base.h>

namespace bcs
{

	// generic expression classes

	template<typename Fun, class Mat>
	struct ewise_unary_matrix_expr
	: public IMatrixXpr<ewise_unary_matrix_expr<Fun, Mat>, typename Fun::result_type>
	{
		typedef typename Fun::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Fun fun;
		const Mat& arg;

		ewise_unary_matrix_expr(Fun f, const Mat& a)
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

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return arg.is_empty();
		}
	};


	template<typename Fun, class LMat, class RMat>
	struct ewise_binary_matrix_expr
	: public IMatrixXpr<ewise_binary_matrix_expr<Fun, LMat, RMat>, typename Fun::result_type>
	{
		typedef typename Fun::result_type result_type;
		BCS_MAT_TRAITS_CDEFS(result_type)

		Fun fun;
		const LMat& arg1;
		const RMat& arg2;

		ewise_binary_matrix_expr(Fun f, const LMat& a1, const RMat& a2)
		: fun(f), arg1(a1), arg2(a2)
		{
			check_arg( is_same_size(a1, a2),
					"The size of two operand matrices are inconsistent." );
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return arg1.nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return arg1.size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return arg1.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return arg1.ncolumns();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return arg1.is_empty();
		}
	};


	template<typename Fun, typename T, class Mat>
	BCS_ENSURE_INLINE
	ewise_unary_matrix_expr<Fun, Mat>
	make_ewise_matrix_expr(const Fun& fun, const IMatrixXpr<Mat, T>& arg)
	{
		return ewise_unary_matrix_expr<Fun, Mat>(fun, arg);
	}


	template<typename Fun, typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	ewise_binary_matrix_expr<Fun, LMat, RMat>
	make_ewise_matrix_expr(const Fun& fun, const IMatrixXpr<LMat, T>& lhs, const IMatrixXpr<RMat, T>& rhs)
	{
		return ewise_binary_matrix_expr<Fun, LMat, RMat>(fun, lhs, rhs);
	}


	// assignment

	template<typename T, class Expr, class Mat>
	BCS_ENSURE_INLINE
	void assign_to(const IMatrixXpr<Expr, T>& expr, IDenseMatrix<Mat, T>& dst)
	{
		dst.resize(expr.nrows(), expr.ncolumns());
		evaluate_to(expr.derived(), dst.derived());
	}

	// operator tags

	struct plus_t { };
	struct minus_t { };
	struct times_t { };
	struct divides_t { };

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	ewise_binary_matrix_expr<plus_t, LMat, RMat>
	operator + (const IDenseMatrix<LMat, T>& A, const IMatrixXpr<RMat, T>& B)
	{
		return make_ewise_matrix_expr(plus_t(), A.derived(), B.derived());
	}


}

#endif
