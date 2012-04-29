/**
 * @file matrix_assign.h
 *
 * The facility to support matrix assignment
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_ASSIGN_H_
#define BCSLIB_MATRIX_ASSIGN_H_

#include <bcslib/matrix/matrix_base.h>
#include <bcslib/matrix/matrix_xpr.h>

namespace bcs
{
	template<typename T, class SMat, class DMat>
	BCS_ENSURE_INLINE
	inline void assign_to(const IMatrixView<SMat, T>& src, IRegularMatrix<DMat, T>& dst)
	{
		copy(src.derived(), dst.derived());
	}

	template<typename T, class SMat, class DMat>
	BCS_ENSURE_INLINE
	inline void assign_to(const IMatrixView<SMat, T>& src, IDenseMatrix<DMat, T>& dst)
	{
		dst.resize(src.nrows(), src.ncolumns());
		copy(src.derived(), dst.derived());
	}

	template<typename T, class Expr, class DMat>
	BCS_ENSURE_INLINE
	inline void assign_to(const IMatrixXpr<Expr, T>& expr, IRegularMatrix<DMat, T>& dst)
	{
		check_arg( is_same_size(expr, dst),
				"The sizes of expression and destination are inconsistent." );

		evaluate_to(expr.derived(), dst.derived());
	}

	template<typename T, class Expr, class DMat>
	BCS_ENSURE_INLINE
	inline void assign_to(const IMatrixXpr<Expr, T>& expr, IDenseMatrix<DMat, T>& dst)
	{
		dst.resize(expr.nrows(), expr.ncolumns());
		evaluate_to(expr.derived(), dst.derived());
	}

}

#endif /* MATRIX_ASSIGN_H_ */
