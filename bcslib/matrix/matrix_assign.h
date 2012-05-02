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

#include <bcslib/matrix/matrix_xpr.h>

namespace bcs
{
	namespace detail
	{
		template<bool IsResizale> struct ensure_same_size_helper;

		template<> struct ensure_same_size_helper<true>
		{
			template<typename T, class Expr, class DMat>
			BCS_ENSURE_INLINE
			static void run(const IMatrixXpr<Expr, T>& expr, DMat& dst)
			{
				dst.resize(expr.nrows(), expr.ncolumns());
			}
		};

		template<> struct ensure_same_size_helper<false>
		{
			template<typename T, class Expr, class DMat>
			BCS_ENSURE_INLINE
			static void run(const IMatrixXpr<Expr, T>& expr, DMat& dst)
			{
				check_same_size(expr, dst,
						"The sizes of expression and destination are inconsistent.");
			}
		};
	}

	template<typename T, class Expr, class DMat>
	BCS_ENSURE_INLINE
	inline void ensure_same_size(const IMatrixXpr<Expr, T>& expr, IRegularMatrix<DMat, T>& dst)
	{
		typedef detail::ensure_same_size_helper<matrix_traits<DMat>::is_resizable> helper_t;
		helper_t::run(expr, dst.derived());
	}


	template<typename T, class Expr, class DMat>
	BCS_ENSURE_INLINE
	inline void assign_to(const IMatrixXpr<Expr, T>& expr, IRegularMatrix<DMat, T>& dst)
	{
		ensure_same_size(expr, dst);
		evaluate_to(expr.derived(), dst.derived());
	}


}

#endif /* MATRIX_ASSIGN_H_ */
