/**
 * @file matrix_arithmetic.h
 *
 * Arithmetic evaluation for matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_ARITHMETIC_H_
#define BCSLIB_MATRIX_ARITHMETIC_H_

#include <bcslib/matrix/ewise_matrix_expr.h>
#include <bcslib/engine/arithmetic_functors.h>

namespace bcs
{

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline binary_ewise_expr<binary_plus<T>, LArg, RArg>
	operator + (const IMatrixXpr<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return make_binary_ewise_expr(binary_plus<T>(), A, B);
	}


	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_sqr<T>, Arg>
	sqr(const IMatrixXpr<Arg, T>& A)
	{
		return make_unary_ewise_expr(unary_sqr<T>(), A);
	}

}

#endif /* MATRIX_ARITHMETIC_H_ */
