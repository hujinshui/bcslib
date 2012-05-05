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
	// plus

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline binary_ewise_expr<binary_plus<T>, LArg, RArg>
	operator + (const IMatrixXpr<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return map_ewise(binary_plus<T>(), A.derived(), B.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<plus_scalar<T>, LArg>
	operator + (const IMatrixXpr<LArg, T>& A, const T& b)
	{
		return map_ewise(plus_scalar<T>(b), A.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<plus_scalar<T>, LArg>
	operator + (const T& a, const IMatrixXpr<LArg, T>& B)
	{
		return map_ewise(plus_scalar<T>(a), B.derived());
	}

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline LArg& operator += (IDenseMatrix<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return A.derived() = A.derived() + B.derived();
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline LArg& operator += (IDenseMatrix<LArg, T>& A, const T& b)
	{
		return A.derived() = A.derived() + b;
	}

	// minus

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline binary_ewise_expr<binary_minus<T>, LArg, RArg>
	operator - (const IMatrixXpr<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return map_ewise(binary_minus<T>(), A.derived(), B.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<minus_scalar<T>, LArg>
	operator - (const IMatrixXpr<LArg, T>& A, const T& b)
	{
		return map_ewise(minus_scalar<T>(b), A.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<rminus_scalar<T>, LArg>
	operator - (const T& a, const IMatrixXpr<LArg, T>& B)
	{
		return map_ewise(rminus_scalar<T>(a), B.derived());
	}

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline LArg& operator -= (IDenseMatrix<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return A.derived() = A.derived() - B.derived();
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline LArg& operator -= (IDenseMatrix<LArg, T>& A, const T& b)
	{
		return A.derived() = A.derived() - b;
	}

	// times

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline binary_ewise_expr<binary_times<T>, LArg, RArg>
	operator * (const IMatrixXpr<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return map_ewise(binary_times<T>(), A.derived(), B.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<times_scalar<T>, LArg>
	operator * (const IMatrixXpr<LArg, T>& A, const T& b)
	{
		return map_ewise(times_scalar<T>(b), A.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<times_scalar<T>, LArg>
	operator * (const T& a, const IMatrixXpr<LArg, T>& B)
	{
		return map_ewise(times_scalar<T>(a), B.derived());
	}

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline LArg& operator *= (IDenseMatrix<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return A.derived() = A.derived() * B.derived();
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline LArg& operator *= (IDenseMatrix<LArg, T>& A, const T& b)
	{
		return A.derived() = A.derived() * b;
	}


	// divides

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline binary_ewise_expr<binary_divides<T>, LArg, RArg>
	operator / (const IMatrixXpr<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return map_ewise(binary_divides<T>(), A.derived(), B.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<divides_scalar<T>, LArg>
	operator / (const IMatrixXpr<LArg, T>& A, const T& b)
	{
		return map_ewise(divides_scalar<T>(b), A.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<rdivides_scalar<T>, LArg>
	operator / (const T& a, const IMatrixXpr<LArg, T>& B)
	{
		return map_ewise(rdivides_scalar<T>(a), B.derived());
	}

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline LArg& operator /= (IDenseMatrix<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return A.derived() = A.derived() / B.derived();
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline LArg& operator /= (IDenseMatrix<LArg, T>& A, const T& b)
	{
		return A.derived() = A.derived() / b;
	}


	// negate

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_negate<T>, Arg>
	operator - (const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_negate<T>(), A.derived());
	}

	// rcp

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_rcp<T>, Arg>
	rcp(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_rcp<T>(), A.derived());
	}

	// abs

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_abs<T>, Arg>
	abs(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_abs<T>(), A.derived());
	}


	// sqr

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_sqr<T>, Arg>
	sqr(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_sqr<T>(), A.derived());
	}

	// cube

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_cube<T>, Arg>
	cube(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_cube<T>(), A.derived());
	}

	// min

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline binary_ewise_expr<binary_min<T>, LArg, RArg>
	fmin(const IMatrixXpr<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return map_ewise(binary_min<T>(), A.derived(), B.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_min<T>, LArg>
	fmin(const IMatrixXpr<LArg, T>& A, const T& b)
	{
		return map_ewise(unary_min<T>(b), A.derived());
	}

	// max

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline binary_ewise_expr<binary_max<T>, LArg, RArg>
	fmax(const IMatrixXpr<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return map_ewise(binary_max<T>(), A.derived(), B.derived());
	}

	template<typename T, class LArg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_max<T>, LArg>
	fmax(const IMatrixXpr<LArg, T>& A, const T& b)
	{
		return map_ewise(unary_max<T>(b), A.derived());
	}


}

#endif /* MATRIX_ARITHMETIC_H_ */




