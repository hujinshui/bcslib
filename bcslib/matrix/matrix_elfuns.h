/**
 * @file matrix_elfuns.h
 *
 * Elementary functions on Matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_ELFUNS_H_
#define BCSLIB_MATRIX_ELFUNS_H_

#include <bcslib/matrix/ewise_matrix_expr.h>
#include <bcslib/math/elementary_functors.h>

namespace bcs
{
	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_sqrt<T>, Arg>
	sqrt(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_sqrt<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_pow<T>, Arg>
	pow(const IMatrixXpr<Arg, T>& A, const T& e)
	{
		return map_ewise(unary_pow<T>(e), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_exp<T>, Arg>
	exp(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_exp<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_log<T>, Arg>
	log(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_log<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_log10<T>, Arg>
	log10(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_log10<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_floor<T>, Arg>
	floor(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_floor<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_ceil<T>, Arg>
	ceil(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_ceil<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_sin<T>, Arg>
	sin(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_sin<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_cos<T>, Arg>
	cos(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_cos<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_tan<T>, Arg>
	tan(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_tan<T>(), A.derived());
	}


	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_asin<T>, Arg>
	asin(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_asin<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_acos<T>, Arg>
	acos(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_acos<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_atan<T>, Arg>
	atan(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_atan<T>(), A.derived());
	}

	template<typename T, class LArg, class RArg>
	BCS_ENSURE_INLINE
	inline binary_ewise_expr<binary_atan2<T>, LArg, RArg>
	atan2(const IMatrixXpr<LArg, T>& A, const IMatrixXpr<RArg, T>& B)
	{
		return map_ewise(binary_atan2<T>(), A.derived(), B.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_sinh<T>, Arg>
	sinh(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_sinh<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_cosh<T>, Arg>
	cosh(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_cosh<T>(), A.derived());
	}

	template<typename T, class Arg>
	BCS_ENSURE_INLINE
	inline unary_ewise_expr<unary_tanh<T>, Arg>
	tanh(const IMatrixXpr<Arg, T>& A)
	{
		return map_ewise(unary_tanh<T>(), A.derived());
	}



}

#endif /* MATRIX_ELFUNS_H_ */
