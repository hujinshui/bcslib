/*
 * @file matrix_broadcast.h
 *
 * Broadcast calculation on matrices
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_BROADCAST_H_
#define BCSLIB_MATRIX_BROADCAST_H_

#include <bcslib/matrix/matrix_xpr.h>
#include <bcslib/matrix/slicewise_proxy.h>
#include <bcslib/matrix/repeat_vectors.h>

#include <bcslib/engine/arithmetic_functors.h>


namespace bcs
{
	/********************************************
	 *
	 *  Generic Expressions
	 *
	 ********************************************/


	template<class Fun, class LMat, class RVec>
	struct colwise_map_type
	{
		typedef binary_ewise_expr<Fun, LMat, repeat_cols_expr<RVec, DynamicDim> > type;
	};

	template<class Fun, class LMat, class RVec>
	struct rowwise_map_type
	{
		typedef binary_ewise_expr<Fun, LMat, repeat_rows_expr<RVec, DynamicDim> > type;
	};


	template<class Fun, typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename colwise_map_type<Fun, LMat, RVec>::type
	map_ewise(Fun f, const_colwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_binary_ewise_functor<Fun>::value, "Fun must be a binary-ewise-functor");
#endif
		return map_ewise( f, A.ref(), repeat_cols(b.derived(), A.ref().ncolumns()) );
	}


	template<class Fun, typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename rowwise_map_type<Fun, LMat, RVec>::type
	map_ewise(Fun f, const_rowwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_binary_ewise_functor<Fun>::value, "Fun must be a binary-ewise-functor");
#endif
		return map_ewise( f, A.ref(), repeat_rows(b.derived(), A.ref().nrows()) );
	}


	/********************************************
	 *
	 *  Specific Expressions
	 *
	 ********************************************/

	// plus

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename colwise_map_type<binary_plus<T>, LMat, RVec>::type
	operator + (const_colwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		return map_ewise(binary_plus<T>(), A, b);
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline void operator += (colwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		A.ref() = A + b;
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename rowwise_map_type<binary_plus<T>, LMat, RVec>::type
	operator + (const_rowwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		return map_ewise(binary_plus<T>(), A, b);
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline void operator += (rowwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		A.ref() = A + b;
	}


	// minus

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename colwise_map_type<binary_minus<T>, LMat, RVec>::type
	operator - (const_colwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		return map_ewise(binary_minus<T>(), A, b);
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline void operator -= (colwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		A.ref() = A - b;
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename rowwise_map_type<binary_minus<T>, LMat, RVec>::type
	operator - (const_rowwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		return map_ewise(binary_minus<T>(), A, b);
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline void operator -= (rowwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		A.ref() = A - b;
	}


	// times

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename colwise_map_type<binary_times<T>, LMat, RVec>::type
	operator * (const_colwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		return map_ewise(binary_times<T>(), A, b);
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline void operator *= (colwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		A.ref() = A * b;
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename rowwise_map_type<binary_times<T>, LMat, RVec>::type
	operator * (const_rowwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		return map_ewise(binary_times<T>(), A, b);
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline void operator *= (rowwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		A.ref() = A * b;
	}


	// divides

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename colwise_map_type<binary_divides<T>, LMat, RVec>::type
	operator / (const_colwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		return map_ewise(binary_divides<T>(), A, b);
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline void operator /= (colwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		A.ref() = A / b;
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline typename rowwise_map_type<binary_divides<T>, LMat, RVec>::type
	operator / (const_rowwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		return map_ewise(binary_divides<T>(), A, b);
	}

	template<typename T, class LMat, class RVec>
	BCS_ENSURE_INLINE
	inline void operator /= (rowwise_proxy<LMat, T> A, const IMatrixXpr<RVec, T>& b)
	{
		A.ref() = A / b;
	}

}

#endif 
