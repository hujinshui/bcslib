/**
 * @file matrix_redux.h
 *
 * The matrix reduction functions
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_REDUCTION_H_
#define BCSLIB_MATRIX_REDUCTION_H_

#include <bcslib/matrix/bits/matrix_reduction_internal.h>
#include <bcslib/engine/reduction_functors.h>

namespace bcs
{

	/********************************************
	 *
	 *  Generic full-reduction evaluation
	 *
	 ********************************************/

	template<typename Reductor, class Mat>
	BCS_ENSURE_INLINE
	typename Reductor::result_type
	reduce(const Reductor& reduc,
			const IMatrixXpr<Mat, typename Reductor::argument_type>& A)
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_unary_reduction_functor<Reductor>::value,
				"Reductor must be a unary reduction functor");
#endif

		return detail::unary_reduction_eval_helper<
				Reductor, Mat, is_accessible_as_vector<Mat>::value>::run(reduc, A.derived());
	}

	template<typename Reductor, class LMat, class RMat>
	BCS_ENSURE_INLINE
	typename Reductor::result_type
	reduce(const Reductor& reduc,
			const IMatrixXpr<LMat, typename Reductor::argument_type>& A,
			const IMatrixXpr<RMat, typename Reductor::argument_type>& B)
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_binary_reduction_functor<Reductor>::value,
				"Reductor must be a binary reduction functor");
#endif

		check_arg( is_same_size(A, B), "The sizes of two operands are inconsistent.");

		return detail::binary_reduction_eval_helper<
				Reductor, LMat, RMat,
				is_accessible_as_vector<LMat>::value && is_accessible_as_vector<RMat>::value
				>::run(reduc, A.derived(), B.derived());
	}



	/********************************************
	 *
	 *  Specific reduction functions
	 *
	 ********************************************/

	// Unary

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	T sum(const IMatrixXpr<Mat, T>& A)
	{
		return reduce(sum_reductor<T>(), A);
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	T mean(const IMatrixXpr<Mat, T>& A)
	{
		return reduce(mean_reductor<T>(), A);
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	T min_val(const IMatrixXpr<Mat, T>& A)
	{
		return reduce(min_reductor<T>(), A);
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	T max_val(const IMatrixXpr<Mat, T>& A)
	{
		return reduce(max_reductor<T>(), A);
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	T L1norm(const IMatrixXpr<Mat, T>& A)
	{
		return reduce(L1norm_reductor<T>(), A);
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	T sqL2norm(const IMatrixXpr<Mat, T>& A)
	{
		return reduce(sqL2norm_reductor<T>(), A);
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	T L2norm(const IMatrixXpr<Mat, T>& A)
	{
		return reduce(L2norm_reductor<T>(), A);
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	T Linfnorm(const IMatrixXpr<Mat, T>& A)
	{
		return reduce(Linfnorm_reductor<T>(), A);
	}


	// Binary

	template<typename T, class LMat, class RMat >
	BCS_ENSURE_INLINE
	T dot_prod(const IMatrixXpr<LMat, T>& A, const IMatrixXpr<RMat, T>& B)
	{
		return reduce(dot_reductor<T>(), A, B);
	}

	template<typename T, class LMat, class RMat >
	BCS_ENSURE_INLINE
	T L1norm_diff(const IMatrixXpr<LMat, T>& A, const IMatrixXpr<RMat, T>& B)
	{
		return reduce(L1diffnorm_reductor<T>(), A, B);
	}

	template<typename T, class LMat, class RMat >
	BCS_ENSURE_INLINE
	T sqL2norm_diff(const IMatrixXpr<LMat, T>& A, const IMatrixXpr<RMat, T>& B)
	{
		return reduce(sqL2diffnorm_reductor<T>(), A, B);
	}

	template<typename T, class LMat, class RMat >
	BCS_ENSURE_INLINE
	T L2norm_diff(const IMatrixXpr<LMat, T>& A, const IMatrixXpr<RMat, T>& B)
	{
		return reduce(L2diffnorm_reductor<T>(), A, B);
	}

	template<typename T, class LMat, class RMat >
	BCS_ENSURE_INLINE
	T Linfnorm_diff(const IMatrixXpr<LMat, T>& A, const IMatrixXpr<RMat, T>& B)
	{
		return reduce(Linfdiffnorm_reductor<T>(), A, B);
	}

}

#endif /* MATRIX_REDUX_H_ */
