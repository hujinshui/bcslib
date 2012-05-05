/**
 * @file matrix_manip.h
 *
 * Memory operation functions for matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_MEMOP_H_
#define BCSLIB_MATRIX_MEMOP_H_

#include <bcslib/matrix/bits/matrix_memop_internal.h>
#include <bcslib/core/type_traits.h>

namespace bcs
{

	/********************************************
	 *
	 *  Matrix-level operations
	 *
	 ********************************************/

	// is_equal

	template<typename T, class LMat, class RMat>
	inline bool is_equal(const IMatrixView<LMat, T>& A, const IMatrixView<RMat, T>& B)
	{
		if (has_same_size(A, B))
		{
			typedef typename detail::equal_helper<LMat, RMat>::type helper_t;
			return helper_t::test(A.derived(), B.derived());
		}
		else
		{
			return false;
		}
	}

	// is_approx

	/**
	 * TODO: Replace this with a faster is_approx method
	 */
	template<typename T, class LMat, class RMat>
	inline bool is_approx(const IMatrixView<LMat, T>& A, const IMatrixView<RMat, T>& B, const T& tol)
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_floating_point<T>::value, "T must be a floating point value type");
#endif

		if (has_same_size(A, B))
		{
			index_t m = A.nrows();
			index_t n = A.ncolumns();

			for (index_t j = 0; j < n; ++j)
				for (index_t i = 0; i < m; ++i)
				{
					T a = A.elem(i, j);
					T b = B.elem(i, j);

					if (a > b + tol || b > a + tol) return false;
				}

			return true;
		}
		else
		{
			return false;
		}
	}


	// copy

	template<typename T, class SMat, class DMat>
	inline void copy(const IMatrixView<SMat, T>& src, IDenseMatrix<DMat, T>& dst)
	{
		typedef typename detail::copy_helper<SMat, DMat>::type helper_t;
		helper_t::run(src.derived(), dst.derived());
	}

	// fill

	template<typename T, class Mat>
	inline void fill(IDenseMatrix<Mat, T>& dst, const T& v)
	{
		typedef typename detail::fill_helper<Mat>::type helper_t;
		helper_t::run(dst.derived(), v);
	}


	// zero

	template<typename T, class Mat>
	inline void zero(IDenseMatrix<Mat, T>& dst)
	{
		typedef typename detail::zero_helper<Mat>::type helper_t;
		helper_t::run(dst.derived());
	}



}

#endif /* MATRIX_MANIP_H_ */
