/**
 * @file matrix_helpers.h
 *
 * Some helpful devices for matrix implementation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_HELPERS_H_
#define BCSLIB_MATRIX_HELPERS_H_

#include <bcslib/matrix/matrix_base.h>

namespace bcs
{
	namespace detail
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		inline void check_matrix_indices(const Mat& mat, index_t i, index_t j)
		{
#ifndef BCSLIB_NO_DEBUG
			check_arg(i >= 0 && i < mat.nrows() && j >= 0 && j < mat.ncolumns(),
					"Index out of boundary.");
#endif
		}

		template<class LMat, class RMat>
		BCS_ENSURE_INLINE
		inline void check_rhs_view(const LMat& lhs, const RMat& rhs)
		{
			index_t lm = lhs.nrows();
			index_t ln = lhs.ncolumns();

			bool is_fit;

			if (ln == 1)  // this is column
			{
				is_fit = ((rhs.nrows() == 1 && rhs.ncolumns() == lm) ||
						  (rhs.nrows() == lm && rhs.ncolumns() == 1));
			}
			else if (lm == 1) // this is row
			{
				is_fit = ((rhs.nrows() == 1 && rhs.ncolumns() == ln) ||
						  (rhs.nrows() == ln && rhs.ncolumns() == 1));
			}
			else
			{
				is_fit = (lm == rhs.nrows() && ln == rhs.ncolumns());
			}

			check_arg(is_fit, "The right hand side assignment does not fit the view");
		}


		template<bool SingleRow, bool SingleColumn> struct offset_helper;

		template<> struct offset_helper<false, false>
		{
			BCS_ENSURE_INLINE
			static inline index_t calc(index_t lead_dim, index_t i, index_t j)
			{
				return i + lead_dim * j;
			}
		};

		template<> struct offset_helper<false, true>
		{
			BCS_ENSURE_INLINE
			static inline index_t calc(index_t lead_dim, index_t i, index_t)
			{
				return i;
			}
		};

		template<> struct offset_helper<true, false>
		{
			BCS_ENSURE_INLINE
			static inline index_t calc(index_t lead_dim, index_t, index_t j)
			{
				return j;
			}
		};


		template<> struct offset_helper<true, true>
		{
			BCS_ENSURE_INLINE
			static inline index_t calc(index_t lead_dim, index_t, index_t)
			{
				return 0;
			}
		};


		template<int RowDim, int ColDim>
		BCS_ENSURE_INLINE
		inline index_t calc_offset(index_t lead_dim, index_t i, index_t j)
		{
			return offset_helper<RowDim == 1, ColDim == 1>::calc(lead_dim, i, j);
		}

	}

}

#endif /* MATRIX_HELPERS_H_ */
