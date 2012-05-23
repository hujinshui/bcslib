/**
 * @file linalg_base.h
 *
 * Basic definitions for Linear Algebra
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_LINALG_BASE_H_
#define BCSLIB_LINALG_BASE_H_

#include <bcslib/matrix.h>

namespace bcs
{
	template<class Mat>
	class sym_mat_proxy
	{
	public:

		BCS_ENSURE_INLINE
		explicit sym_mat_proxy(const Mat& mat)
		: m_mat(mat)
		{
			check_arg(mat.nrows() == mat.ncolumns(),
					"mat must be a square matrix for sym_mat_proxy");
		}

		BCS_ENSURE_INLINE
		const Mat& get() const
		{
			return m_mat;
		}

	private:
		const Mat& m_mat;
	};


	struct mm_col_tag { };
	struct mm_row_tag { };
	struct mm_mat_tag { };
	struct mm_tmat_tag { };
	struct mm_smat_tag { };

}

#endif /* LINALG_BASE_H_ */
