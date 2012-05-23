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

	template<class LArg, class RArg> class mm_evaluator;


	template<class Mat> class sym_mat_proxy;

	template<class Mat>
	struct matrix_traits<sym_mat_proxy<Mat> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_rows<Mat>::value;
		static const int compile_time_num_cols = ct_cols<Mat>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<Mat>::value_type value_type;
		typedef index_t index_type;
	};

	template<class Mat>
	class sym_mat_proxy
	: public IMatrixXpr<sym_mat_proxy<Mat>, typename matrix_traits<Mat>::value_type>
	{
	public:
		typedef Mat mat_type;

		BCS_MAT_TRAITS_CDEFS(typename matrix_traits<Mat>::value_type)

		BCS_ENSURE_INLINE
		explicit sym_mat_proxy(const Mat& mat)
		: m_mat(mat)
		{
			check_arg(mat.nrows() == mat.ncolumns(),
					"mat must be a square matrix for sym_mat_proxy");
		}

		BCS_ENSURE_INLINE const Mat& get() const
		{
			return m_mat;
		}

		BCS_ENSURE_INLINE index_type nelems() const { return m_mat.nelems(); }

		BCS_ENSURE_INLINE size_type size() const { return m_mat.size(); }

		BCS_ENSURE_INLINE index_type nrows() const { return m_mat.nrows(); }

		BCS_ENSURE_INLINE index_type ncolumns() const { return m_mat.ncolumns(); }

	private:
		const Mat& m_mat;
	};


	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	sym_mat_proxy<Mat> as_sym(const IMatrixXpr<Mat, T>& mat)
	{
		return sym_mat_proxy<Mat>(mat.derived());
	}

}

#endif /* LINALG_BASE_H_ */
