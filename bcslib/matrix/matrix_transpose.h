/**
 * @file matrix_transpose.h
 *
 * Expression to represent matrix/vector transposition
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_TRANSPOSE_H_
#define BCSLIB_MATRIX_TRANSPOSE_H_

#include <bcslib/matrix/matrix_xpr.h>

#include <bcslib/matrix/dense_matrix.h>
#include <bcslib/matrix/ref_matrix.h>
#include <bcslib/matrix/ref_grid2d.h>

#include <bcslib/matrix/bits/matrix_transpose_internal.h>

namespace bcs
{

	/********************************************
	 *
	 *  matrix transposition expression
	 *
	 ********************************************/

	template<class Mat>
	struct matrix_traits<transpose_expr<Mat> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = ct_cols<Mat>::value;
		static const int compile_time_num_cols = ct_rows<Mat>::value;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef typename matrix_traits<Mat>::value_type value_type;
		typedef index_t index_type;
	};

	template<class Mat>
	struct has_continuous_layout<transpose_expr<Mat> >
	{
		static const bool value = false;
	};

	template<class Mat>
	struct is_always_aligned<transpose_expr<Mat> >
	{
		static const bool value = false;
	};

	template<class Mat>
	struct is_linear_accessible<transpose_expr<Mat> >
	{
		static const bool value = false;
	};

	template<class Mat>
	class transpose_expr
	: public IMatrixXpr<transpose_expr<Mat>, typename matrix_traits<Mat>::value_type>
	{
	public:
		BCS_MAT_TRAITS_DEFS(typename matrix_traits<Mat>::value_type)

		BCS_ENSURE_INLINE
		explicit transpose_expr(const Mat& mat)
		: m_mat(mat)
		{
		}

		BCS_ENSURE_INLINE
		const Mat& arg() const
		{
			return m_mat;
		}

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_mat.nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return m_mat.size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_mat.ncolumns();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_mat.nrows();
		}

	private:
		const Mat& m_mat;
	};


	template<class Mat>
	struct expr_evaluator<transpose_expr<Mat> >
	{
		typedef transpose_expr<Mat> expr_type;
		typedef typename matrix_traits<Mat>::value_type value_type;

		static const int ctrows = ct_rows<Mat>::value;
		static const int ctcols = ct_cols<Mat>::value;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, value_type>& dst)
		{
			const Mat& src = expr.arg();

			detail::matrix_transposer<value_type, ctrows, ctcols>::run(
					src.nrows(), src.ncolumns(),
					src.ptr_data(), src.lead_dim(),
					dst.ptr_data(), dst.lead_dim());
		}
	};



	/********************************************
	 *
	 *  dispatching
	 *
	 ********************************************/

	namespace detail
	{
		template<class Mat>
		struct column_transpose_helper
		{
			typedef typename matrix_traits<Mat>::value_type T;
			typedef cref_matrix<T, 1, ct_rows<Mat>::value> type;

			BCS_ENSURE_INLINE static type get(const Mat& mat)
			{
				return type(mat.ptr_data(), 1, mat.nrows());
			}
		};

		template<class Mat>
		struct continuous_row_transpose_helper
		{
			typedef typename matrix_traits<Mat>::value_type T;
			typedef cref_matrix<T, ct_cols<Mat>::value, 1> type;

			BCS_ENSURE_INLINE static type get(const Mat& mat)
			{
				return type(mat.ptr_data(), mat.ncolumns(), 1);
			}
		};

		template<class Mat>
		struct general_row_transpose_helper
		{
			typedef typename matrix_traits<Mat>::value_type T;
			typedef cref_grid2d<T, ct_cols<Mat>::value, 1> type;

			BCS_ENSURE_INLINE static type get(const Mat& mat)
			{
				return type(mat.ptr_data(), mat.ncolumns(), 1, mat.lead_dim(), 0);
			}
		};

		template<class Mat>
		struct general_mat_transpose_helper
		{
			typedef transpose_expr<Mat> type;

			BCS_ENSURE_INLINE static type get(const Mat& mat)
			{
				return type(mat);
			}
		};
	}

	template<class Mat>  // Mat MUST have interface IDenseMatrix
	struct transposed
	{
		typedef typename select_type<ct_is_col<Mat>::value,
					detail::column_transpose_helper<Mat>,  // is column
					typename select_type<ct_is_row<Mat>::value,
						typename select_type<has_continuous_layout<Mat>::value,
							detail::continuous_row_transpose_helper<Mat>,  // is row (continuous)
							detail::general_row_transpose_helper<Mat> 	// is row (non-continuous)
						>::type, // is row
						detail::general_mat_transpose_helper<Mat> // general matrix (not column/row)
					>::type
				>::type helper_t;

		typedef typename helper_t::type type;

		BCS_ENSURE_INLINE static type get(const Mat& mat)
		{
			return helper_t::get(mat);
		}
	};


	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	typename transposed<Mat>::type transpose(const IDenseMatrix<Mat, T>& mat)
	{
		return transposed<Mat>::get(mat.derived());
	}



}

#endif /* MATRIX_TRANSPOSE_H_ */
