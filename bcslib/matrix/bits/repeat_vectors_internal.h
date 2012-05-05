/*
 * @file repeat_vectors_internal.h
 *
 * The internal implementation of repeat-vectors
 *
 * @author Dahua Lin
 */

#ifndef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REPEAT_VECTORS_INTERNAL_H_
#define BCSLIB_REPEAT_VECTORS_INTERNAL_H_

#include <bcslib/matrix/vector_operations.h>

namespace bcs { namespace detail {


	template<class Arg, class DMat, int CTRows>
	struct repeat_cols_evaluator
	{
		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef matrix_capture<Arg, is_dense_mat<Arg>::value> capture_t;

		static void evaluate(const Arg& col, const index_t n, DMat& dst)
		{
			capture_t cap(col);
			const value_type *src = cap.get().ptr_data();

			for (index_t j = 0; j < n; ++j)
			{
				mem<value_type, CTRows>::copy(src, col_ptr(dst, j));
			}
		}
	};

	template<class Arg, class DMat>
	struct repeat_cols_evaluator<Arg, DMat, DynamicDim>
	{
		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef matrix_capture<Arg, is_dense_mat<Arg>::value> capture_t;

		static void evaluate(const Arg& col, const index_t n, DMat& dst)
		{
			capture_t cap(col);
			const value_type *src = cap.get().ptr_data();
			const index_t m = col.nrows();

			for (index_t j = 0; j < n; ++j)
			{
				copy_elems(m, src, col_ptr(dst, j));
			}
		}
	};


	template<class Arg, class DMat, int CTRows>
	struct repeat_rows_evaluator
	{
		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef matrix_capture<Arg, is_linear_accessible<Arg>::value> capture_t;

		static void evaluate(const Arg& row, const index_t m, DMat& dst)
		{
			capture_t cap(row);
			const typename capture_t::captured_type& src = cap.get();
			const index_t n = row.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				mem<value_type, CTRows>::fill(col_ptr(dst, j), src[j]);
			}
		}
	};


	template<class Arg, class DMat>
	struct repeat_rows_evaluator<Arg, DMat, DynamicDim>
	{
		typedef typename matrix_traits<Arg>::value_type value_type;
		typedef matrix_capture<Arg, is_linear_accessible<Arg>::value> capture_t;

		static void evaluate(const Arg& row, const index_t m, DMat& dst)
		{
			capture_t cap(row);
			const typename capture_t::captured_type& src = cap.get();
			const index_t n = row.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				fill_elems(m, col_ptr(dst, j), src[j]);
			}
		}
	};




} }


#endif 

