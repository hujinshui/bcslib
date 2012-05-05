/**
 * @file ewise_matrix_eval.h
 *
 * Internal implementation for element-wise evaluation
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_EWISE_MATRIX_EVAL_INTERNAL_H_
#define BCSLIB_EWISE_MATRIX_EVAL_INTERNAL_H_

#include <bcslib/matrix/ewise_matrix_expr.h>
#include <bcslib/matrix/vector_operations.h>

namespace bcs { namespace detail {


	/********************************************
	 *
	 *  Evaluation
	 *
	 ********************************************/

	template<class Expr, class DMat, int CTSize>
	BCS_ENSURE_INLINE
	inline void ewise_evaluate_as_single_vector(const Expr& src, DMat& dst)
	{
		typedef typename vec_reader<Expr>::type in_t;
		typedef typename vec_accessor<DMat>::type out_t;

		in_t in(src);
		out_t out(dst);
		transfer_vec<CTSize, in_t, out_t>::run(src.nelems(), in, out);
	}

	template<class Expr, class DMat, int CTRows>
	inline void ewise_evaluate_by_columns(const Expr& src, DMat& dst)
	{
		typedef typename colwise_reader_bank<Expr>::type in_bank_t;
		typedef typename colwise_accessor_bank<DMat>::type out_bank_t;
		typedef typename in_bank_t::reader_type in_t;
		typedef typename out_bank_t::accessor_type out_t;

		typedef typename matrix_traits<DMat>::value_type T;

		in_bank_t in_bank(src);
		out_bank_t out_bank(dst);

		const index_t m = src.nrows();
		const index_t n = src.ncolumns();

		for (index_t j = 0; j < n; ++j)
		{
			in_t in(in_bank, j);
			out_t out(out_bank, j);
			transfer_vec<CTRows, in_t, out_t>::run(m, in, out);
		}
	}


	template<class Expr, class DMat, bool IsDMatLinearAccessible>
	struct ewise_evaluator;

	template<class Expr, class DMat>
	struct ewise_evaluator<Expr, DMat, true>
	{
		// (compile-time) calculation to select the better way to perform calculation

		static const int comptime_nrows = binary_ct_rows<Expr, DMat>::value;
		static const int comptime_size = binary_ct_size<Expr, DMat>::value;

		static const bool has_short_columns = (comptime_nrows <= ShortColumnBound);

		static const bool prefer_by_columns_way =
				(has_short_columns ?
						(vecacc2_cost<Expr, DMat, by_short_columns_tag>::value <
						 vecacc2_cost<Expr, DMat, as_single_vector_tag>::value)
						:
						(vecacc2_cost<Expr, DMat, by_columns_tag>::value <
						 vecacc2_cost<Expr, DMat, as_single_vector_tag>::value)
				);

		BCS_ENSURE_INLINE
		static void evaluate(const Expr& src, DMat& dst)
		{
			if (prefer_by_columns_way)
			{
				ewise_evaluate_by_columns<Expr, DMat, comptime_nrows>(src, dst);
			}
			else
			{
				ewise_evaluate_as_single_vector<Expr, DMat, comptime_size>(src, dst);
			}
		}
	};

	template<class Expr, class DMat>
	struct ewise_evaluator<Expr, DMat, false>
	{
		static const int comptime_nrows = binary_ct_rows<Expr, DMat>::value;

		BCS_ENSURE_INLINE
		static void evaluate(const Expr& src, DMat& dst)
		{
			ewise_evaluate_by_columns<Expr, DMat, comptime_nrows>(src, dst);
		}
	};


} }

#endif



