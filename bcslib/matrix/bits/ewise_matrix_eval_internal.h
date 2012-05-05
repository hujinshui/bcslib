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

	template<class Expr, class DMat>
	BCS_ENSURE_INLINE
	inline void ewise_evaluate_as_single_vector(const Expr& src, DMat& dst)
	{
		typename vec_reader<Expr>::type in;
		typename vec_accessor<DMat>::type out;

		copy_vec(single_vecscheme<Expr>::get(src), in, out);
	}

	template<class Expr, class DMat>
	BCS_ENSURE_INLINE
	inline void ewise_evaluate_by_columns(const Expr& src, DMat& dst)
	{
		typename colwise_reader_set<Expr>::type in_set(src);
		typename colwise_accessor_set<DMat>::type out_set(dst);

		copy_vecs(colwise_vecscheme<Expr>::get(src), src.ncolumns(), in_set, out_set);
	}


	template<class Expr, class DMat, bool IsDMatLinearAccessible>
	struct ewise_evaluator;

	template<class Expr, class DMat>
	struct ewise_evaluator<Expr, DMat, true>
	{
		// (compile-time) calculation to select the better way to perform calculation

		static const bool has_short_columns =
				binary_ct_rows<Expr, DMat>::value <= ShortColumnBound;

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
				ewise_evaluate_by_columns(src, dst);
			}
			else
			{
				ewise_evaluate_as_single_vector(src, dst);
			}
		}
	};

	template<class Expr, class DMat>
	struct ewise_evaluator<Expr, DMat, false>
	{
		BCS_ENSURE_INLINE
		static void evaluate(const Expr& src, DMat& dst)
		{
			ewise_evaluate_by_columns(src, dst);
		}
	};





} }

#endif



