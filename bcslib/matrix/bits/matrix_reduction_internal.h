/**
 * @file matrix_reduction_internal.h
 *
 * Internal implementation of full reduction on matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef MATRIX_REDUCTION_INTERNAL_H_
#define MATRIX_REDUCTION_INTERNAL_H_

#include <bcslib/matrix/vector_operations.h>
#include <bcslib/matrix/ewise_matrix_expr.h>

namespace bcs { namespace detail {

	// generic reduction

	template<class Reductor, class Expr>
	BCS_ENSURE_INLINE
	inline typename Reductor::result_type
	full_reduce_as_single_vector(Reductor reduc, const Expr& a)
	{
		typename vec_reader<Expr>::type in;

		typedef typename Reductor::accum_type accum_t;
		accum_t s = accum_vec(single_vecscheme<Expr>::get(a), reduc, in);
		return reduc.get(s, a.nelems());
	}


	template<class Reductor, class LExpr, class RExpr>
	BCS_ENSURE_INLINE
	inline typename Reductor::result_type
	full_reduce_as_single_vector(Reductor reduc, const LExpr& a, const RExpr& b)
	{
		typename vec_reader<LExpr>::type in_a;
		typename vec_reader<RExpr>::type in_b;

		typedef typename Reductor::accum_type accum_t;
		typedef typename binary_nil_expr<LExpr, RExpr> dummy_t;

		accum_t s = accum_vec(single_vecscheme<dummy_t>::get(dummy_t(a, b)), reduc, in_a, in_b);
		return reduc.get(s, a.nelems());
	}

	template<class Reductor, class Expr>
	inline typename Reductor::result_type
	full_reduce_by_columns(Reductor reduc, const Expr& a)
	{
		typedef typename colwise_reader_set<Expr>::type reader_set_t;
		typedef typename reader_set_t::reader_type reader_t;
		typedef typename Reductor::accum_type accum_t;

		typename colwise_vecscheme<Expr>::type sch = colwise_vecscheme<Expr>::get(a);

		reader_set_t in_set(a);

		reader_t in0(in_set, 0);
		accum_t s = accum_vec(sch, reduc, in0);

		const index_t n = a.ncolumns();
		for (index_t j = 1; j < n; ++j)
		{
			reader_t in(in_set, j);
			s = reduc.combine(s, accum_vec(sch, reduc, in));
		}

		return reduc.get(s, a.nelems());
	}


	template<class Reductor, class LExpr, class RExpr>
	inline typename Reductor::result_type
	full_reduce_by_columns(Reductor reduc, const LExpr& a, const RExpr& b)
	{
		typedef typename colwise_reader_set<LExpr>::type left_reader_set_t;
		typedef typename colwise_reader_set<RExpr>::type right_reader_set_t;
		typedef typename left_reader_set_t::reader_type left_reader_t;
		typedef typename right_reader_set_t::reader_type right_reader_t;

		typedef typename Reductor::accum_type accum_t;
		typedef typename binary_nil_expr<LExpr, RExpr> dummy_t;

		typename colwise_vecscheme<dummy_t>::type sch = colwise_vecscheme<dummy_t>::get(dummy_t(a, b));

		left_reader_set_t  a_in_set(a);
		right_reader_set_t b_in_set(b);

		left_reader_t a_in0(a_in_set, 0);
		right_reader_t b_in0(b_in_set, 0);
		accum_t s = accum_vec(sch, reduc, a_in0, b_in0);

		const index_t n = a.ncolumns();
		for (index_t j = 1; j < n; ++j)
		{
			left_reader_t a_in(a_in_set, j);
			right_reader_t b_in(b_in_set, j);
			s = reduc.combine(s, accum_vec(sch, reduc, a_in, b_in));
		}

		return reduc.get(s, a.nelems());
	}


	// Unary

	template<typename Reductor, class Expr>
	struct unary_reduction_eval_helper
	{
		// (compile-time) calculation to select the better way to perform calculation

		static const bool has_short_columns = (ct_rows<Expr>::value <= ShortColumnBound);

		static const bool prefer_by_columns_way =
				(has_short_columns ?
						(vecacc_cost<Expr, by_short_columns_tag>::value <
						 vecacc_cost<Expr, as_single_vector_tag>::value)
						:
						(vecacc_cost<Expr, by_columns_tag>::value <
						 vecacc_cost<Expr, as_single_vector_tag>::value)
				);


		BCS_ENSURE_INLINE
		static typename Reductor::result_type
		run(Reductor reduc, const Expr& A)
		{
			typedef typename Reductor::accum_type accum_t;

			if (!is_empty(A))
			{
				if (prefer_by_columns_way)
				{
					return full_reduce_by_columns(reduc, A);
				}
				else
				{
					return full_reduce_as_single_vector(reduc, A);
				}
			}
			else
			{
				return reduc();
			}
		}
	};


	// Binary

	template<typename Reductor, class LExpr, class RExpr>
	struct binary_reduction_eval_helper
	{
		// (compile-time) calculation to select the better way to perform calculation

		static const bool has_short_columns = (binary_ct_rows<LExpr, RExpr>::value <= ShortColumnBound);

		static const bool prefer_by_columns_way =
				(has_short_columns ?
						(vecacc2_cost<LExpr, RExpr, by_short_columns_tag>::value <
						 vecacc2_cost<LExpr, RExpr, as_single_vector_tag>::value)
						:
						(vecacc2_cost<LExpr, RExpr, by_columns_tag>::value <
						 vecacc2_cost<LExpr, RExpr, as_single_vector_tag>::value)
				);


		BCS_ENSURE_INLINE
		static typename Reductor::result_type
		run(Reductor reduc, const LExpr& A, const RExpr& B)
		{
			typedef typename Reductor::accum_type accum_t;

			if (!is_empty(A))
			{
				if (prefer_by_columns_way)
				{
					return full_reduce_by_columns(reduc, A, B);
				}
				else
				{
					return full_reduce_as_single_vector(reduc, A, B);
				}
			}
			else
			{
				return reduc();
			}
		}
	};


} }

#endif /* MATRIX_REDUCTION_INTERNAL_H_ */




