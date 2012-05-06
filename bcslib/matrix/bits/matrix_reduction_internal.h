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

namespace bcs { namespace detail {

	// generic reduction

	template<class Reductor, class Expr>
	BCS_ENSURE_INLINE
	inline typename Reductor::result_type
	full_reduce_as_single_vector(const Reductor& reduc, const Expr& a)
	{
		typedef typename Reductor::accum_type accum_t;
		typedef typename vec_reader<Expr>::type in_t;

		const index_t n = a.nelems();
		in_t in(a);

		accum_t s = accum_vec<Reductor, ct_size<Expr>::value, in_t>::run(reduc, n, in);

		return reduc.get(s, n);
	}


	template<class Reductor, class LExpr, class RExpr>
	BCS_ENSURE_INLINE
	inline typename Reductor::result_type
	full_reduce_as_single_vector(Reductor reduc, const LExpr& a, const RExpr& b)
	{
		typedef typename Reductor::accum_type accum_t;
		typedef typename vec_reader<LExpr>::type left_in_t;
		typedef typename vec_reader<RExpr>::type right_in_t;

		const index_t n = a.nelems();
		left_in_t in1(a);
		right_in_t in2(b);

		accum_t s = accum_vec2<Reductor,
				binary_ct_size<LExpr, RExpr>::value, left_in_t, right_in_t>::run(reduc, n, in1, in2);
		return reduc.get(s, n);
	}

	template<class Reductor, class Expr>
	inline typename Reductor::result_type
	full_reduce_by_columns(Reductor reduc, const Expr& a)
	{
		typedef typename Reductor::accum_type accum_t;
		typedef typename colwise_reader_bank<Expr>::type bank_t;
		typedef typename bank_t::reader_type in_t;

		const index_t m = a.nrows();
		const index_t n = a.ncolumns();

		bank_t bank(a);

		in_t in0(bank, 0);
		accum_t s = accum_vec<Reductor, ct_rows<Expr>::value, in_t>::run(reduc, m, in0);

		for (index_t j = 1; j < n; ++j)
		{
			in_t in(bank, j);

			accum_t sj = accum_vec<Reductor, ct_rows<Expr>::value, in_t>::run(reduc, m, in);

			s = reduc.combine(s, sj);
		}

		return reduc.get(s, a.nelems());
	}


	template<class Reductor, class LExpr, class RExpr>
	inline typename Reductor::result_type
	full_reduce_by_columns(Reductor reduc, const LExpr& a, const RExpr& b)
	{
		typedef typename Reductor::accum_type accum_t;
		typedef typename colwise_reader_bank<LExpr>::type left_bank_t;
		typedef typename colwise_reader_bank<RExpr>::type right_bank_t;
		typedef typename left_bank_t::reader_type left_in_t;
		typedef typename right_bank_t::reader_type right_in_t;

		const index_t m = a.nrows();
		const index_t n = a.ncolumns();

		left_bank_t bank_a(a);
		right_bank_t bank_b(b);

		left_in_t in_a0(bank_a, 0);
		right_in_t in_b0(bank_b, 0);
		accum_t s = accum_vec2<Reductor,
				binary_ct_rows<LExpr, RExpr>::value, left_in_t, right_in_t>::run(reduc, m, in_a0, in_b0);

		for (index_t j = 1; j < n; ++j)
		{
			left_in_t in_a(bank_a, j);
			right_in_t in_b(bank_b, j);

			accum_t sj = accum_vec2<Reductor,
					binary_ct_rows<LExpr, RExpr>::value, left_in_t, right_in_t>::run(reduc, m, in_a, in_b);

			s = reduc.combine(s, sj);
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
				return reduc.empty_result();
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
				return reduc.empty_result();
			}
		}
	};


} }

#endif /* MATRIX_REDUCTION_INTERNAL_H_ */




