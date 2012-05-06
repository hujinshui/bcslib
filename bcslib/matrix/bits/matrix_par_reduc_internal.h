/**
 * @file matrix_par_reduc_internal.h
 *
 * Internal implementation of matrix partial reduction
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_PAR_REDUC_INTERNAL_H_
#define BCSLIB_MATRIX_PAR_REDUC_INTERNAL_H_

#include <bcslib/matrix/vector_operations.h>

namespace bcs { namespace detail {


	/********************************************
	 *
	 *  Evaluation
	 *
	 ********************************************/

	template<class Reductor, class Arg, class DMat>
	struct unary_colwise_reduction_evaluator
	{
		typedef typename Reductor::result_type result_type;
		typedef typename Reductor::accum_type accum_t;

		typedef typename colwise_reader_bank<Arg>::type bank_t;
		typedef typename bank_t::reader_type reader_t;

		static void run(Reductor reduc, const Arg& arg, DMat& dst)
		{
			index_t m = arg.nrows();
			index_t n = arg.ncolumns();

			if (m > 0)
			{
				bank_t bank(arg);

				for (index_t j = 0; j < n; ++j)
				{
					reader_t in(bank, j);
					accum_t s = accum_vec<Reductor,
							ct_rows<Arg>::value, reader_t>::run(reduc, m, in);

					dst(0, j) = reduc.get(s, m);
				}
			}
			else
			{
				fill(dst, reduc.empty_result());
			}
		}
	};


	template<class Reductor, class LArg, class RArg, class DMat>
	struct binary_colwise_reduction_evaluator
	{
		typedef typename Reductor::result_type result_type;
		typedef typename Reductor::accum_type accum_t;

		typedef typename colwise_reader_bank<LArg>::type left_bank_t;
		typedef typename colwise_reader_bank<RArg>::type right_bank_t;
		typedef typename left_bank_t::reader_type left_reader_t;
		typedef typename right_bank_t::reader_type right_reader_t;

		static void run(Reductor reduc, const LArg& larg, const RArg& rarg, DMat& dst)
		{
			index_t m = larg.nrows();
			index_t n = larg.ncolumns();

			if (m > 0)
			{
				left_bank_t left_bank(larg);
				right_bank_t right_bank(rarg);

				for (index_t j = 0; j < n; ++j)
				{
					left_reader_t left_in(left_bank, j);
					right_reader_t right_in(right_bank, j);

					accum_t s = accum_vec2<Reductor,
							binary_ct_rows<LArg, RArg>::value,
							left_reader_t,
							right_reader_t>::run(reduc, m, left_in, right_in);

					dst(0, j) = reduc.get(s, m);
				}
			}
			else
			{
				fill(dst, reduc.empty_result());
			}
		}
	};





	template<class Reductor, class Arg, class DMat>
	struct unary_rowwise_reduction_evaluator
	{
		typedef typename Reductor::result_type result_t;
		typedef typename Reductor::accum_type accum_t;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<result_t, accum_t>::value,
				"rowwise reduction is only supported when accum_type and result_type are identical.");
#endif

		typedef typename colwise_reader_bank<Arg>::type bank_t;
		typedef typename bank_t::reader_type reader_t;

		static void run(Reductor reduc, const Arg& arg, DMat& dst)
		{
			index_t m = arg.nrows();
			index_t n = arg.ncolumns();

			if (n > 0)
			{
				bank_t bank(arg);
				typename vec_accessor<DMat>::type out(dst);

				reader_t in0(bank, 0);
				for (index_t i = 0; i < m; ++i)
					out.set(i, reduc.init(in0.get(i)));

				for (index_t j = 1; j < n; ++j)
				{
					reader_t in(bank, j);
					for (index_t i = 0; i < m; ++i)
						out.set(i, reduc.add(out.get(i), in.get(i)));
				}

				for (index_t i = 0; i < m; ++i)
					out.set(i, reduc.get(out.get(i), n));
			}
			else
			{
				fill(dst.derived(), reduc.empty_result());
			}
		}
	};


	template<class Reductor, class LArg, class RArg, class DMat>
	struct binary_rowwise_reduction_evaluator
	{
		typedef typename Reductor::result_type result_t;
		typedef typename Reductor::accum_type accum_t;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<result_t, accum_t>::value,
				"rowwise reduction is only supported when accum_type and result_type are identical.");
#endif

		typedef typename colwise_reader_bank<LArg>::type left_bank_t;
		typedef typename left_bank_t::reader_type left_reader_t;
		typedef typename colwise_reader_bank<RArg>::type right_bank_t;
		typedef typename right_bank_t::reader_type right_reader_t;

		static void run(Reductor reduc, const LArg& larg, const RArg& rarg, DMat& dst)
		{
			index_t m = larg.nrows();
			index_t n = larg.ncolumns();

			if (n > 0)
			{
				left_bank_t left_bank(larg);
				right_bank_t right_bank(rarg);
				typename vec_accessor<DMat>::type out(dst);

				left_reader_t left_in0(left_bank, 0);
				right_reader_t right_in0(right_bank, 0);

				for (index_t i = 0; i < m; ++i)
					out.set(i, reduc.init(left_in0.get(i), right_in0.get(i)));

				for (index_t j = 1; j < n; ++j)
				{
					left_reader_t left_in(left_bank, j);
					right_reader_t right_in(right_bank, j);

					for (index_t i = 0; i < m; ++i)
						out.set(i, reduc.add(out.get(i), left_in.get(i), right_in.get(i)));
				}

				for (index_t i = 0; i < m; ++i)
					out.set(i, reduc.get(out.get(i), n));
			}
			else
			{
				fill(dst.derived(), reduc.empty_result());
			}
		}
	};



} }

#endif /* MATRIX_PAR_REDUC_INTERNAL_H_ */
