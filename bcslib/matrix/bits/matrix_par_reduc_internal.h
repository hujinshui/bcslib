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

#include <bcslib/matrix/vector_proxy.h>

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

		static void run(Reductor reduc, const Arg& arg, DMat& dst)
		{
			index_t m = arg.nrows();
			index_t n = arg.ncolumns();

			if (m > 0)
			{
				vecwise_reader<Arg> a(arg);

				if (n == 1)
				{
					if (m > 1)
					{
						accum_t s = accum_vec(reduc, m, a);
						dst(0, 0) = reduc.get(s, m);
					}
					else
					{
						accum_t s = reduc(a.load_scalar(0));
						dst(0, 0) = reduc.get(s, 1);
					}
				}
				else
				{
					if (m > 1)
					{
						for (index_t j = 0; j < n; ++j, ++a)
						{
							accum_t s = accum_vec(reduc, m, a);
							dst(0, j) = reduc.get(s, m);
						}
					}
					else
					{
						for (index_t j = 0; j < n; ++j, ++a)
						{
							accum_t s = reduc(a.load_scalar(0));
							dst(0, j) = reduc.get(s, 1);
						}
					}
				}
			}
			else
			{
				fill(dst, reduc());
			}
		}
	};


	template<class Reductor, class LArg, class RArg, class DMat>
	struct binary_colwise_reduction_evaluator
	{
		typedef typename Reductor::result_type result_type;
		typedef typename Reductor::accum_type accum_t;

		static void run(Reductor reduc, const LArg& larg, const RArg& rarg, DMat& dst)
		{
			index_t m = larg.nrows();
			index_t n = larg.ncolumns();

			if (m > 0)
			{
				vecwise_reader<LArg> a(larg);
				vecwise_reader<RArg> b(rarg);

				if (n == 1)
				{
					if (m > 1)
					{
						accum_t s = accum_vec(reduc, m, a, b);
						dst(0, 0) = reduc.get(s, m);
					}
					else
					{
						accum_t s = reduc(a.load_scalar(0), b.load_scalar(0));
						dst(0, 0) = reduc.get(s, 1);
					}
				}
				else
				{
					if (m > 1)
					{
						for (index_t j = 0; j < n; ++j, ++a, ++b)
						{
							accum_t s = accum_vec(reduc, m, a, b);
							dst(0, j) = reduc.get(s, m);
						}
					}
					else
					{
						for (index_t j = 0; j < n; ++j, ++a, ++b)
						{
							accum_t s = reduc(a.load_scalar(0), b.load_scalar(0));
							dst(0, j) = reduc.get(s, 1);
						}
					}
				}
			}
			else
			{
				fill(dst, reduc());
			}
		}
	};





	template<class Reductor, class Arg, class DMat>
	struct unary_rowwise_reduction_evaluator
	{
		typedef typename Reductor::result_type result_type;

		typedef typename Reductor::accum_type accum_t;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<result_type, accum_t>::value,
				"rowwise reduction is only supported when accum_type and result_type are identical.");
#endif

		static void run(Reductor reduc, const Arg& arg, DMat& dst)
		{
			index_t m = arg.nrows();
			index_t n = arg.ncolumns();

			if (n > 0)
			{
				vecwise_reader<Arg> a(arg);
				vecwise_reader<DMat> dr(dst);
				vecwise_writer<DMat> dw(dst);

				if (m == 1)
				{
					accum_t s = reduc(a.load_scalar(0));

					if (n > 1)
					{
						++ a;
						for (index_t j = 1; j < n; ++j, ++a) s = reduc(s, a.load_scalar(0));
					}
					dst(0, 0) = reduc.get(s, n);
				}
				else
				{
					for (index_t i = 0; i < m; ++i)
						dw.store_scalar(i, reduc(a.load_scalar(i)));

					if (n > 1)
					{
						++ a;
						for (index_t j = 1; j < n; ++j, ++a)
						{
							for (index_t i = 0; i < m; ++i)
								dw.store_scalar(i, reduc(dr.load_scalar(i), a.load_scalar(i)));
						}
					}

					for (index_t i = 0; i < m; ++i)
						dw.store_scalar(i, reduc.get(dr.load_scalar(i), n));
				}
			}
			else
			{
				fill(dst.derived(), reduc());
			}
		}
	};


	template<class Reductor, class LArg, class RArg, class DMat>
	struct binary_rowwise_reduction_evaluator
	{
		typedef typename Reductor::result_type result_type;

		typedef typename Reductor::accum_type accum_t;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::is_same<result_type, accum_t>::value,
				"rowwise reduction is only supported when accum_type and result_type are identical.");
#endif

		static void run(Reductor reduc, const LArg& larg, const RArg& rarg, DMat& dst)
		{
			index_t m = larg.nrows();
			index_t n = larg.ncolumns();

			if (n > 0)
			{
				vecwise_reader<LArg> a(larg);
				vecwise_reader<RArg> b(rarg);

				vecwise_reader<DMat> dr(dst);
				vecwise_writer<DMat> dw(dst);

				if (m == 1)
				{
					accum_t s = reduc(a.load_scalar(0), b.load_scalar(0));
					if (n > 1)
					{
						++ a; ++ b;
						for (index_t j = 1; j < n; ++j, ++a, ++b)
							s = reduc(s, a.load_scalar(0), b.load_scalar(0));
					}
					dst(0, 0) = reduc.get(s, n);
				}
				else
				{
					for (index_t i = 0; i < m; ++i)
						dw.store_scalar(i, reduc(a.load_scalar(i), b.load_scalar(i)));

					if (n > 1)
					{
						++ a; ++ b;
						for (index_t j = 1; j < n; ++j, ++a, ++b)
						{
							for (index_t i = 0; i < m; ++i)
								dw.store_scalar(i, reduc(dr.load_scalar(i), a.load_scalar(i), b.load_scalar(i)));
						}
					}

					for (index_t i = 0; i < m; ++i)
						dw.store_scalar(i, reduc.get(dr.load_scalar(i), n));
				}
			}
			else
			{
				fill(dst.derived(), reduc());
			}
		}
	};



} }

#endif /* MATRIX_PAR_REDUC_INTERNAL_H_ */
