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

#include <bcslib/matrix/vector_proxy.h>

namespace bcs { namespace detail {


	/********************************************
	 *
	 *  Evaluation
	 *
	 ********************************************/

	template<class Expr>
	struct ewise_evaluator
	{
		typedef vecwise_reader<Expr> reader_t;

		typedef typename matrix_traits<Expr>::value_type value_type;

		template<class DMat>
		static void run(const Expr& expr, IRegularMatrix<DMat, value_type>& dst)
		{
			reader_t reader(expr);
			vecwise_writer<DMat> writer(dst.derived());

			index_t m = expr.nrows();
			index_t n = expr.ncolumns();

			if (n == 1)
			{
				copy_vec(m, reader, writer);
			}
			else
			{
				for (index_t j = 0; j < n; ++j, ++reader, ++writer)
				{
					copy_vec(m, reader, writer);
				}
			}
		}

	};  // ewise_evaluator



} }

#endif



