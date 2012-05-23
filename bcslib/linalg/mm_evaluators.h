/**
 * @file mm_evaluators.h
 *
 * Matrix product evalutor classes
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MM_EVALUATORS_H_
#define BCSLIB_MM_EVALUATORS_H_

#include <bcslib/linalg/linalg_base.h>
#include <bcslib/linalg/bits/mm_evaluators_internal.h>

namespace bcs
{

	template<class LArg, class RArg>
	class mm_evaluator
	{
	public:
		typedef typename matrix_traits<LArg>::value_type T;

		template<class DMat>
		BCS_ENSURE_INLINE static void eval(
				const T& alpha, const LArg& larg, const RArg& rarg,
				const T& beta, IDenseMatrix<DMat, T>& dst)
		{
			typedef typename detail::mm_evaluator_dispatcher<LArg, RArg>::type impl_t;

			impl_t::eval(alpha, larg, rarg, beta, dst.derived());
		}
	};



}

#endif /* MM_EVALUATORS_H_ */
