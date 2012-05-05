/**
 * @file vector_operations.h
 *
 * Operations on vector reader/accessors
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_VECTOR_OPERATIONS_H_
#define BCSLIB_VECTOR_OPERATIONS_H_

#include <bcslib/matrix/vector_accessors.h>

namespace bcs
{

	/********************************************
	 *
	 *  transfer
	 *
	 ********************************************/

	template<int N, class SVec, class DVec>
	struct transfer_vec
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(N >= 2, "The value of N must have N >= 2");
#endif

		BCS_ENSURE_INLINE
		static void run(const index_t, const SVec& in, DVec& out)
		{
			for (index_t i = 0; i < N; ++i)
			{
				out.set(i, in.get(i));
			}
		}
	};


	template<class SVec, class DVec>
	struct transfer_vec<DynamicDim, SVec, DVec>
	{
		BCS_ENSURE_INLINE
		static void run(const index_t len, const SVec& in, DVec& out)
		{
			for (index_t i = 0; i < len; ++i)
			{
				out.set(i, in.get(i));
			}
		}
	};


	template<class SVec, class DVec>
	struct transfer_vec<1, SVec, DVec>
	{
		BCS_ENSURE_INLINE
		static void run(const index_t, const SVec& in, DVec& out)
		{
			out.set(0, in.get(0));
		}
	};


}


#endif /* VECTOR_OPERATIONS_H_ */
