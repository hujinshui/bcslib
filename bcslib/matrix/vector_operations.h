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

#include <bcslib/matrix/bits/vector_operations_internal.h>

namespace bcs
{

	template<typename T, class SVec, class DVec, class Scheme>
	BCS_ENSURE_INLINE
	void copy_vec(const IVecReader<SVec, T>& src, IVecWriter<DVec, T>& dst, const Scheme& sch)
	{
		detail::vec_copy_helper<SVec, DVec, Scheme>::run(src.derived(), dst.derived(), sch);
	}


	template<typename T, class DVec, class Scheme>
	BCS_ENSURE_INLINE
	void set_vec(IVecWriter<DVec, T>& dst, const T& v, const Scheme& sch)
	{
		detail::vec_set_helper<T, DVec, Scheme>::run(dst.derived(), v, sch);
	}


}


#endif /* VECTOR_OPERATIONS_H_ */
