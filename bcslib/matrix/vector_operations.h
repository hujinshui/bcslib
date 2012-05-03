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

	/********************************************
	 *
	 *  general operations
	 *
	 ********************************************/

	template<class Scheme, typename T, class SVec, class DVec>
	BCS_ENSURE_INLINE
	void copy_vec(const Scheme& sch,
			const IVecReader<SVec, T>& in, IVecAccessor<DVec, T>& out)
	{
		detail::vec_copy_helper<Scheme>::run(sch, in.derived(), out.derived());
	}

	template<class Scheme, typename T, class SVS, class DVS>
	inline void copy_vecs(const Scheme& sch, const index_t n,
			const IVecReaderSet<SVS, T>& in_set, IVecAccessorSet<DVS, T>& out_set)
	{
		for (index_t j = 0; j < n; ++j)
		{
			typename SVS::reader_type in(in_set, j);
			typename DVS::accessor_type out(out_set, j);
			copy_vec(sch, in, out);
		}
	}


	template<class Scheme, class Reductor, typename T, class Mat>
	BCS_ENSURE_INLINE
	inline typename Reductor::accum_type accum_vec(const Scheme& sch,
			Reductor reduc, const IVecReader<Mat, T>& in)
	{
		return detail::vec_accum_helper<Scheme>::run(reduc, in);
	}

	template<class Scheme, class Reductor, typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	inline typename Reductor::accum_type accum_vec(const Scheme& sch,
			Reductor reduc, const IVecReader<LMat, T>& left_in, const IVecReader<RMat, T>& right_in)
	{
		return detail::vec_accum_helper<Scheme>::run(reduc, left_in, right_in);
	}


}


#endif /* VECTOR_OPERATIONS_H_ */
