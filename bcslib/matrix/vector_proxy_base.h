/**
 * @file vector_proxy_base.h
 *
 * Basic definitions for vector proxies
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_VECTOR_PROXY_BASE_H_
#define BCSLIB_VECTOR_PROXY_BASE_H_

#include <bcslib/matrix/matrix_base.h>

namespace bcs
{
	template<class Derived, typename T>
	class IVecReader
	{
	public:
		BCS_CRTP_REF

		typedef T value_type;
		typedef index_t index_type;

		BCS_ENSURE_INLINE
		T load_scalar(const index_type i) const
		{
			return derived().load_scalar(i);
		}

	}; // end class IVecReader


	template<class Derived, typename T>
	class IVecWriter
	{
	public:
		BCS_CRTP_REF

		typedef T value_type;
		typedef index_t index_type;

		BCS_ENSURE_INLINE
		void store_scalar(const index_type i, const value_type& v)
		{
			derived().store_scalar(i, v);
		}

	}; // end class IVecWriter


	// generic functions


	template<typename T, class SMat, class DMat>
	BCS_ENSURE_INLINE
	inline void copy_vec(const index_t len, const IVecReader<SMat, T>& reader, IVecWriter<DMat, T>& writer)
	{
		for (index_t i = 0; i < len; ++i)
		{
			writer.store_scalar(i, reader.load_scalar(i));
		}
	}

	template<class Reductor, typename T, class Mat>
	BCS_ENSURE_INLINE
	inline typename Reductor::accum_type accum_vec(Reductor reduc, const index_t len,
			const IVecReader<Mat, T>& vec)
	{
		// pre-condition: len > 0

		typename Reductor::accum_type s = reduc(vec.load_scalar(0));
		for (index_t i = 1; i < len; ++i) s = reduc(s, vec.load_scalar(i));
		return s;
	}

	template<class Reductor, typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	inline typename Reductor::accum_type accum_vec(Reductor reduc, const index_t len,
			const IVecReader<LMat, T>& lvec, const IVecReader<RMat, T>& rvec)
	{
		// pre-condition: len > 0

		typename Reductor::accum_type s = reduc(lvec.load_scalar(0), rvec.load_scalar(0));
		for (index_t i = 1; i < len; ++i) s = reduc(s, lvec.load_scalar(i), rvec.load_scalar(i));
		return s;
	}

}

#endif /* VECTOR_PROXY_BASE_H_ */
