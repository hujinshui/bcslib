/**
 * @file matrix_transpose_internal.h
 *
 * Internal implementation of matrix transposition
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_TRANSPOSE_INTERNAL_H_
#define BCSLIB_MATRIX_TRANSPOSE_INTERNAL_H_

#include <bcslib/core/basic_defs.h>

namespace bcs { namespace detail {

	// TODO: replace this with a faster algorithm
	template<typename T, int CTRows, int CTCols>
	struct matrix_transposer
	{
		// m x n --> n x m
		static void run(const index_t m, const index_t n,
				const T* __restrict__ src, const index_t src_ldim,
				T* __restrict__ dst, const index_t dst_ldim)
		{
			if (m > n)
			{
				for (index_t j = 0; j < n; ++j)
					unpack_vector(m, src + j * src_ldim, dst + j, dst_ldim);
			}
			else
			{
				for (index_t i = 0; i < m; ++i)
					pack_vector(n, src + i, src_ldim, dst + i * dst_ldim);
			}
		}
	};



} }

#endif
