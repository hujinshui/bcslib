/**
 * @file matrix_transpose_internal.h
 *
 * Internal implementation for Matrix transposition
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_TRANSPOSE_INTERNAL_H_
#define BCSLIB_MATRIX_TRANSPOSE_INTERNAL_H_


#include <bcslib/matrix/matrix_capture.h>

namespace bcs { namespace detail {

	template<typename T>
	inline void direct_transpose(index_t m, index_t n,
			const T* __restrict__ a,  index_t lda, T * __restrict__ b, index_t ldb)
	{
		// a: m x n --> b : n x m  (no overlap between a and b)

		if (m > n)
		{
			for (index_t j = 0; j < n; ++j, a += lda) bcs::unpack_vector(m, a, b + j, ldb);
		}
		else
		{
			for (index_t i = 0; i < m; ++i, b += ldb) bcs::pack_vector(n, a + i, lda, b);
		}
	}


	/**
	 * TODO: replace this dummy one with a faster algorithm
	 */
	template<typename T, int CTRows, int CTCols>
	struct default_transpose_algorithm
	{
		BCS_ENSURE_INLINE
		void run(index_t m, index_t n,
				const T* __restrict__ a,  index_t lda, T * __restrict__ b, index_t ldb)
		{
			direct_transpose(m, n, a, lda, b, ldb);
		}
	};


	template<class Src, class Dst>
	struct transpose_alg_selector
	{
		typedef typename matrix_traits<Src>::value_type value_type;

		static const int src_M = ct_rows<Src>::value;
		static const int src_N = ct_cols<Src>::value;

		static const int dst_M = ct_rows<Dst>::value;
		static const int dst_N = ct_cols<Dst>::value;

		static const int M = (src_M > 0 ? src_M : dst_N);
		static const int N = (src_N > 0 ? src_N : dst_M);

		typedef default_transpose_algorithm<value_type, M, N> type;
	};


	template<class Src, class Dst, bool IsDstDense> struct matrix_transposer;

	template<class Src, class Dst>
	struct matrix_transposer<Src, Dst, true>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(bcs::has_matrix_interface<Dst, IDenseMatrix>::value,
				"Dst should be a model of IDenseMatrix");
#endif

		typedef typename transpose_alg_selector<Src, Dst>::type alg_t;

		static void run(const Src& src, Dst& dst)
		{
			typedef typename matrix_traits<Src>::value_type T;
			matrix_capture<Src, bcs::has_matrix_interface<Src, IDenseMatrix>::value> scap(src);

			alg_t::run(src.nrows(), src.ncolumns(),
					scap.get().ptr_data(), scap.get().lead_dim(), dst.ptr_data(), dst.lead_dim());
		}
	};


	template<class Src, class Dst>
	struct matrix_transposer<Src, Dst, false>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(!bcs::has_matrix_interface<Dst, IDenseMatrix>::value,
				"Dst should NOT be a model of IDenseMatrix");
#endif

		typedef typename transpose_alg_selector<Src, Dst>::type alg_t;

		static void run(const Src& src, Dst& dst)
		{
			typedef typename matrix_traits<Src>::value_type T;
			matrix_capture<Src, bcs::has_matrix_interface<Src, IDenseMatrix>::value> scap(src);

			index_t m = src.nrows();
			index_t n = src.ncolumns();
			dense_matrix<T, ct_rows<Dst>::value, ct_cols<Dst>::value> dtmp(n, m);

			alg_t::run(m, n,
					scap.get().ptr_data(), scap.get().lead_dim(), dtmp.ptr_data(), dtmp.lead_dim());
			bcs::copy(dtmp, dst);
		}
	};


} }

#endif /* MATRIX_TRANSPOSE_INTERNAL_H_ */
