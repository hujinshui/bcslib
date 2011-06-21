/**
 * @file generic_array_transpose.h
 *
 * The generic_implementation of array transposition
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_TRANSPOSE_H
#define BCSLIB_ARRAY_TRANSPOSE_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_mem.h>
#include <bcslib/array/array_base.h>

#include <algorithm>
#include <cmath>

#define BCS_TRANSPOSITION_BLOCK_BYTES 1024

namespace bcs
{

	/**
	 * Transpose the matrix in source to destination (using direct assignment)
	 *
	 * Note: This is out-of-place transposition, and there
	 * should be no overlap between src and dst
	 *
	 * The source have m slices, each of length n,
	 * The result have n slices, each of length m.
	 */
	template<typename T>
	inline void direct_transpose_matrix(const T *src, T *dst, size_t m, size_t n)
	{
		for (size_t i = 0; i < m; ++i)
		{
			T *pd = dst + i;
			for (size_t j = 0; j < n; ++j, pd += m)
			{
				*pd = *(src++);
			}
		}
	}


	template<typename T>
	inline void direct_transpose_sqmatrix_inplace(T *a, size_t n)
	{
		using std::swap;

		for (size_t i = 1; i < n; ++i)
		{
			for (size_t j = 0; j < i; ++j)
			{
				swap(a[i * n + j], a[j * n + i]);
			}
		}
	}


	namespace _detail
	{
		template<typename T>
		inline void _tr_process_block(const T *src, T *dst, T *cache, size_t m, size_t n,
				size_t bdim, size_t bi, size_t bj,
				size_t bm, size_t bn, size_t bu)
		{
			size_t ii = bi * bdim;
			size_t jj = bj * bdim;

			const T *ps = src + (ii * n + jj);
			T *pd = dst + (jj * m + ii);

			// copy source block [bm x bn] to cache [bu x bu]
			T *c = cache;
			for (size_t i = 0; i < bm; ++i, ps += n, c += bu)
			{
				for (size_t j = 0; j < bn; ++j)
				{
					c[j] = ps[j];
				}
			}

			// within-cache transpose
			direct_transpose_sqmatrix_inplace(cache, bu);

			// copy transposed cache [bu x bu] to destination block [bn x bm]
			c = cache;
			for (size_t i = 0; i < bn; ++i, c += bu, pd += m)
			{
				for (size_t j = 0; j < bm; ++j)
				{
					pd[j] = c[j];
				}
			}
		}

	}



	template<typename T>
	void blockwise_transpose_matrix(const T *src, T *dst, size_t m, size_t n, size_t bdim, T *cache)
	{
		size_t n_br = m / bdim;
		size_t n_bc = n / bdim;

		size_t rm = m - n_br * bdim;
		size_t rn = n - n_bc * bdim;

		// deal with main blocks [bdim x bdim]

		for (size_t bi = 0; bi < n_br; ++bi)
		{
			for (size_t bj = 0; bj < n_bc; ++bj)
			{
				_detail::_tr_process_block(src, dst, cache, m, n, bdim, bi, bj, bdim, bdim, bdim);
			}
		}

		// deal with right-edge blocks [bdim x rn]

		if (rn > 0)
		{
			for (size_t bi = 0; bi < n_br; ++bi)
			{
				_detail::_tr_process_block(src, dst, cache, m, n, bdim, bi, n_bc, bdim, rn, bdim);
			}
		}

		// deal with bottom-edge blocks [rm x bdim]

		if (rm > 0)
		{
			for (size_t bj = 0; bj < n_bc; ++bj)
			{
				_detail::_tr_process_block(src, dst, cache, m, n, bdim, n_br, bj, rm, bdim, bdim);
			}
		}

		// deal with block at bottom-right [rm x rn]

		size_t rbu = std::max(rm, rn);
		if (rbu > 0)
		{
			_detail::_tr_process_block(src, dst, cache, m, n, bdim, n_br, n_bc, rm, rn, rbu);
		}
	}



	template<typename T>
	void transpose_matrix(const T *src, T *dst, size_t m, size_t n)
	{
		const size_t block_size = BCS_TRANSPOSITION_BLOCK_BYTES / sizeof(T);

		aligned_array<T, block_size> cache;

		if (block_size < 4)
		{
			direct_transpose_matrix(src, dst, m, n);
		}
		else
		{
			size_t bdim = (size_t)std::sqrt((double)block_size);
			blockwise_transpose_matrix(src, dst, m, n, bdim, cache.data);
		}
	}


	namespace _detail
	{
		template<typename TOrd> struct _transposer_helper;

		template<> struct _transposer_helper<row_major_t>
		{
			static void get_mn(const std::array<index_t, 2>& shape, size_t& m, size_t& n)
			{
				m = static_cast<size_t>(shape[0]);
				n = static_cast<size_t>(shape[1]);
			}
		};

		template<> struct _transposer_helper<column_major_t>
		{
			static void get_mn(const std::array<index_t, 2>& shape, size_t& m, size_t& n)
			{
				m = static_cast<size_t>(shape[1]);
				n = static_cast<size_t>(shape[0]);
			}
		};
	}


	template<class Arr>
	inline typename std::enable_if<is_array_view_ndim<Arr, 2>::value,
	typename array_creater<Arr>::result_type>::type
	transpose(const Arr& a)
	{
		typedef typename array_view_traits<Arr>::value_type value_t;
		typedef typename array_creater<Arr>::result_type result_t;
		typedef typename array_view_traits<Arr>::layout_order ord_t;

		auto sh = get_array_shape(a);
		auto rsh = arr_shape(sh[1], sh[0]);

		size_t m, n;
		_detail::_transposer_helper<ord_t>::get_mn(sh, m, n);

		result_t r = array_creater<Arr>::create(rsh);

		if (is_dense_view(a))
		{
			transpose_matrix(ptr_base(a), ptr_base(r), m, n);
		}
		else
		{
			scoped_buffer<value_t> buf(m * n);
			std::copy_n(begin(a), m * n, buf.pbase());
			transpose_matrix(buf.pbase(), ptr_base(r), m, n);
		}

		return r;
	}


	template<class Arr>
	struct transpose_ftor
	{
		typedef typename array_creater<Arr>::result_type result_type;

		result_type evaluate(const Arr& a)
		{
			return transpose(a);
		}
	};

}


#endif 
