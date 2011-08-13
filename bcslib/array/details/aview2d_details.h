/**
 * @file array2d_details.h
 *
 * Implementation details of array2d
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY2D_DETAILS_H
#define BCSLIB_ARRAY2D_DETAILS_H

#include <bcslib/base/basic_defs.h>

#define BCS_TRANSPOSITION_BLOCK_BYTES 1024

namespace bcs
{
	namespace _detail
	{

		/**************************************************
		 *
		 *  layout order specific indexing implementation
		 *
		 **************************************************/

		template<typename TOrd> class index_core2d;

		template<>
		class index_core2d<row_major_t>
		{
		public:
			index_core2d(index_t base_d0, index_t base_d1)
			: m_base_d0(base_d0), m_base_d1(base_d1)
			{
			}

			index_t base_nelems() const
			{
				return m_base_d0 * m_base_d1;
			}

			index_t base_dim0() const
			{
				return m_base_d0;
			}

			index_t base_dim1() const
			{
				return m_base_d1;
			}

			array_shape_t<2> base_shape() const
			{
				return arr_shape(m_base_d0, m_base_d1);
			}

			index_t offset(index_t i, index_t j) const
			{
				return i * m_base_d1 + j;
			}

		private:
			index_t m_base_d0;
			index_t m_base_d1;
		};

		template<>
		class index_core2d<column_major_t>
		{
		public:
			index_core2d(index_t base_d0, index_t base_d1)
			: m_base_d0(base_d0), m_base_d1(base_d1)
			{
			}

			index_t base_nelems() const
			{
				return m_base_d0 * m_base_d1;
			}

			index_t base_dim0() const
			{
				return m_base_d0;
			}

			index_t base_dim1() const
			{
				return m_base_d1;
			}

			array_shape_t<2> base_shape() const
			{
				return arr_shape(m_base_d0, m_base_d1);
			}

			index_t offset(index_t i, index_t j) const
			{
				return i + m_base_d0 * j;
			}

		private:
			index_t m_base_d0;
			index_t m_base_d1;
		};


		template<typename T, typename TOrd> struct slice_helper2d;

		template<typename T>
		struct slice_helper2d<T, row_major_t>
		{
			typedef caview1d<T> row_cview_type;
			typedef aview1d<T>  row_view_type;
			typedef caview1d_ex<T, step_ind> column_cview_type;
			typedef aview1d_ex<T, step_ind>  column_view_type;

			static row_cview_type row_cview(const T* pbase, index_t d0, index_t d1, index_t irow)
			{
				return caview1d<T>(pbase + irow * d1, d1);
			}

			static row_view_type row_view(T *pbase, index_t d0, index_t d1, index_t irow)
			{
				return aview1d<T>(pbase + irow * d1, d1);
			}

			static column_cview_type column_cview(const T* pbase, index_t d0, index_t d1, index_t icol)
			{
				return caview1d_ex<T, step_ind>(pbase + icol, step_ind(d0, d1));
			}

			static column_view_type column_view(T* pbase, index_t d0, index_t d1, index_t icol)
			{
				return aview1d_ex<T, step_ind>(pbase + icol, step_ind(d0, d1));
			}
		};


		template<typename T>
		struct slice_helper2d<T, column_major_t>
		{
			typedef caview1d<T> column_cview_type;
			typedef aview1d<T>  column_view_type;
			typedef caview1d_ex<T, step_ind> row_cview_type;
			typedef aview1d_ex<T, step_ind>  row_view_type;

			static row_cview_type row_cview(const T* pbase, index_t d0, index_t d1, index_t irow)
			{
				return caview1d_ex<T, step_ind>(pbase + irow, step_ind(d1, d0));
			}

			static row_view_type row_view(T *pbase, index_t d0, index_t d1, index_t irow)
			{
				return aview1d_ex<T, step_ind>(pbase + irow, step_ind(d1, d0));
			}

			static column_cview_type column_cview(const T* pbase, index_t d0, index_t d1, index_t icol)
			{
				return caview1d<T>(pbase + icol * d0, d0);
			}

			static column_view_type column_view(T* pbase, index_t d0, index_t d1, index_t icol)
			{
				return aview1d<T>(pbase + icol * d0, d0);
			}
		};


		template<typename T, typename TOrd, class TRange> struct slice_range_helper2d;

		template<typename T, class TRange>
		struct slice_range_helper2d<T, row_major_t, TRange>
		{
			typedef index_core2d<row_major_t> index_core_t;

			typedef caview1d_ex<T, typename indexer_map<TRange>::type> row_range_cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::type>  row_range_view_type;
			typedef caview1d_ex<T, typename indexer_map<TRange>::stepped_type> column_range_cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::stepped_type>  column_range_view_type;

			static row_range_cview_type row_range_cview(const T *pbase, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return caview1d_ex<T, typename indexer_map<TRange>::type>(
						pbase + d1 * irow + indexer_map<TRange>::get_offset(d1, rgn),
						indexer_map<TRange>::get_indexer(d1, rgn));
			}

			static row_range_view_type row_range_view(T *pbase, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return aview1d_ex<T, typename indexer_map<TRange>::type>(
						pbase + d1 * irow + indexer_map<TRange>::get_offset(d1, rgn),
						indexer_map<TRange>::get_indexer(d1, rgn));
			}

			static column_range_cview_type column_range_cview(const T *pbase, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return caview1d_ex<T, typename indexer_map<TRange>::stepped_type>(
						pbase + d1 * indexer_map<TRange>::get_offset(d0, rgn) + icol,
						indexer_map<TRange>::get_stepped_indexer(d0, d1, rgn));
			}

			static column_range_view_type column_range_view(T *pbase, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return aview1d_ex<T, typename indexer_map<TRange>::stepped_type>(
						pbase + d1 * indexer_map<TRange>::get_offset(d0, rgn) + icol,
						indexer_map<TRange>::get_stepped_indexer(d0, d1, rgn));
			}
		};

		template<typename T, class TRange>
		struct slice_range_helper2d<T, column_major_t, TRange>
		{
			typedef index_core2d<column_major_t> index_core_t;

			typedef caview1d_ex<T, typename indexer_map<TRange>::stepped_type> row_range_cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::stepped_type>  row_range_view_type;
			typedef caview1d_ex<T, typename indexer_map<TRange>::type> column_range_cview_type;
			typedef aview1d_ex<T, typename indexer_map<TRange>::type>  column_range_view_type;

			static row_range_cview_type row_range_cview(const T *pbase, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return caview1d_ex<T, typename indexer_map<TRange>::stepped_type>(
						pbase + d0 * indexer_map<TRange>::get_offset(d1, rgn) + irow,
						indexer_map<TRange>::get_stepped_indexer(d1, d0, rgn));
			}

			static row_range_view_type row_range_view(T *pbase, index_t d0, index_t d1, index_t irow, const TRange& rgn)
			{
				return aview1d_ex<T, typename indexer_map<TRange>::stepped_type>(
						pbase + d0 * indexer_map<TRange>::get_offset(d1, rgn) + irow,
						indexer_map<TRange>::get_stepped_indexer(d1, d0, rgn));
			}

			static column_range_cview_type column_range_cview(const T *pbase, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return caview1d_ex<T, typename indexer_map<TRange>::type>(
						pbase + d0 * icol + indexer_map<TRange>::get_offset(d0, rgn),
						indexer_map<TRange>::get_indexer(d0, rgn));
			}

			static column_range_view_type column_range_view(T *pbase, index_t d0, index_t d1, index_t icol, const TRange& rgn)
			{
				return aview1d_ex<T, typename indexer_map<TRange>::type>(
						pbase + d0 * icol + indexer_map<TRange>::get_offset(d0, rgn),
						indexer_map<TRange>::get_indexer(d0, rgn));
			}
		};

	}


	/**************************************************
	 *
	 *  transposition implementation
	 *
	 **************************************************/


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
}

#endif 
