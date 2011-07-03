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
#include <type_traits>
#include <array>

#define BCS_TRANSPOSITION_BLOCK_BYTES 1024

namespace bcs
{
	struct slice2d_info
	{
		size_t nslices;
		size_t len;
		index_t stride;

		bool is_compact() const
		{
			return (index_t)len == stride;
		}
	};

	namespace _detail
	{
		/**************************************************
		 *
		 *  iterator implementation
		 *
		 **************************************************/

		template<typename T>
		class array2d_iterator_impl
		{
		public:
			BCS_STATIC_ASSERT( !std::is_reference<T>::value  );

			typedef typename std::remove_const<T>::type value_type;
			typedef T& reference;
			typedef T* pointer;

		public:
			array2d_iterator_impl()
			: m_p(BCS_NULL), m_p_slend(BCS_NULL), m_slice_blen(0), m_slice_elen(0)
			{
			}

			array2d_iterator_impl(pointer p, pointer p_slend, index_t sl_blen, index_t sl_elen)
			: m_p(p), m_p_slend(p_slend), m_slice_blen(sl_blen), m_slice_elen(sl_elen)
			{
			}

			pointer ptr() const
			{
				return m_p;
			}

			reference ref() const
			{
				return *m_p;
			}

			bool operator == (const array2d_iterator_impl& rhs) const
			{
				return m_p == rhs.m_p;
			}

			void move_next()
			{
				++ m_p;

				if (m_p == m_p_slend)
				{
					m_p_slend += m_slice_blen;	// move slice end to next slice
					m_p = m_p_slend - m_slice_elen; 	// infer the starting of such slice
				}
			}

		public:
			static array2d_iterator_impl slice_begin_iter(pointer p, index_t sl_blen, index_t sl_elen)
			{
				return array2d_iterator_impl(p, p + sl_elen, sl_blen, sl_elen);
			}

		private:
			pointer m_p;
			pointer m_p_slend;

			index_t m_slice_blen;	// base length (whole line)
			index_t m_slice_elen; 	// effective length

		}; // end class array2d_iterator_impl



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

			size_t base_size() const
			{
				return static_cast<size_t>(m_base_d0 * m_base_d1);
			}

			index_t base_dim0() const
			{
				return m_base_d0;
			}

			index_t base_dim1() const
			{
				return m_base_d1;
			}

			std::array<index_t, 2> base_shape() const
			{
				return arr_shape(m_base_d0, m_base_d1);
			}

		public:
			index_t offset(index_t i, index_t j) const
			{
				return i * m_base_d1 + j;
			}

			index_t row_offset(index_t i) const
			{
				return i * m_base_d1;
			}

			index_t column_offset(index_t j) const
			{
				return j;
			}

			slice2d_info get_slice_info(size_t m, size_t n) const
			{
				slice2d_info s;
				s.nslices = m;
				s.len = n;
				s.stride = m_base_d1;
				return s;
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

			size_t base_size() const
			{
				return static_cast<size_t>(m_base_d0 * m_base_d1);
			}

			index_t base_dim0() const
			{
				return m_base_d0;
			}

			index_t base_dim1() const
			{
				return m_base_d1;
			}

			std::array<index_t, 2> base_shape() const
			{
				return arr_shape(m_base_d0, m_base_d1);
			}

		public:
			index_t offset(index_t i, index_t j) const
			{
				return i + m_base_d0 * j;
			}

			index_t row_offset(index_t i) const
			{
				return i;
			}

			index_t column_offset(index_t j) const
			{
				return m_base_d0 * j;
			}

			slice2d_info get_slice_info(size_t m, size_t n) const
			{
				slice2d_info s;
				s.nslices = n;
				s.len = m;
				s.stride = m_base_d0;
				return s;
			}

		private:
			index_t m_base_d0;
			index_t m_base_d1;
		};


		template<typename T, typename TOrd> struct iter_helper2d;

		template<typename T>
		struct iter_helper2d<T, row_major_t>
		{
			typedef index_core2d<row_major_t> index_core_t;
			typedef forward_iterator_wrapper<array2d_iterator_impl<const T> > const_iterator;
			typedef forward_iterator_wrapper<array2d_iterator_impl<T> > iterator;

			static const_iterator get_const_begin(const T *pbase, const index_core_t& idxcore, index_t d0, index_t d1)
			{
				return array2d_iterator_impl<const T>::slice_begin_iter(pbase, idxcore.base_dim1(), d1);
			}

			static const_iterator get_const_end(const T *pbase, const index_core_t& idxcore, index_t d0, index_t d1)
			{
				return array2d_iterator_impl<const T>::slice_begin_iter(pbase + idxcore.row_offset(d0), idxcore.base_dim1(), d1);
			}

			static iterator get_begin(T *pbase, const index_core_t& idxcore, index_t d0, index_t d1)
			{
				return array2d_iterator_impl<T>::slice_begin_iter(pbase, idxcore.base_dim1(), d1);
			}

			static iterator get_end(T *pbase, const index_core_t& idxcore, index_t d0, index_t d1)
			{
				return array2d_iterator_impl<T>::slice_begin_iter(pbase + idxcore.row_offset(d0), idxcore.base_dim1(), d1);
			}
		};


		template<typename T>
		struct iter_helper2d<T, column_major_t>
		{
			typedef index_core2d<column_major_t> index_core_t;
			typedef forward_iterator_wrapper<array2d_iterator_impl<const T> > const_iterator;
			typedef forward_iterator_wrapper<array2d_iterator_impl<T> > iterator;

			static const_iterator get_const_begin(const T *pbase, const index_core_t& idxcore, index_t d0, index_t d1)
			{
				return array2d_iterator_impl<const T>::slice_begin_iter(pbase, idxcore.base_dim0(), d0);
			}

			static const_iterator get_const_end(const T *pbase, const index_core_t& idxcore, index_t d0, index_t d1)
			{
				return array2d_iterator_impl<const T>::slice_begin_iter(pbase + idxcore.column_offset(d1), idxcore.base_dim0(), d0);
			}

			static iterator get_begin(T *pbase, const index_core_t& idxcore, index_t d0, index_t d1)
			{
				return array2d_iterator_impl<T>::slice_begin_iter(pbase, idxcore.base_dim0(), d0);
			}

			static iterator get_end(T *pbase, const index_core_t& idxcore, index_t d0, index_t d1)
			{
				return array2d_iterator_impl<T>::slice_begin_iter(pbase + idxcore.column_offset(d1), idxcore.base_dim0(), d0);
			}
		};



		template<typename T, typename TOrd> struct slice_helper2d;

		template<typename T>
		struct slice_helper2d<T, row_major_t>
		{
			typedef index_core2d<row_major_t> index_core_t;

			typedef caview1d<T> row_cview_type;
			typedef aview1d<T>  row_view_type;
			typedef caview1d_ex<T, step_range> column_cview_type;
			typedef aview1d_ex<T, step_range>  column_view_type;

			static row_cview_type row_cview(const index_core_t& idxcore, const T *prowbase, size_t rowlen)
			{
				return caview1d<T>(prowbase, rowlen);
			}

			static row_view_type row_view(const index_core_t& idxcore, T *prowbase, size_t rowlen)
			{
				return aview1d<T>(prowbase, rowlen);
			}

			static column_cview_type column_cview(const index_core_t& idxcore, const T *pcolbase, size_t collen)
			{
				return caview1d_ex<T, step_range>( pcolbase,
						step_range::from_begin_dim(0, (index_t)collen, idxcore.base_dim1()) );
			}

			static column_view_type column_view(const index_core_t& idxcore, T *pcolbase, size_t collen)
			{
				return aview1d_ex<T, step_range>( pcolbase,
						step_range::from_begin_dim(0, (index_t)collen, idxcore.base_dim1()) );
			}
		};


		template<typename T>
		struct slice_helper2d<T, column_major_t>
		{
			typedef index_core2d<column_major_t> index_core_t;

			typedef caview1d<T> column_cview_type;
			typedef aview1d<T>  column_view_type;
			typedef caview1d_ex<T, step_range> row_cview_type;
			typedef aview1d_ex<T, step_range>  row_view_type;

			static column_cview_type column_cview(const index_core_t& idxcore, const T *pcolbase, size_t collen)
			{
				return caview1d<T>(pcolbase, collen);
			}

			static column_view_type column_view(const index_core_t& idxcore, T *pcolbase, size_t collen)
			{
				return aview1d<T>(pcolbase, collen);
			}

			static row_cview_type row_cview(const index_core_t& idxcore, const T *prowbase, size_t rowlen)
			{
				return caview1d_ex<T, step_range>( prowbase,
						step_range::from_begin_dim(0, (index_t)rowlen, idxcore.base_dim0()) );
			}

			static row_view_type row_view(const index_core_t& idxcore, T *prowbase, size_t rowlen)
			{
				return aview1d_ex<T, step_range>( prowbase,
						step_range::from_begin_dim(0, (index_t)rowlen, idxcore.base_dim0()) );
			}
		};


		template<typename T, typename TOrd, class TRange> struct slice_range_helper2d;

		template<typename T>
		struct slice_range_helper2d<T, row_major_t, range>
		{
			typedef index_core2d<row_major_t> index_core_t;

			typedef caview1d<T> row_range_cview_type;
			typedef aview1d<T>  row_range_view_type;
			typedef caview1d_ex<T, step_range> column_range_cview_type;
			typedef aview1d_ex<T, step_range>  column_range_view_type;

			static row_range_cview_type row_range_cview(const index_core_t& idxcore, const T *prowbase, const range& rgn)
			{
				return caview1d<T>(prowbase + rgn.begin_index(), rgn.size());
			}

			static row_range_view_type row_range_view(const index_core_t& idxcore, T *prowbase, const range& rgn)
			{
				return aview1d<T>(prowbase + rgn.begin_index(), rgn.size());
			}

			static column_range_cview_type column_range_cview(const index_core_t& idxcore, const T *pcolbase, const range& rgn)
			{
				return caview1d_ex<T, step_range>(pcolbase, inject_step<range>::get(rgn, idxcore.base_dim1()) );
			}

			static column_range_view_type column_range_view(const index_core_t& idxcore, T *pcolbase, const range& rgn)
			{
				return aview1d_ex<T, step_range>(pcolbase, inject_step<range>::get(rgn, idxcore.base_dim1()) );
			}
		};

		template<typename T, class TRange>
		struct slice_range_helper2d<T, row_major_t, TRange>
		{
			typedef index_core2d<row_major_t> index_core_t;

			typedef caview1d_ex<T, TRange> row_range_cview_type;
			typedef aview1d_ex<T, TRange>  row_range_view_type;
			typedef caview1d_ex<T, typename inject_step<TRange>::result_type> column_range_cview_type;
			typedef aview1d_ex<T, typename inject_step<TRange>::result_type>  column_range_view_type;

			static row_range_cview_type row_range_cview(const index_core_t& idxcore, const T *prowbase, const TRange& rgn)
			{
				return caview1d_ex<T, TRange>(prowbase, rgn);
			}

			static row_range_view_type row_range_view(const index_core_t& idxcore, T *prowbase, const TRange& rgn)
			{
				return aview1d_ex<T, TRange>(prowbase, rgn);
			}

			static column_range_cview_type column_range_cview(const index_core_t& idxcore, const T *pcolbase, const TRange& rgn)
			{
				return caview1d_ex<T, typename inject_step<TRange>::result_type>(pcolbase,
						inject_step<TRange>::get(rgn, idxcore.base_dim1()) );
			}

			static column_range_view_type column_range_view(const index_core_t& idxcore, T *pcolbase, const TRange& rgn)
			{
				return aview1d_ex<T, typename inject_step<TRange>::result_type>(pcolbase,
						inject_step<TRange>::get(rgn, idxcore.base_dim1()) );
			}
		};


		template<typename T>
		struct slice_range_helper2d<T, column_major_t, range>
		{
			typedef index_core2d<column_major_t> index_core_t;

			typedef caview1d<T> column_range_cview_type;
			typedef aview1d<T>  column_range_view_type;
			typedef caview1d_ex<T, step_range> row_range_cview_type;
			typedef aview1d_ex<T, step_range>  row_range_view_type;

			static column_range_cview_type column_range_cview(const index_core_t& idxcore, const T *pcolbase, const range& rgn)
			{
				return caview1d<T>(pcolbase + rgn.begin_index(), rgn.size());
			}

			static column_range_view_type column_range_view(const index_core_t& idxcore, T *pcolbase, const range& rgn)
			{
				return aview1d<T>(pcolbase + rgn.begin_index(), rgn.size());
			}

			static row_range_cview_type row_range_cview(const index_core_t& idxcore, const T *prowbase, const range& rgn)
			{
				return caview1d_ex<T, step_range>(prowbase, inject_step<range>::get(rgn, idxcore.base_dim0()) );
			}

			static row_range_view_type row_range_view(const index_core_t& idxcore, T *prowbase, const range& rgn)
			{
				return aview1d_ex<T, step_range>(prowbase, inject_step<range>::get(rgn, idxcore.base_dim0()) );
			}
		};


		template<typename T, class TRange>
		struct slice_range_helper2d<T, column_major_t, TRange>
		{
			typedef index_core2d<column_major_t> index_core_t;

			typedef caview1d_ex<T, TRange> column_range_cview_type;
			typedef aview1d_ex<T, TRange>  column_range_view_type;
			typedef caview1d_ex<T, typename inject_step<TRange>::result_type> row_range_cview_type;
			typedef aview1d_ex<T, typename inject_step<TRange>::result_type>  row_range_view_type;

			static column_range_cview_type column_range_cview(const index_core_t& idxcore, const T *pcolbase, const TRange& rgn)
			{
				return caview1d_ex<T, TRange>(pcolbase, rgn);
			}

			static column_range_view_type column_range_view(const index_core_t& idxcore, T *pcolbase, const TRange& rgn)
			{
				return aview1d_ex<T, TRange>(pcolbase, rgn);
			}

			static row_range_cview_type row_range_cview(const index_core_t& idxcore, const T *prowbase, const TRange& rgn)
			{
				return caview1d_ex<T, typename inject_step<TRange>::result_type>(prowbase,
						inject_step<TRange>::get(rgn, idxcore.base_dim0()) );
			}

			static row_range_view_type row_range_view(const index_core_t& idxcore, T *prowbase, const TRange& rgn)
			{
				return aview1d_ex<T, typename inject_step<TRange>::result_type>(prowbase,
						inject_step<TRange>::get(rgn, idxcore.base_dim0()) );
			}
		};


		/**************************************************
		 *
		 *  memory operations
		 *
		 **************************************************/

		inline bool is_compact_layout2d(size_t len, index_t stride)
		{
			return (index_t)len == stride;
		}

		inline bool is_compact_layout2d(size_t len, index_t stride1, index_t stride2)
		{
			index_t l = (index_t)len;
			return l == stride1 && l == stride2;
		}


		template<typename T>
		inline void copy_elements_2d(size_t ns, size_t len, const T* src, const index_t src_stride, T *dst, index_t dst_stride)
		{
			if (is_compact_layout2d(len, src_stride, dst_stride))
			{
				copy_elements(src, dst, ns * len);
			}
			else
			{
				for (size_t i = 0; i < ns; ++i, src += src_stride, dst += dst_stride)
				{
					copy_elements(src, dst, len);
				}
			}
		}

		template<typename T>
		inline void set_elements_2d(size_t ns, size_t len, const T& v, T *dst, index_t dst_stride)
		{
			for (size_t i = 0; i < ns; ++i, dst += dst_stride)
			{
				for (size_t j = 0; j < len; ++j)
				{
					dst[j] = v;
				}
			}
		}

		template<typename T>
		inline void set_zeros_2d(size_t ns, size_t len, T *dst, index_t dst_stride)
		{
			if (is_compact_layout2d(len, dst_stride))
			{
				set_zeros_to_elements(dst, ns * len);
			}
			else
			{
				for (size_t i = 0; i < ns; ++i, dst += dst_stride)
				{
					set_zeros_to_elements(dst, len);
				}
			}
		}

		template<typename T>
		inline bool elements_equal_2d(size_t ns, size_t len,
				const T* lhs, const index_t lhs_stride, const T *rhs, index_t rhs_stride)
		{
			if (is_compact_layout2d(len, lhs_stride, rhs_stride))
			{
				return elements_equal(lhs, rhs, ns * len);
			}
			else
			{
				for (size_t i = 0; i < ns; ++i, lhs += lhs_stride, rhs += rhs_stride)
				{
					if (!elements_equal(lhs, rhs, len)) return false;
				}
				return true;
			}
		}

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
