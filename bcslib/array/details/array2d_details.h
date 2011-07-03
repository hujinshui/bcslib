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

namespace bcs
{
	namespace _detail
	{

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
				return caview1d_ex<T>( pcolbase, step_range::from_begin_dim(0, collen, idxcore.base_dim1()) );
			}

			static column_view_type column_view(const index_core_t& idxcore, T *pcolbase, size_t collen)
			{
				return aview1d_ex<T>( pcolbase, step_range::from_begin_dim(0, collen, idxcore.base_dim1()) );
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

			static row_cview_type column_cview(const index_core_t& idxcore, const T *pcolbase, size_t collen)
			{
				return caview1d<T>(pcolbase, collen);
			}

			static row_view_type column_view(const index_core_t& idxcore, T *pcolbase, size_t collen)
			{
				return aview1d<T>(pcolbase, collen);
			}

			static column_cview_type row_cview(const index_core_t& idxcore, const T *prowbase, size_t rowlen)
			{
				return caview1d_ex<T>( prowbase, step_range::from_begin_dim(0, rowlen, idxcore.base_dim0()) );
			}

			static column_view_type row_view(const index_core_t& idxcore, T *prowbase, size_t rowlen)
			{
				return aview1d_ex<T>( prowbase, step_range::from_begin_dim(0, rowlen, idxcore.base_dim0()) );
			}
		};


		template<typename T, typename TOrd, class TRange> struct slice_range_helper2d;

		template<typename T>
		struct slice_range_helper2d<T, row_major_t, range>
		{
			typedef index_core2d<row_major_t> index_core_t;

			typedef caview1d_ex<T> row_range_cview_type;
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
				return caview1d_ex<T, step_range>(pcolbase, inject_step<range>::get(rgn, idxcore.base_dim1()) );
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
				return caview1d_ex<T, typename inject_step<TRange>::result_type>(pcolbase,
						inject_step<TRange>::get(rgn, idxcore.base_dim1()) );
			}
		};


		template<typename T>
		struct slice_range_helper2d<T, column_major_t, range>
		{
			typedef index_core2d<column_major_t> index_core_t;

			typedef caview1d_ex<T> column_range_cview_type;
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

			static row_range_view_type column_range_view(const index_core_t& idxcore, T *prowbase, const range& rgn)
			{
				return caview1d_ex<T, step_range>(prowbase, inject_step<range>::get(rgn, idxcore.base_dim0()) );
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

			static column_range_cview_type row_range_cview(const index_core_t& idxcore, const T *pcolbase, const TRange& rgn)
			{
				return caview1d_ex<T, Range>(pcolbase, rgn);
			}

			static column_range_view_type row_range_view(const index_core_t& idxcore, T *pcolbase, const TRange& rgn)
			{
				return aview1d_ex<T, Range>(pcolbase, rgn);
			}

			static row_range_cview_type column_range_cview(const index_core_t& idxcore, const T *prowbase, const TRange& rgn)
			{
				return caview1d_ex<T, typename inject_step<TRange>::result_type>(prowbase,
						inject_step<TRange>::get(rgn, idxcore.base_dim0()) );
			}

			static row_range_view_type column_range_view(const index_core_t& idxcore, T *prowbase, const TRange& rgn)
			{
				return caview1d_ex<T, typename inject_step<TRange>::result_type>(prowbase,
						inject_step<TRange>::get(rgn, idxcore.base_dim0()) );
			}
		};

	}
}

#endif 
