/**
 * @file aview2d_ops.h
 *
 * 2D array view based operations
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_AVIEW2D_OPS_H_
#define BCSLIB_AVIEW2D_OPS_H_

#include <bcslib/array/aview2d_base.h>
#include <bcslib/base/basic_mem.h>

namespace bcs
{
	// comparison

	template<class LDerived, class RDerived>
	inline bool is_equal(const dense_caview2d_base<LDerived>& lhs, const dense_caview2d_base<RDerived>& rhs)
	{
		return is_same_shape(lhs, rhs) && elements_equal(lhs.pbase(), rhs.pbase(), lhs.size());
	}

	// import, export, fill

	namespace _detail
	{
		template<class Derived>
		inline void import_from(aview2d_base<Derived>& a, typename Derived::const_pointer src, row_major_t)
		{
			Derived& ad = a.derived();
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t i = 0; i < m; ++i)
			{
				for (index_t j = 0; j < n; ++j)
				{
					ad(i, j) = *(src++);
				}
			}
		}

		template<class Derived>
		inline void import_from(aview2d_base<Derived>& a, typename Derived::const_pointer src, column_major_t)
		{
			Derived& ad = a.derived();
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					ad(i, j) = *(src++);
				}
			}
		}

		template<class Derived>
		inline void export_to(const caview2d_base<Derived>& a, typename Derived::pointer dst, row_major_t)
		{
			const Derived& ad = a.derived();
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t i = 0; i < m; ++i)
			{
				for (index_t j = 0; j < n; ++j)
				{
					*(dst++) = ad(i, j);
				}
			}
		}

		template<class Derived>
		inline void export_to(const caview2d_base<Derived>& a, typename Derived::pointer dst, column_major_t)
		{
			const Derived& ad = a.derived();
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					*(dst++) = ad(i, j);
				}
			}
		}

		template<class Derived>
		inline void fill(aview2d_base<Derived>& a, const typename Derived::value_type& v, row_major_t)
		{
			Derived& ad = a.derived();
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t i = 0; i < m; ++i)
			{
				for (index_t j = 0; j < n; ++j)
				{
					ad(i, j) = v;
				}
			}
		}

		template<class Derived>
		inline void fill(aview2d_base<Derived>& a, const typename Derived::value_type& v, column_major_t)
		{
			Derived& ad = a.derived();
			index_t m = a.nrows();
			index_t n = a.ncolumns();

			for (index_t j = 0; j < n; ++j)
			{
				for (index_t i = 0; i < m; ++i)
				{
					ad(i, j) = v;
				}
			}
		}

		template<class LDerived, class RDerived>
		inline void copy(const caview2d_base<LDerived>& src, aview2d_base<RDerived>& dst, row_major_t)
		{
			const LDerived& srcd = src.derived();
			RDerived& dstd = dst.derived();

			index_t d0 = src.dim0();
			index_t d1 = src.dim1();

			for (index_t i = 0; i < d0; ++i)
			{
				for (index_t j = 0; j < d1; ++j)
				{
					dstd(i, j) = srcd(i, j);
				}
			}
		}

		template<class LDerived, class RDerived>
		inline void copy(const caview2d_base<LDerived>& src, aview2d_base<RDerived>& dst, column_major_t)
		{
			const LDerived& srcd = src.derived();
			RDerived& dstd = dst.derived();

			index_t d0 = src.dim0();
			index_t d1 = src.dim1();

			for (index_t j = 0; j < d1; ++j)
			{
				for (index_t i = 0; i < d0; ++i)
				{
					dstd(i, j) = srcd(i, j);
				}
			}
		}

	}

	template<class Derived>
	inline void import_from(aview2d_base<Derived>& a, typename Derived::const_pointer src)
	{
		_detail::import_from(a, src, typename Derived::layout_order());
	}

	template<class Derived>
	inline void import_from(dense_aview2d_base<Derived>& a, typename Derived::const_pointer src)
	{
		copy_elements(src, a.pbase(), a.size());
	}

	template<class Derived>
	inline void export_to(const caview2d_base<Derived>& a, typename Derived::pointer dst)
	{
		_detail::export_to(a, dst, typename Derived::layout_order());
	}

	template<class Derived>
	inline void export_to(const dense_caview2d_base<Derived>& a, typename Derived::pointer dst)
	{
		copy_elements(a.pbase(), dst, a.size());
	}

	template<class Derived>
	inline void fill(aview2d_base<Derived>& a, const typename Derived::value_type& v)
	{
		_detail::fill(a, v, typename Derived::layout_order());
	}


	// copy

	template<class LDerived, class RDerived>
	inline void copy(const dense_caview2d_base<LDerived>& src, dense_aview2d_base<RDerived>& dst)
	{
		BCS_STATIC_ASSERT( (has_same_layout_order<LDerived, RDerived>::value) );

		check_arg(is_same_shape(src, dst), "aview2d copy: the shapes of src and dst are inconsistent.");
		copy_elements(src.pbase(), dst.pbase(), src.size());
	}

	template<class LDerived, class RDerived>
	inline void copy(const dense_caview2d_base<LDerived>& src, aview2d_base<RDerived>& dst)
	{
		BCS_STATIC_ASSERT( (has_same_layout_order<LDerived, RDerived>::value) );

		check_arg(is_same_shape(src, dst), "aview2d copy: the shapes of src and dst are inconsistent.");
		import_from(dst, src.pbase());
	}

	template<class LDerived, class RDerived>
	inline void copy(const caview2d_base<LDerived>& src, dense_aview2d_base<RDerived>& dst)
	{
		BCS_STATIC_ASSERT( (has_same_layout_order<LDerived, RDerived>::value) );

		check_arg(is_same_shape(src, dst), "aview2d copy: the shapes of src and dst are inconsistent.");
		export_to(src, dst.pbase());
	}

	template<class LDerived, class RDerived>
	inline void copy(const caview2d_base<LDerived>& src, aview2d_base<RDerived>& dst)
	{
		BCS_STATIC_ASSERT( (has_same_layout_order<LDerived, RDerived>::value) );

		check_arg(is_same_shape(src, dst), "aview2d copy: the shapes of src and dst are inconsistent.");
		_detail::copy(src, dst, typename LDerived::layout_order());
	}


}

#endif
