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
#include <bcslib/base/mem_op.h>

namespace bcs
{
	// Comparison

	template<class LDerived, class RDerived, typename T>
	inline bool is_equal(
			const IConstContinuousAView2D<LDerived, T>& lhs,
			const IConstContinuousAView2D<RDerived, T>& rhs)
	{
		return is_same_shape(lhs, rhs) && mem<T>::equal(lhs.size(), lhs.pbase(), rhs.pbase());
	}


	// Import

	template<class Derived, typename T>
	inline void import_from(IRegularAView2D<Derived, T>& a, const T *src)
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

	template<class Derived, typename T>
	inline void import_from(IContinuousAView2D<Derived, T>& a, const T *src)
	{
		mem<T>::copy(a.size(), src, a.pbase());
	}


	// Export

	template<class Derived, typename T>
	inline void export_to(const IConstRegularAView2D<Derived, T>& a, T *dst)
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

	template<class Derived, typename T>
	inline void export_to(const IConstContinuousAView2D<Derived, T>& a, T *dst)
	{
		mem<T>::copy(a.size(), a.pbase(), dst);
	}


	// Fill

	template<class Derived, typename T>
	inline void fill(IRegularAView2D<Derived, T>& a, const T& v)
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

	template<class Derived, typename T>
	inline void fill(IContinuousAView2D<Derived, T>& a, const T& v)
	{
		index_t N = a.nelems();
		T *p = a.pbase();

		for (index_t i = 0; i < N; ++i)
		{
			p[i] = v;
		}
	}


	// copy


	template<class LDerived, class RDerived, typename T>
	inline void copy(const IConstRegularAView2D<LDerived, T>& src, IRegularAView2D<RDerived, T>& dst)
	{
		const LDerived& srcd = src.derived();
		RDerived& dstd = dst.derived();

		check_arg(is_same_shape(srcd, dstd), "aview2d copy: the shapes of src and dst are inconsistent.");

		index_t d0 = src.nrows();
		index_t d1 = src.ncolumns();

		for (index_t j = 0; j < d1; ++j)
		{
			for (index_t i = 0; i < d0; ++i)
			{
				dstd(i, j) = srcd(i, j);
			}
		}
	}


	template<class LDerived, class RDerived, typename T>
	inline void copy(const IConstContinuousAView2D<LDerived, T>& src, IContinuousAView2D<RDerived, T>& dst)
	{
		const LDerived& srcd = src.derived();
		RDerived& dstd = dst.derived();

		check_arg(is_same_shape(srcd, dstd), "aview2d copy: the shapes of src and dst are inconsistent.");
		mem<T>::copy(srcd.size(), srcd.pbase(), dstd.pbase());
	}

	template<class LDerived, class RDerived, typename T>
	inline void copy(const IConstContinuousAView2D<LDerived, T>& src, IRegularAView2D<RDerived, T>& dst)
	{
		const LDerived& srcd = src.derived();
		RDerived& dstd = dst.derived();

		check_arg(is_same_shape(srcd, dstd), "aview2d copy: the shapes of src and dst are inconsistent.");
		import_from(dstd, src.pbase());
	}

	template<class LDerived, class RDerived, typename T>
	inline void copy(const IConstRegularAView2D<LDerived, T>& src, IContinuousAView2D<RDerived, T>& dst)
	{
		const LDerived& srcd = src.derived();
		RDerived& dstd = dst.derived();

		check_arg(is_same_shape(srcd, dstd), "aview2d copy: the shapes of src and dst are inconsistent.");
		export_to(srcd, dstd.pbase());
	}

}

#endif
