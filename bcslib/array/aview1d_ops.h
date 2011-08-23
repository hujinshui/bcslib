/**
 * @file aview1d_ops.h
 *
 * 1D Array view based operations
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_AVIEW1D_OPS_H_
#define BCSLIB_AVIEW1D_OPS_H_

#include <bcslib/array/aview1d_base.h>
#include <bcslib/base/basic_mem.h>

namespace bcs
{
	// comparison

	template<class LDerived, class RDerived>
	inline bool is_equal(const continuous_caview1d_base<LDerived>& lhs, const continuous_caview1d_base<RDerived>& rhs)
	{
		return is_same_shape(lhs, rhs) && elements_equal(lhs.pbase(), rhs.pbase(), lhs.size());
	}

	// import, export, fill

	template<class Derived>
	inline void import_from(dense_aview1d_base<Derived>& a, typename Derived::const_pointer src)
	{
		Derived& ad = a.derived();
		index_t n = a.nelems();

		for (index_t i = 0; i < n; ++i) ad(i) = src[i];
	}

	template<class Derived>
	inline void import_from(continuous_aview1d_base<Derived>& a, typename Derived::const_pointer src)
	{
		copy_elements(src, a.pbase(), a.size());
	}

	template<class Derived>
	inline void export_to(const dense_caview1d_base<Derived>& a, typename Derived::pointer dst)
	{
		const Derived& ad = a.derived();
		index_t n = a.nelems();

		for (index_t i = 0; i < n; ++i) dst[i] = ad(i);
	}

	template<class Derived>
	inline void export_to(const continuous_caview1d_base<Derived>& a, typename Derived::pointer dst)
	{
		copy_elements(a.pbase(), dst, a.size());
	}


	template<class Derived>
	inline void fill(dense_aview1d_base<Derived>& a, const typename Derived::value_type& v)
	{
		Derived& ad = a.derived();
		index_t n = a.nelems();

		for (index_t i = 0; i < n; ++i) ad(i) = v;
	}


	// copy

	template<class LDerived, class RDerived>
	inline void copy(const continuous_caview1d_base<LDerived>& src, continuous_aview1d_base<RDerived>& dst)
	{
		check_arg(is_same_shape(src, dst), "aview1d copy: the shapes of src and dst are inconsistent.");
		copy_elements(src.pbase(), dst.pbase(), src.size());
	}

	template<class LDerived, class RDerived>
	inline void copy(const continuous_caview1d_base<LDerived>& src, dense_aview1d_base<RDerived>& dst)
	{
		check_arg(is_same_shape(src, dst), "aview1d copy: the shapes of src and dst are inconsistent.");
		import_from(dst, src.pbase());
	}

	template<class LDerived, class RDerived>
	inline void copy(const dense_caview1d_base<LDerived>& src, continuous_aview1d_base<RDerived>& dst)
	{
		check_arg(is_same_shape(src, dst), "aview1d copy: the shapes of src and dst are inconsistent.");
		export_to(src, dst.pbase());
	}

	template<class LDerived, class RDerived>
	inline void copy(const dense_caview1d_base<LDerived>& src, dense_aview1d_base<RDerived>& dst)
	{
		check_arg(is_same_shape(src, dst), "aview1d copy: the shapes of src and dst are inconsistent.");

		const LDerived& srcd = src.derived();
		RDerived& dstd = dst.derived();

		index_t d0 = src.dim0();
		for (index_t i = 0; i < d0; ++i)
		{
			dstd(i) = srcd(i);
		}
	}

}

#endif
