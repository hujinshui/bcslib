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

	template<class LDerived, class RDerived, typename T>
	inline bool is_equal(
			const IConstAView1D<LDerived, T, continuous_form>& a,
			const IConstAView1D<RDerived, T, continuous_form>& b)
	{
		return is_same_shape(a, b) && elements_equal(a.pbase(), b.pbase(), a.size());
	}

	// import, export, fill

	template<class Derived, typename T>
	inline void import_from(IAView1D<Derived, T, regular_form>& a, const T *src)
	{
		Derived& ad = a.derived();
		index_t n = a.nelems();

		for (index_t i = 0; i < n; ++i) ad(i) = src[i];
	}

	template<class Derived, typename T>
	inline void import_from(IAView1D<Derived, T, continuous_form>& a, const T *src)
	{
		copy_elements(src, a.pbase(), a.size());
	}

	template<class Derived, typename T>
	inline void export_to(const IConstAView1D<Derived, T, regular_form>& a, T *dst)
	{
		const Derived& ad = a.derived();
		index_t n = a.nelems();

		for (index_t i = 0; i < n; ++i) dst[i] = ad(i);
	}

	template<class Derived, typename T>
	inline void export_to(const IConstAView1D<Derived, T, continuous_form>& a, T *dst)
	{
		copy_elements(a.pbase(), dst, a.size());
	}


	template<class Derived, typename T>
	inline void fill(IAView1D<Derived, T, regular_form>& a, const T& v)
	{
		Derived& ad = a.derived();
		index_t n = a.nelems();

		for (index_t i = 0; i < n; ++i) ad(i) = v;
	}


	// copy

	template<class LDerived, class RDerived, typename T>
	inline void copy(const IConstAView1D<LDerived, T, continuous_form>& src, IAView1D<RDerived, T, continuous_form>& dst)
	{
		const LDerived& sd = src.derived();
		RDerived& rd = dst.derived();

		check_arg(is_same_shape(sd, rd), "aview1d copy: the shapes of src and dst are inconsistent.");
		copy_elements(src.pbase(), dst.pbase(), src.size());
	}

	template<class LDerived, class RDerived, typename T>
	inline void copy(const IConstAView1D<LDerived, T, continuous_form>& src, IAView1D<RDerived, T, regular_form>& dst)
	{
		const LDerived& sd = src.derived();
		RDerived& rd = dst.derived();

		check_arg(is_same_shape(sd, rd), "aview1d copy: the shapes of src and dst are inconsistent.");
		import_from(dst, src.pbase());
	}

	template<class LDerived, class RDerived, typename T>
	inline void copy(const IConstAView1D<LDerived, T, regular_form>& src, IAView1D<RDerived, T, continuous_form>& dst)
	{
		const LDerived& sd = src.derived();
		RDerived& rd = dst.derived();

		check_arg(is_same_shape(sd, rd), "aview1d copy: the shapes of src and dst are inconsistent.");
		export_to(src, dst.pbase());
	}

	template<class LDerived, class RDerived, typename T>
	inline void copy(const IConstAView1D<LDerived, T, regular_form>& src, IAView1D<RDerived, T, regular_form>& dst)
	{
		const LDerived& sd = src.derived();
		RDerived& rd = dst.derived();

		check_arg(is_same_shape(sd, rd), "aview1d copy: the shapes of src and dst are inconsistent.");

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
