/**
 * @file vector_operations_internal.h
 *
 * The internal implementation of vector operations
 *
 * This is the key to achieve high performance
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_VECTOR_OPERATIONS_INTERNAL_H_
#define BCSLIB_VECTOR_OPERATIONS_INTERNAL_H_

#include <bcslib/matrix/vector_accessors.h>
#include <bcslib/matrix/vecproc_schemes.h>

namespace bcs { namespace detail {

	/********************************************
	 *
	 *  copy
	 *
	 ********************************************/

	template<class SVec, class DVec, class Scheme>
	struct vec_copy_helper;

	template<class SVec, class DVec>
	struct vec_copy_helper<SVec, DVec, vecscheme_by_scalars>
	{
		BCS_ENSURE_INLINE
		static void run(const SVec& svec, DVec& dvec, const vecscheme_by_scalars& sch)
		{
			index_t len = sch.length;
			for (index_t i = 0; i < len; ++i)
			{
				dvec.store_scalar(i, svec.load_scalar(i));
			}
		}
	};

	template<class SVec, class DVec, int N>
	struct vec_copy_helper<SVec, DVec, vecscheme_by_fixed_num_scalars<N> >
	{
		BCS_ENSURE_INLINE
		static void run(const SVec& svec, DVec& dvec, vecscheme_by_fixed_num_scalars<N>)
		{
			for (index_t i = 0; i < N; ++i)
			{
				dvec.store_scalar(i, svec.load_scalar(i));
			}
		}
	};

	template<class SVec, class DVec>
	struct vec_copy_helper<SVec, DVec, vecscheme_by_fixed_num_scalars<1> >
	{
		BCS_ENSURE_INLINE
		static void run(const SVec& svec, DVec& dvec, vecscheme_by_fixed_num_scalars<1>)
		{
			dvec.store_scalar(0, svec.load_scalar(0));
		}
	};

	template<class SVec, class DVec>
	struct vec_copy_helper<SVec, DVec, vecscheme_by_fixed_num_scalars<2> >
	{
		BCS_ENSURE_INLINE
		static void run(const SVec& svec, DVec& dvec, vecscheme_by_fixed_num_scalars<2>)
		{
			dvec.store_scalar(0, svec.load_scalar(0));
			dvec.store_scalar(1, svec.load_scalar(1));
		}
	};

	template<class SVec, class DVec>
	struct vec_copy_helper<SVec, DVec, vecscheme_by_fixed_num_scalars<3> >
	{
		BCS_ENSURE_INLINE
		static void run(const SVec& svec, DVec& dvec, vecscheme_by_fixed_num_scalars<3>)
		{
			dvec.store_scalar(0, svec.load_scalar(0));
			dvec.store_scalar(1, svec.load_scalar(1));
			dvec.store_scalar(2, svec.load_scalar(2));
		}
	};

	template<class SVec, class DVec>
	struct vec_copy_helper<SVec, DVec, vecscheme_by_fixed_num_scalars<4> >
	{
		BCS_ENSURE_INLINE
		static void run(const SVec& svec, DVec& dvec, vecscheme_by_fixed_num_scalars<4>)
		{
			dvec.store_scalar(0, svec.load_scalar(0));
			dvec.store_scalar(1, svec.load_scalar(1));
			dvec.store_scalar(2, svec.load_scalar(2));
			dvec.store_scalar(3, svec.load_scalar(3));
		}
	};



	/********************************************
	 *
	 *  set
	 *
	 ********************************************/

	template<typename T, class DVec, class Scheme>
	struct vec_set_helper;

	template<typename T, class DVec>
	struct vec_set_helper<T, DVec, vecscheme_by_scalars>
	{
		BCS_ENSURE_INLINE
		static void run(DVec& dvec, const T val, const vecscheme_by_scalars& sch)
		{
			index_t len = sch.length;
			for (index_t i = 0; i < len; ++i)
			{
				dvec.store_scalar(i, val);
			}
		}
	};

	template<typename T, class DVec, int N>
	struct vec_set_helper<T, DVec, vecscheme_by_fixed_num_scalars<N> >
	{
		BCS_ENSURE_INLINE
		static void run(DVec& dvec, const T val, vecscheme_by_fixed_num_scalars<N>)
		{
			for (index_t i = 0; i < N; ++i)
			{
				dvec.store_scalar(i, val);
			}
		}
	};


	template<typename T, class DVec>
	struct vec_set_helper<T, DVec, vecscheme_by_fixed_num_scalars<1> >
	{
		BCS_ENSURE_INLINE
		static void run(DVec& dvec, const T val, vecscheme_by_fixed_num_scalars<1>)
		{
			dvec.store_scalar(0, val);
		}
	};

	template<typename T, class DVec>
	struct vec_set_helper<T, DVec, vecscheme_by_fixed_num_scalars<2> >
	{
		BCS_ENSURE_INLINE
		static void run(DVec& dvec, const T val, vecscheme_by_fixed_num_scalars<2>)
		{
			dvec.store_scalar(0, val);
			dvec.store_scalar(1, val);
		}
	};

	template<typename T, class DVec>
	struct vec_set_helper<T, DVec, vecscheme_by_fixed_num_scalars<3> >
	{
		BCS_ENSURE_INLINE
		static void run(DVec& dvec, const T val, vecscheme_by_fixed_num_scalars<3>)
		{
			dvec.store_scalar(0, val);
			dvec.store_scalar(1, val);
			dvec.store_scalar(2, val);
		}
	};

	template<typename T, class DVec>
	struct vec_set_helper<T, DVec, vecscheme_by_fixed_num_scalars<4> >
	{
		BCS_ENSURE_INLINE
		static void run(DVec& dvec, const T val, vecscheme_by_fixed_num_scalars<4>)
		{
			dvec.store_scalar(0, val);
			dvec.store_scalar(1, val);
			dvec.store_scalar(2, val);
			dvec.store_scalar(3, val);
		}
	};

} }

#endif /* VECTOR_OPERATIONS_INTERNAL_H_ */
