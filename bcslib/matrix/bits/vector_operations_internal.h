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

	template<class Scheme> struct vec_copy_helper;

	template<>
	struct vec_copy_helper<vecscheme_by_scalars>
	{
		template<class SVec, class DVec>
		BCS_ENSURE_INLINE static void run(const vecscheme_by_scalars& sch, const SVec& in, DVec& out)
		{
			index_t len = sch.length;
			for (index_t i = 0; i < len; ++i)
				out.store_scalar(i, in.load_scalar(i));
		}
	};


	template<int N>
	struct vec_copy_helper<vecscheme_by_fixed_num_scalars<N> >
	{
		template<class SVec, class DVec>
		BCS_ENSURE_INLINE static void run(vecscheme_by_fixed_num_scalars<N>, const SVec& in, DVec& out)
		{
			for (index_t i = 0; i < N; ++i)
				out.store_scalar(i, in.load_scalar(i));
		}
	};

	template<>
	struct vec_copy_helper<vecscheme_by_fixed_num_scalars<1> >
	{
		template<class SVec, class DVec>
		BCS_ENSURE_INLINE static void run(vecscheme_by_fixed_num_scalars<1>, const SVec& in, DVec& out)
		{
			out.store_scalar(0, in.load_scalar(0));
		}
	};

	template<>
	struct vec_copy_helper<vecscheme_by_fixed_num_scalars<2> >
	{
		template<class SVec, class DVec>
		BCS_ENSURE_INLINE static void run(vecscheme_by_fixed_num_scalars<2>, const SVec& in, DVec& out)
		{
			out.store_scalar(0, in.load_scalar(0));
			out.store_scalar(1, in.load_scalar(1));
		}
	};

	template<>
	struct vec_copy_helper<vecscheme_by_fixed_num_scalars<3> >
	{
		template<class SVec, class DVec>
		BCS_ENSURE_INLINE static void run(vecscheme_by_fixed_num_scalars<3>, const SVec& in, DVec& out)
		{
			out.store_scalar(0, in.load_scalar(0));
			out.store_scalar(1, in.load_scalar(1));
			out.store_scalar(2, in.load_scalar(2));
		}
	};

	template<>
	struct vec_copy_helper<vecscheme_by_fixed_num_scalars<4> >
	{
		template<class SVec, class DVec>
		BCS_ENSURE_INLINE static void run(vecscheme_by_fixed_num_scalars<4>, const SVec& in, DVec& out)
		{
			out.store_scalar(0, in.load_scalar(0));
			out.store_scalar(1, in.load_scalar(1));
			out.store_scalar(2, in.load_scalar(2));
			out.store_scalar(3, in.load_scalar(3));
		}
	};


	/********************************************
	 *
	 *  accum
	 *
	 *  pre-condition: len > 0
	 *
	 ********************************************/

	template<class Scheme> struct vec_accum_helper;

	template<>
	struct vec_accum_helper<vecscheme_by_scalars>
	{
		template<class Reductor, class Vec>
		inline static typename Reductor::accum_type
		run(const vecscheme_by_scalars& sch, Reductor reduc, const Vec& in)
		{
			typename Reductor::accum_type a = reduc(in.load_scalar(0));

			index_t len = sch.length;
			for (index_t i = 1; i < len; ++i) a = reduc(a, in.load_scalar(i));

			return a;
		}

		template<class Reductor, class LVec, class RVec>
		inline static typename Reductor::accum_type
		run(const vecscheme_by_scalars& sch, Reductor reduc, const LVec& left_in, const RVec& right_in)
		{
			typename Reductor::accum_type a = reduc(left_in.load_scalar(0), right_in.load_scalar(0));

			index_t len = sch.length;
			for (index_t i = 1; i < len; ++i) a = reduc(a, left_in.load_scalar(i), right_in.load_scalar(i));

			return a;
		}
	};


	template<int N>
	struct vec_accum_helper<vecscheme_by_fixed_num_scalars<N> >
	{
		template<class Reductor, class Vec>
		inline static typename Reductor::accum_type
		run(const vecscheme_by_scalars& sch, Reductor reduc, const Vec& in)
		{
			typename Reductor::accum_type a = reduc(in.load_scalar(0));
			for (index_t i = 1; i < N; ++i) a = reduc(a, in.load_scalar(i));
			return a;
		}

		template<class Reductor, class LVec, class RVec>
		inline static typename Reductor::accum_type
		run(const vecscheme_by_scalars& sch, Reductor reduc, const LVec& left_in, const RVec& right_in)
		{
			typename Reductor::accum_type a = reduc(left_in.load_scalar(0), right_in.load_scalar(0));
			for (index_t i = 1; i < N; ++i) a = reduc(a, left_in.load_scalar(i), right_in.load_scalar(i));
			return a;
		}
	};


	template<int N>
	struct vec_accum_helper<vecscheme_by_fixed_num_scalars<1> >
	{
		template<class Reductor, class Vec>
		inline static typename Reductor::accum_type
		run(const vecscheme_by_scalars& sch, Reductor reduc, const Vec& in)
		{
			return reduc(in.load_scalar(0));
		}

		template<class Reductor, class LVec, class RVec>
		inline static typename Reductor::accum_type
		run(const vecscheme_by_scalars& sch, Reductor reduc, const LVec& left_in, const RVec& right_in)
		{
			return reduc(left_in.load_scalar(0), right_in.load_scalar(0));
		}
	};


} }

#endif /* VECTOR_OPERATIONS_INTERNAL_H_ */


