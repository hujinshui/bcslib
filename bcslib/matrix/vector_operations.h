/**
 * @file vector_operations.h
 *
 * Operations on vector reader/accessors
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_VECTOR_OPERATIONS_H_
#define BCSLIB_VECTOR_OPERATIONS_H_

#include <bcslib/matrix/vector_accessors.h>

namespace bcs
{

	/********************************************
	 *
	 *  transfer
	 *
	 ********************************************/

	template<int N, class SVec, class DVec>
	struct transfer_vec
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(N >= 2, "The value of N must have N >= 2");
#endif

		BCS_ENSURE_INLINE
		static void run(const index_t, const SVec& in, DVec& out)
		{
			for (index_t i = 0; i < N; ++i)
			{
				out.set(i, in.get(i));
			}
		}
	};


	template<class SVec, class DVec>
	struct transfer_vec<DynamicDim, SVec, DVec>
	{
		BCS_ENSURE_INLINE
		static void run(const index_t len, const SVec& in, DVec& out)
		{
			for (index_t i = 0; i < len; ++i)
			{
				out.set(i, in.get(i));
			}
		}
	};


	template<class SVec, class DVec>
	struct transfer_vec<1, SVec, DVec>
	{
		BCS_ENSURE_INLINE
		static void run(const index_t, const SVec& in, DVec& out)
		{
			out.set(0, in.get(0));
		}
	};


	/********************************************
	 *
	 *  accumulation
	 *
	 ********************************************/

	template<class Reductor, int N, class Vec>
	struct accum_vec
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(N >= 3, "The value of N must have N >= 3");
		static_assert(is_reductor<Reductor, 1>::value, "Reductor must be a unary reductor");
#endif

		typedef typename Reductor::accum_type accum_t;

		BCS_ENSURE_INLINE
		static accum_t run(const Reductor& reduc, const index_t, const Vec& in)
		{
			accum_t a = reduc.init(in.get(0));
			for (index_t i = 1; i < N; ++i) a = reduc.add(a, in.get(i));
			return a;
		}
	};

	template<class Reductor, class Vec>
	struct accum_vec<Reductor, DynamicDim, Vec>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 1>::value, "Reductor must be a unary reductor");
#endif

		typedef typename Reductor::accum_type accum_t;

		BCS_ENSURE_INLINE
		static accum_t run(const Reductor& reduc, const index_t len, const Vec& in)
		{
			accum_t a = reduc.init(in.get(0));
			for (index_t i = 1; i < len; ++i) a = reduc.add(a, in.get(i));
			return a;
		}
	};


	template<class Reductor, class Vec>
	struct accum_vec<Reductor, 1, Vec>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 1>::value, "Reductor must be a unary reductor");
#endif

		typedef typename Reductor::accum_type accum_t;

		BCS_ENSURE_INLINE
		static accum_t run(const Reductor& reduc, const index_t len, const Vec& in)
		{
			return reduc.init(in.get(0));
		}
	};


	template<class Reductor, class Vec>
	struct accum_vec<Reductor, 2, Vec>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 1>::value, "Reductor must be a unary reductor");
#endif

		typedef typename Reductor::accum_type accum_t;

		BCS_ENSURE_INLINE
		static accum_t run(const Reductor& reduc, const index_t, const Vec& in)
		{
			accum_t a = reduc.init(in.get(0));
			return reduc.add(a, in.get(1));
		}
	};


	template<class Reductor, int N, class Vec1, class Vec2>
	struct accum_vec2
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 2>::value, "Reductor must be a binary reductor");
		static_assert(N >= 3, "The value of N must have N >= 3");
#endif

		typedef typename Reductor::accum_type accum_t;

		BCS_ENSURE_INLINE
		static accum_t run(const Reductor& reduc, const index_t,
				const Vec1& in1, const Vec2& in2)
		{
			accum_t a = reduc.init(in1.get(0), in2.get(0));
			for (index_t i = 1; i < N; ++i)
				a = reduc.add(a, in1.get(i), in2.get(i));
			return a;
		}
	};

	template<class Reductor, class Vec1, class Vec2>
	struct accum_vec2<Reductor, DynamicDim, Vec1, Vec2>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 2>::value, "Reductor must be a binary reductor");
#endif

		typedef typename Reductor::accum_type accum_t;

		BCS_ENSURE_INLINE
		static accum_t run(const Reductor& reduc, const index_t len,
				const Vec1& in1, const Vec2& in2)
		{
			accum_t a = reduc.init(in1.get(0), in2.get(0));
			for (index_t i = 1; i < len; ++i)
				a = reduc.add(a, in1.get(i), in2.get(i));
			return a;
		}
	};


	template<class Reductor, class Vec1, class Vec2>
	struct accum_vec2<Reductor, 1, Vec1, Vec2>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 2>::value, "Reductor must be a binary reductor");
#endif

		typedef typename Reductor::accum_type accum_t;

		BCS_ENSURE_INLINE
		static accum_t run(const Reductor& reduc, const index_t len,
				const Vec1& in1, const Vec2& in2)
		{
			return reduc.init(in1.get(0), in2.get(0));
		}
	};


	template<class Reductor, class Vec1, class Vec2>
	struct accum_vec2<Reductor, 2, Vec1, Vec2>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(is_reductor<Reductor, 2>::value, "Reductor must be a binary reductor");
#endif

		typedef typename Reductor::accum_type accum_t;

		BCS_ENSURE_INLINE
		static accum_t run(const Reductor& reduc, const index_t,
				const Vec1& in1, const Vec2& in2)
		{
			accum_t a = reduc.init(in1.get(0), in2.get(0));
			return reduc.add(a, in1.get(1), in2.get(1));
		}
	};

}


#endif /* VECTOR_OPERATIONS_H_ */
