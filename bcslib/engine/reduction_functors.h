/**
 * @file reduction_functors.h
 *
 * The functors for reduction
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REDUCTION_FUNCTORS_H_
#define BCSLIB_REDUCTION_FUNCTORS_H_

#include <bcslib/core/basic_defs.h>
#include <bcslib/utils/arg_check.h>

namespace bcs
{
	/*********************************************************************
	 *
	 *  Reduction functor concept
	 *  -------------------------
	 *
	 *  Typedef:
	 *
	 *  argument_type
	 *  accum_type
	 *  result_type
	 *
	 *
	 *  Methods:
	 *
	 *  fun(); 	--> result on an empty array (result_type)
	 *
	 *  fun(a); --> initialize the accumulator (accum_type)
	 *
	 *  fun(a, x); --> merge a new element to the accumulator (accum_type)
	 *
	 *  fun.combine(a1, a2); --> merge two accumulators (accum_type)
	 *
	 *  fun.get(a, n); --> get the final result (result_type)
	 *
	 *
	 *********************************************************************/



	/********************************************
	 *
	 *  Basic reductors
	 *
	 ********************************************/

	template<typename T>
	struct sum_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x) const { return x; }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x) const { return a + x; }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};


	template<typename T>
	struct mean_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { throw invalid_operation("Attempted to get the mean of an empty array."); }

		BCS_ENSURE_INLINE
		T operator() (const T& x) const { return x; }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x) const { return a + x; }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t n) const { return a / n; }
	};


	template<typename T>
	struct min_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { throw invalid_operation("Attempted to get the minimum of an empty array."); }

		BCS_ENSURE_INLINE
		T operator() (const T& x) const { return x; }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x) const { return bcs::min(a, x); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return bcs::min(a, a2); }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};


	template<typename T>
	struct max_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { throw invalid_operation("Attempted to get the maximum of an empty array."); }

		BCS_ENSURE_INLINE
		T operator() (const T& x) const { return x; }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x) const { return bcs::max(a, x); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return bcs::max(a, a2); }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};


	/********************************************
	 *
	 *  Norm reductors
	 *
	 ********************************************/

	template<typename T>
	struct L1norm_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x) const { return math::abs(x); }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x) const { return a + math::abs(x); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};

	template<typename T>
	struct sqL2norm_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x) const { return math::sqr(x); }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x) const { return a + math::sqr(x); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};


	template<typename T>
	struct L2norm_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x) const { return math::sqr(x); }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x) const { return a + math::sqr(x); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return math::sqrt(a); }
	};


	template<typename T>
	struct Linfnorm_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x) const { return math::abs(x); }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x) const { return bcs::max(a, math::abs(x)); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return bcs::max(a, a2); }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};


	/********************************************
	 *
	 *  Binary reductors
	 *
	 ********************************************/

	template<typename T>
	struct dot_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x, const T& y) const { return x * y; }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x, const T& y) const { return a + x * y; }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};


	template<typename T>
	struct L1diffnorm_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x, const T& y) const { return math::abs(x - y); }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x, const T& y) const { return a + math::abs(x - y); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};

	template<typename T>
	struct sqL2diffnorm_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x, const T& y) const { return math::sqr(x - y); }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x, const T& y) const { return a + math::sqr(x - y); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};


	template<typename T>
	struct L2diffnorm_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x, const T& y) const { return math::sqr(x - y); }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x, const T& y) const { return a + math::sqr(x - y); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return math::sqrt(a); }
	};


	template<typename T>
	struct Linfdiffnorm_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T operator() () const { return 0; }

		BCS_ENSURE_INLINE
		T operator() (const T& x, const T& y) const { return math::abs(x - y); }

		BCS_ENSURE_INLINE
		T operator() (const T& a, const T& x, const T& y) const { return bcs::max(a, math::abs(x - y)); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return bcs::max(a, a2); }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};


	// Declaration

	DECLARE_UNARY_REDUCTION_FUNCTOR( sum_reductor )
	DECLARE_UNARY_REDUCTION_FUNCTOR( mean_reductor )
	DECLARE_UNARY_REDUCTION_FUNCTOR( min_reductor )
	DECLARE_UNARY_REDUCTION_FUNCTOR( max_reductor )

	DECLARE_UNARY_REDUCTION_FUNCTOR( L1norm_reductor )
	DECLARE_UNARY_REDUCTION_FUNCTOR( sqL2norm_reductor )
	DECLARE_UNARY_REDUCTION_FUNCTOR( L2norm_reductor )
	DECLARE_UNARY_REDUCTION_FUNCTOR( Linfnorm_reductor )

	DECLARE_BINARY_REDUCTION_FUNCTOR( dot_reductor )

	DECLARE_BINARY_REDUCTION_FUNCTOR( L1diffnorm_reductor )
	DECLARE_BINARY_REDUCTION_FUNCTOR( sqL2diffnorm_reductor )
	DECLARE_BINARY_REDUCTION_FUNCTOR( L2diffnorm_reductor )
	DECLARE_BINARY_REDUCTION_FUNCTOR( Linfdiffnorm_reductor )

}

#endif /* REDUCTION_FUNCTORS_H_ */
