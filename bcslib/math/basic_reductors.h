/**
 * @file basic_reductors.h
 *
 * The functors for reduction
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REDUCTORS_H_
#define BCSLIB_REDUCTORS_H_

#include <bcslib/core/basic_defs.h>
#include <bcslib/utils/arg_check.h>

#include <bcslib/math/scalar_math.h>
#include <algorithm>

namespace bcs
{

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x) const { return x; }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x) const { return a + x; }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return a + a2; }

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
		T empty_result() const { throw invalid_operation("Attempted to get the mean of an empty array."); }

		BCS_ENSURE_INLINE
		T init(const T& x) const { return x; }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x) const { return a + x; }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return a + a2; }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t n) const { return a / static_cast<T>(n); }
	};


	template<typename T>
	struct min_reductor
	{
		typedef T argument_type;
		typedef T accum_type;
		typedef T result_type;

		BCS_ENSURE_INLINE
		T empty_result() const { throw invalid_operation("Attempted to get the minimum of an empty array."); }

		BCS_ENSURE_INLINE
		T init(const T& x) const { return x; }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x) const { return std::min(a, x); }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return std::min(a, a2); }

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
		T empty_result() const { throw invalid_operation("Attempted to get the maximum of an empty array."); }

		BCS_ENSURE_INLINE
		T init(const T& x) const { return x; }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x) const { return std::max(a, x); }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return std::max(a, a2); }

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x) const { return math::abs(x); }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x) const { return a + math::abs(x); }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return a + a2; }

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x) const { return math::sqr(x); }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x) const { return a + math::sqr(x); }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return a + a2; }

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x) const { return math::sqr(x); }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x) const { return a + math::sqr(x); }

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x) const { return math::abs(x); }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x) const { return std::max(a, math::abs(x)); }

		BCS_ENSURE_INLINE
		T combine (const T& a, const T& a2) const { return std::max(a, a2); }

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x, const T& y) const { return x * y; }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x, const T& y) const { return a + x * y; }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return a + a2; }

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x, const T& y) const { return math::abs(x - y); }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x, const T& y) const { return a + math::abs(x - y); }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return a + a2; }

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x, const T& y) const { return math::sqr(x - y); }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x, const T& y) const { return a + math::sqr(x - y); }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return a + a2; }

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x, const T& y) const { return math::sqr(x - y); }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x, const T& y) const { return a + math::sqr(x - y); }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return a + a2; }

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
		T empty_result() const { return 0; }

		BCS_ENSURE_INLINE
		T init(const T& x, const T& y) const { return math::abs(x - y); }

		BCS_ENSURE_INLINE
		T add(const T& a, const T& x, const T& y) const { return std::max(a, math::abs(x - y)); }

		BCS_ENSURE_INLINE
		T combine(const T& a, const T& a2) const { return std::max(a, a2); }

		BCS_ENSURE_INLINE
		T get(const T& a, const index_t) const { return a; }
	};


	// Declaration

	BCS_DECLARE_REDUCTOR( sum_reductor, 1)
	BCS_DECLARE_REDUCTOR( mean_reductor, 1 )
	BCS_DECLARE_REDUCTOR( min_reductor, 1 )
	BCS_DECLARE_REDUCTOR( max_reductor, 1 )

	BCS_DECLARE_REDUCTOR( L1norm_reductor, 1 )
	BCS_DECLARE_REDUCTOR( sqL2norm_reductor, 1 )
	BCS_DECLARE_REDUCTOR( L2norm_reductor, 1 )
	BCS_DECLARE_REDUCTOR( Linfnorm_reductor, 1 )

	BCS_DECLARE_REDUCTOR( dot_reductor, 2 )

	BCS_DECLARE_REDUCTOR( L1diffnorm_reductor, 2 )
	BCS_DECLARE_REDUCTOR( sqL2diffnorm_reductor, 2 )
	BCS_DECLARE_REDUCTOR( L2diffnorm_reductor, 2 )
	BCS_DECLARE_REDUCTOR( Linfdiffnorm_reductor, 2 )

}

#endif /* REDUCTION_FUNCTORS_H_ */
