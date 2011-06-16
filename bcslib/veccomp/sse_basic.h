/**
 * @file sse_basic.h
 *
 * Basic definitions for SSE/SSE2
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_SSE_BASIC_H
#define BCSLIB_SSE_BASIC_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>

#include <xmmintrin.h> 	// for SSE intrinsics
#include <emmintrin.h> 	// for SSE2 intrinsics

namespace bcs
{

	template<typename T> struct xmm_pack;

	// for double (require SSE2)

	template<>
	struct xmm_pack<double>
	{
		typedef double value_type;

		__m128d data;

		xmm_pack() : data(_mm_setzero_pd()) { }

		xmm_pack(__m128d p) : data(p) { }

		xmm_pack(double x) : data(_mm_set1_pd(x)) { }

		xmm_pack(double x0, double x1) : data(_mm_setr_pd(x0, x1)) { }

		static xmm_pack load(const double *a) { return _mm_load_pd(a); }

		void store(double *a) { _mm_store_pd(a, data); }
	};

	typedef xmm_pack<double> xmm_pd;


	inline xmm_pd operator + (xmm_pd v1, xmm_pd v2)
	{
		return _mm_add_pd(v1.data, v2.data);
	}

	inline xmm_pd operator - (xmm_pd v1, xmm_pd v2)
	{
		return _mm_sub_pd(v1.data, v2.data);
	}

	inline xmm_pd operator * (xmm_pd v1, xmm_pd v2)
	{
		return _mm_mul_pd(v1.data, v2.data);
	}

	inline xmm_pd operator / (xmm_pd v1, xmm_pd v2)
	{
		return _mm_div_pd(v1.data, v2.data);
	}

	inline xmm_pd min(xmm_pd v1, xmm_pd v2)
	{
		return _mm_min_pd(v1.data, v2.data);
	}

	inline xmm_pd max(xmm_pd v1, xmm_pd v2)
	{
		return _mm_max_pd(v1.data, v2.data);
	}

	inline xmm_pd sqr(xmm_pd v)
	{
		return _mm_mul_pd(v.data, v.data);
	}

	inline xmm_pd sqrt(xmm_pd v)
	{
		return _mm_sqrt_pd(v.data);
	}


	// for float (require SSE)

	template<>
	struct xmm_pack<float>
	{
		typedef float value_type;

		__m128 data;

		xmm_pack() : data(_mm_setzero_ps()) { }

		xmm_pack(__m128 p) : data(p) { }

		xmm_pack(float x) : data(_mm_set1_ps(x)) { }

		xmm_pack(float x0, float x1, float x2, float x3) : data(_mm_setr_ps(x0, x1, x2, x3)) { }

		static xmm_pack load(const float *a) { return _mm_load_ps(a); }

		void store(float *a) { _mm_store_ps(a, data); }
	};

	typedef xmm_pack<double> xmm_ps;

	inline xmm_ps operator + (xmm_ps v1, xmm_ps v2)
	{
		return _mm_add_ps(v1.data, v2.data);
	}

	inline xmm_ps operator - (xmm_ps v1, xmm_ps v2)
	{
		return _mm_sub_ps(v1.data, v2.data);
	}

	inline xmm_ps operator * (xmm_ps v1, xmm_ps v2)
	{
		return _mm_mul_ps(v1.data, v2.data);
	}

	inline xmm_ps operator / (xmm_ps v1, xmm_ps v2)
	{
		return _mm_div_ps(v1.data, v2.data);
	}

	inline xmm_ps min(xmm_ps v1, xmm_ps v2)
	{
		return _mm_min_ps(v1.data, v2.data);
	}

	inline xmm_ps max(xmm_ps v1, xmm_ps v2)
	{
		return _mm_max_ps(v1.data, v2.data);
	}

	inline xmm_ps sqr(xmm_ps v)
	{
		return _mm_mul_ps(v.data, v.data);
	}

	inline xmm_ps sqrt(xmm_ps v)
	{
		return _mm_sqrt_ps(v.data);
	}

	inline xmm_ps rcp(xmm_ps v)
	{
		return _mm_rcp_ps(v.data);
	}

	inline xmm_ps rsqrt(xmm_ps v)
	{
		return _mm_rsqrt_ps(v.data);
	}


}

#endif 
