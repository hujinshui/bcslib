/**
 * @file array_calc.h
 *
 * Vectorized calculation on arrays
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_CALC_H
#define BCSLIB_ARRAY_CALC_H

#include <bcslib/array/generic_array_calc.h>

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/base/arg_check.h>
#include <type_traits>

namespace bcs
{

	/******************************************************
	 *
	 *  Arithmetic operators/functions
	 *
	 ******************************************************/

	// addition

	// 1D

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> operator + (const aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return add_arr_arr(lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator + (const aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return add_arr_sca(lhs, rhs);
	}

	template<typename T, class RIndexer>
	inline array1d<T> operator + (const T& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return add_arr_sca(rhs, lhs);
	}

	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>& operator += (aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		add_arr_arr_inplace(lhs, rhs);
		return lhs;
	}

	template<typename T, class LIndexer>
	inline aview1d<T, LIndexer>& operator += (aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		add_arr_sca_inplace(lhs, rhs);
		return lhs;
	}

	// 2D

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator + (
			const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return add_arr_arr(lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> operator + (const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return add_arr_sca(lhs, rhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator + (const T& lhs, const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return add_arr_sca(rhs, lhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator += (
			aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		add_arr_arr_inplace(lhs, rhs);
		return lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator += (aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		add_arr_sca_inplace(lhs, rhs);
		return lhs;
	}



	// subtraction

	// 1D

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> operator - (const aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return sub_arr_arr(lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator - (const aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return sub_arr_sca(lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator - (const T& lhs, const aview1d<T, LIndexer>& rhs)
	{
		return sub_sca_arr(lhs, rhs);
	}

	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>& operator -= (aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		sub_arr_arr_inplace(lhs, rhs);
		return lhs;
	}

	template<typename T, class LIndexer>
	inline aview1d<T, LIndexer>& operator -= (aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		sub_arr_sca_inplace(lhs, rhs);
		return lhs;
	}

	// 2D

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator - (
			const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return sub_arr_arr(lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> operator - (const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return sub_arr_sca(lhs, rhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator - (const T& lhs, const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return sub_sca_arr(lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator -= (
			aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		sub_arr_arr_inplace(lhs, rhs);
		return lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator -= (aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		sub_arr_sca_inplace(lhs, rhs);
		return lhs;
	}



	// multiplication

	// 1D

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> operator * (const aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return mul_arr_arr(lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator * (const aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return mul_arr_sca(lhs, rhs);
	}

	template<typename T, class RIndexer>
	inline array1d<T> operator * (const T& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return mul_arr_sca(rhs, lhs);
	}

	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>& operator *= (aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		mul_arr_arr_inplace(lhs, rhs);
		return lhs;
	}

	template<typename T, class LIndexer>
	inline aview1d<T, LIndexer>& operator *= (aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		mul_arr_sca_inplace(lhs, rhs);
		return lhs;
	}

	// 2D

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator * (
			const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return mul_arr_arr(lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> operator * (const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return mul_arr_sca(lhs, rhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator * (const T& lhs, const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return mul_arr_sca(rhs, lhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator *= (
			aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		mul_arr_arr_inplace(lhs, rhs);
		return lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator *= (aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		mul_arr_sca_inplace(lhs, rhs);
		return lhs;
	}


	// division

	// 1D

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> operator / (const aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		return div_arr_arr(lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator / (const aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return div_arr_sca(lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator / (const T& lhs, const aview1d<T, LIndexer>& rhs)
	{
		return div_sca_arr(lhs, rhs);
	}

	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>& operator /= (aview1d<T, LIndexer>& lhs, const aview1d<T, RIndexer>& rhs)
	{
		div_arr_arr_inplace(lhs, rhs);
		return lhs;
	}

	template<typename T, class LIndexer>
	inline aview1d<T, LIndexer>& operator /= (aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		div_arr_sca_inplace(lhs, rhs);
		return lhs;
	}

	// 2D

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator / (
			const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return div_arr_arr(lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> operator / (const aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return div_arr_sca(lhs, rhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator / (const T& lhs, const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return div_sca_arr(lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator /= (
			aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		div_arr_arr_inplace(lhs, rhs);
		return lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator /= (aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		div_arr_sca_inplace(lhs, rhs);
		return lhs;
	}


	// negation

	template<typename T, class RIndexer>
	inline array1d<T> operator - (const aview1d<T, RIndexer>& rhs)
	{
		return neg_arr(rhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator - (const aview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return neg_arr(rhs);
	}

	template<typename T, class RIndexer>
	inline aview1d<T, RIndexer>& neg_ip(aview1d<T, RIndexer>& a)
	{
		neg_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, RIndexer0, RIndexer1>& neg_ip(aview2d<T, TOrd, RIndexer0, RIndexer1>& a)
	{
		neg_arr_inplace(a);
		return a;
	}


	// absolute value

	template<typename T, class TIndexer>
	inline array1d<T> abs(const aview1d<T, TIndexer>& a)
	{
		return abs_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> abs(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return abs_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& abs_ip(aview1d<T, TIndexer>& a)
	{
		abs_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& abs_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		abs_arr_inplace(a);
		return a;
	}


	/******************************************************
	 *
	 *  Elementary functions
	 *
	 ******************************************************/

	// power and root functions

	// sqr

	template<typename T, class TIndexer>
	inline array1d<T> sqr(const aview1d<T, TIndexer>& a)
	{
		return sqr_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> sqr(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return sqr_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& sqr_ip(aview1d<T, TIndexer>& a)
	{
		sqr_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& sqr_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		sqr_arr_inplace(a);
		return a;
	}


	// sqrt

	template<typename T, class TIndexer>
	inline array1d<T> sqrt(const aview1d<T, TIndexer>& a)
	{
		return sqrt_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> sqrt(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return sqrt_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& sqrt_ip(aview1d<T, TIndexer>& a)
	{
		sqrt_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& sqrt_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		sqrt_arr_inplace(a);
		return a;
	}


	// rcp

	template<typename T, class TIndexer>
	inline array1d<T> rcp(const aview1d<T, TIndexer>& a)
	{
		return rcp_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> rcp(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return rcp_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& rcp_ip(aview1d<T, TIndexer>& a)
	{
		rcp_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& rcp_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		rcp_arr_inplace(a);
		return a;
	}


	// rsqrt

	template<typename T, class TIndexer>
	inline array1d<T> rsqrt(const aview1d<T, TIndexer>& a)
	{
		return rsqrt_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> rsqrt(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return rsqrt_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& rsqrt_ip(aview1d<T, TIndexer>& a)
	{
		rsqrt_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& rsqrt_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		rsqrt_arr_inplace(a);
		return a;
	}


	// pow

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> pow(const aview1d<T, LIndexer>& a, const aview1d<T, RIndexer>& e)
	{
		return pow_arr(a, e);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> pow(
			const aview2d<T, TOrd, LIndexer0, LIndexer1>& a,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& e)
	{
		return pow_arr(a, e);
	}


	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>& pow_ip(aview1d<T, LIndexer>& a, const aview1d<T, RIndexer>& e)
	{
		pow_arr_inplace(a, e);
		return a;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& pow_ip(
			aview2d<T, TOrd, LIndexer0, LIndexer1>& a,
			const aview2d<T, TOrd, RIndexer0, RIndexer1>& e)
	{
		pow_arr_inplace(a, e);
		return a;
	}


	// pow with constant

	template<typename T, class LIndexer>
	inline array1d<T> pow(const aview1d<T, LIndexer>& a, const T& e)
	{
		return pow_arr_sca(a, e);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> pow(const aview2d<T, TOrd, LIndexer0, LIndexer1>& a, const T& e)
	{
		return pow_arr_sca(a, e);
	}

	template<typename T, class LIndexer>
	inline aview1d<T, LIndexer>& pow_ip(aview1d<T, LIndexer>& a, const T& e)
	{
		pow_arr_sca_inplace(a, e);
		return a;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& pow_ip(aview2d<T, TOrd, LIndexer0, LIndexer1>& a, const T& e)
	{
		pow_arr_sca_inplace(a, e);
		return a;
	}


}

#endif 
