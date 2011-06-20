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


	// exponential and logarithm functions

	// exp

	template<typename T, class TIndexer>
	inline array1d<T> exp(const aview1d<T, TIndexer>& a)
	{
		return exp_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> exp(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return exp_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& exp_ip(aview1d<T, TIndexer>& a)
	{
		exp_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& exp_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		exp_arr_inplace(a);
		return a;
	}

	// log

	template<typename T, class TIndexer>
	inline array1d<T> log(const aview1d<T, TIndexer>& a)
	{
		return log_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> log(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return log_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& log_ip(aview1d<T, TIndexer>& a)
	{
		log_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& log_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		log_arr_inplace(a);
		return a;
	}

	// log10

	template<typename T, class TIndexer>
	inline array1d<T> log10(const aview1d<T, TIndexer>& a)
	{
		return log10_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> log10(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return log10_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& log10_ip(aview1d<T, TIndexer>& a)
	{
		log10_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& log10_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		log10_arr_inplace(a);
		return a;
	}


	// rounding functions

	// floor

	template<typename T, class TIndexer>
	inline array1d<T> floor(const aview1d<T, TIndexer>& a)
	{
		return floor_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> floor(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return floor_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& floor_ip(aview1d<T, TIndexer>& a)
	{
		floor_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& floor_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		floor_arr_inplace(a);
		return a;
	}

	// ceil

	template<typename T, class TIndexer>
	inline array1d<T> ceil(const aview1d<T, TIndexer>& a)
	{
		return ceil_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> ceil(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return ceil_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& ceil_ip(aview1d<T, TIndexer>& a)
	{
		ceil_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& ceil_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		ceil_arr_inplace(a);
		return a;
	}


	// trigonometric functions

	// sin

	template<typename T, class TIndexer>
	inline array1d<T> sin(const aview1d<T, TIndexer>& a)
	{
		return sin_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> sin(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return sin_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& sin_ip(aview1d<T, TIndexer>& a)
	{
		sin_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& sin_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		sin_arr_inplace(a);
		return a;
	}

	// cos

	template<typename T, class TIndexer>
	inline array1d<T> cos(const aview1d<T, TIndexer>& a)
	{
		return cos_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> cos(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return cos_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& cos_ip(aview1d<T, TIndexer>& a)
	{
		cos_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& cos_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		cos_arr_inplace(a);
		return a;
	}

	// tan

	template<typename T, class TIndexer>
	inline array1d<T> tan(const aview1d<T, TIndexer>& a)
	{
		return tan_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> tan(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return tan_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& tan_ip(aview1d<T, TIndexer>& a)
	{
		tan_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& tan_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		tan_arr_inplace(a);
		return a;
	}

	// asin

	template<typename T, class TIndexer>
	inline array1d<T> asin(const aview1d<T, TIndexer>& a)
	{
		return asin_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> asin(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return asin_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& asin_ip(aview1d<T, TIndexer>& a)
	{
		asin_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& asin_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		asin_arr_inplace(a);
		return a;
	}

	// acos

	template<typename T, class TIndexer>
	inline array1d<T> acos(const aview1d<T, TIndexer>& a)
	{
		return acos_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> acos(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return acos_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& acos_ip(aview1d<T, TIndexer>& a)
	{
		acos_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& acos_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		acos_arr_inplace(a);
		return a;
	}


	// atan

	template<typename T, class TIndexer>
	inline array1d<T> atan(const aview1d<T, TIndexer>& a)
	{
		return atan_arr(a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> atan(const aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return atan_arr(a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& atan_ip(aview1d<T, TIndexer>& a)
	{
		atan_arr_inplace(a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& atan_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		atan_arr_inplace(a);
		return a;
	}


	// atan2

	template<typename T, class TIndexer>
	inline array1d<T> atan2(const aview1d<T, TIndexer>& a, const aview1d<T, TIndexer>& b)
	{
		return atan2_arr(a, b);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> atan2(
			const aview2d<T, TOrd, TIndexer0, TIndexer1>& a,
			const aview2d<T, TOrd, TIndexer0, TIndexer1>& b)
	{
		return atan2_arr(a, b);
	}

}

#endif 
