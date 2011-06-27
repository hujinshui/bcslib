/**
 * @file array_calc.h
 *
 * Vectorized calculation on arrays
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_ARRAY_CALC_H
#define BCSLIB_ARRAY_CALC_H

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <bcslib/array/generic_array_eval.h>
#include <bcslib/veccomp/veccalc.h>

namespace bcs
{
	/********************************************
	 *
	 *  Comparison
	 *
	 *******************************************/

	// eq

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<bool> eq(const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_eq_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<bool> eq(const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_eq_ftor<T>(rhs), lhs);
	}

	template<typename T, class RIndexer>
	inline array1d<bool> eq(const T& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return eq(rhs, lhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> eq(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_eq_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<bool, TOrd> eq(const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_eq_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> eq(const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return eq(rhs, lhs);
	}


	// ne

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<bool> ne(const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_ne_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<bool> ne(const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_ne_ftor<T>(rhs), lhs);
	}

	template<typename T, class RIndexer>
	inline array1d<bool> ne(const T& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return ne(rhs, lhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> ne(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_ne_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<bool, TOrd> ne(const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_ne_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> ne(const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return ne(rhs, lhs);
	}


	// gt

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<bool> gt(const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_gt_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<bool> gt(const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_gt_ftor<T>(rhs), lhs);
	}

	template<typename T, class RIndexer>
	inline array1d<bool> gt(const T& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(sca_vec_gt_ftor<T>(lhs), rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> gt(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_gt_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<bool, TOrd> gt(const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_gt_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> gt(const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(sca_vec_gt_ftor<T>(lhs), rhs);
	}


	// ge

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<bool> ge(const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_ge_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<bool> ge(const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_ge_ftor<T>(rhs), lhs);
	}

	template<typename T, class RIndexer>
	inline array1d<bool> ge(const T& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(sca_vec_ge_ftor<T>(lhs), rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> ge(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_ge_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<bool, TOrd> ge(const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_ge_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> ge(const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(sca_vec_ge_ftor<T>(lhs), rhs);
	}


	// lt

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<bool> lt(const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_lt_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<bool> lt(const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_lt_ftor<T>(rhs), lhs);
	}

	template<typename T, class RIndexer>
	inline array1d<bool> lt(const T& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(sca_vec_lt_ftor<T>(lhs), rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> lt(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_lt_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<bool, TOrd> lt(const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_lt_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> lt(const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(sca_vec_lt_ftor<T>(lhs), rhs);
	}


	// le

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<bool> le(const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_le_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<bool> le(const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_le_ftor<T>(rhs), lhs);
	}

	template<typename T, class RIndexer>
	inline array1d<bool> le(const T& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(sca_vec_le_ftor<T>(lhs), rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> le(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_le_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<bool, TOrd> le(const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_le_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<bool, TOrd> le(const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(sca_vec_le_ftor<T>(lhs), rhs);
	}


	// max_each

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> max_each(const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_max_each_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> max_each(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_max_each_ftor<T>(), lhs, rhs);
	}

	// min_each

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> min_each(const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_min_each_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> min_each(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_min_each_ftor<T>(), lhs, rhs);
	}


	/********************************************
	 *
	 *  Bounding
	 *
	 ********************************************/

	// lbound

	template<typename T, class TIndexer>
	inline array1d<T> lbound(const caview1d<T, TIndexer>& a, const T& lb)
	{
		return transform_arr(vec_lbound_ftor<T>(lb), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> lbound(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T& lb)
	{
		return transform_arr(vec_lbound_ftor<T>(lb), a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& lbound_ip(aview1d<T, TIndexer>& a, const T& lb)
	{
		inplace_transform_arr(vec_lbound_ftor<T>(lb), a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& lbound_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T& lb)
	{
		inplace_transform_arr(vec_lbound_ftor<T>(lb), a);
		return a;
	}


	// ubound

	template<typename T, class TIndexer>
	inline array1d<T> ubound(const caview1d<T, TIndexer>& a, const T& lb)
	{
		return transform_arr(vec_ubound_ftor<T>(lb), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> ubound(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T& lb)
	{
		return transform_arr(vec_ubound_ftor<T>(lb), a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& ubound_ip(aview1d<T, TIndexer>& a, const T& lb)
	{
		inplace_transform_arr(vec_ubound_ftor<T>(lb), a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& ubound_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T& lb)
	{
		inplace_transform_arr(vec_ubound_ftor<T>(lb), a);
		return a;
	}


	// rgn_bound

	template<typename T, class TIndexer>
	inline array1d<T> rgn_bound(const caview1d<T, TIndexer>& a, const T& lb, const T& ub)
	{
		return transform_arr(vec_rgn_bound_ftor<T>(lb, ub), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> rgn_bound(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T& lb, const T& ub)
	{
		return transform_arr(vec_rgn_bound_ftor<T>(lb, ub), a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& rgn_bound_ip(aview1d<T, TIndexer>& a, const T& lb, const T& ub)
	{
		inplace_transform_arr(vec_rgn_bound_ftor<T>(lb, ub), a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& rgn_bound_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T& lb, const T& ub)
	{
		inplace_transform_arr(vec_rgn_bound_ftor<T>(lb, ub), a);
		return a;
	}


	// abound

	template<typename T, class TIndexer>
	inline array1d<T> abound(const caview1d<T, TIndexer>& a, const T& lb)
	{
		return transform_arr(vec_abound_ftor<T>(lb), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> abound(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T& lb)
	{
		return transform_arr(vec_abound_ftor<T>(lb), a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& abound_ip(aview1d<T, TIndexer>& a, const T& lb)
	{
		inplace_transform_arr(vec_abound_ftor<T>(lb), a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& abound_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a, const T& lb)
	{
		inplace_transform_arr(vec_abound_ftor<T>(lb), a);
		return a;
	}


	/******************************************************
	 *
	 *  Arithmetic operators/functions
	 *
	 ******************************************************/

	// addition

	// 1D

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> operator + (const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_add_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator + (const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_add_ftor<T>(rhs), lhs);
	}

	template<typename T, class RIndexer>
	inline array1d<T> operator + (const T& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return rhs + lhs;
	}

	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>& operator += (aview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		inplace_transform_arr(vec_vec_add_ftor<T>(), lhs, rhs);
		return lhs;
	}

	template<typename T, class LIndexer>
	inline aview1d<T, LIndexer>& operator += (aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		inplace_transform_arr(vec_sca_add_ftor<T>(rhs), lhs);
		return lhs;
	}

	// 2D

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator + (
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_add_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> operator + (const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_add_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator + (const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return rhs + lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator += (
			aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		inplace_transform_arr(vec_vec_add_ftor<T>(), lhs, rhs);
		return lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator += (aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		inplace_transform_arr(vec_sca_add_ftor<T>(rhs), lhs);
		return lhs;
	}



	// subtraction

	// 1D

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> operator - (const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_sub_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator - (const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_sub_ftor<T>(rhs), lhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator - (const T& lhs, const caview1d<T, LIndexer>& rhs)
	{
		return transform_arr(sca_vec_sub_ftor<T>(lhs), rhs);
	}

	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>& operator -= (aview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		inplace_transform_arr(vec_vec_sub_ftor<T>(), lhs, rhs);
		return lhs;
	}

	template<typename T, class LIndexer>
	inline aview1d<T, LIndexer>& operator -= (aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		inplace_transform_arr(vec_sca_sub_ftor<T>(rhs), lhs);
		return lhs;
	}

	// 2D

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator - (
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_sub_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> operator - (const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_sub_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator - (const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(sca_vec_sub_ftor<T>(lhs), rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator -= (
			aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		inplace_transform_arr(vec_vec_sub_ftor<T>(), lhs, rhs);
		return lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator -= (aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		inplace_transform_arr(vec_sca_sub_ftor<T>(rhs), lhs);
		return lhs;
	}



	// multiplication

	// 1D

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> operator * (const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_mul_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator * (const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_mul_ftor<T>(rhs), lhs);
	}

	template<typename T, class RIndexer>
	inline array1d<T> operator * (const T& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return rhs * lhs;
	}

	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>& operator *= (aview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		inplace_transform_arr(vec_vec_mul_ftor<T>(), lhs, rhs);
		return lhs;
	}

	template<typename T, class LIndexer>
	inline aview1d<T, LIndexer>& operator *= (aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		inplace_transform_arr(vec_sca_mul_ftor<T>(rhs), lhs);
		return lhs;
	}

	// 2D

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator * (
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_mul_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> operator * (const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_mul_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator * (const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return rhs * lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator *= (
			aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		inplace_transform_arr(vec_vec_mul_ftor<T>(), lhs, rhs);
		return lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator *= (aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		inplace_transform_arr(vec_sca_mul_ftor<T>(rhs), lhs);
		return lhs;
	}



	// division

	// 1D

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> operator / (const caview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_vec_div_ftor<T>(), lhs, rhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator / (const caview1d<T, LIndexer>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_div_ftor<T>(rhs), lhs);
	}

	template<typename T, class LIndexer>
	inline array1d<T> operator / (const T& lhs, const caview1d<T, LIndexer>& rhs)
	{
		return transform_arr(sca_vec_div_ftor<T>(lhs), rhs);
	}

	template<typename T, class LIndexer, class RIndexer>
	inline aview1d<T, LIndexer>& operator /= (aview1d<T, LIndexer>& lhs, const caview1d<T, RIndexer>& rhs)
	{
		inplace_transform_arr(vec_vec_div_ftor<T>(), lhs, rhs);
		return lhs;
	}

	template<typename T, class LIndexer>
	inline aview1d<T, LIndexer>& operator /= (aview1d<T, LIndexer>& lhs, const T& rhs)
	{
		inplace_transform_arr(vec_sca_div_ftor<T>(rhs), lhs);
		return lhs;
	}

	// 2D

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator / (
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_vec_div_ftor<T>(), lhs, rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> operator / (const caview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		return transform_arr(vec_sca_div_ftor<T>(rhs), lhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator / (const T& lhs, const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(sca_vec_div_ftor<T>(lhs), rhs);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator /= (
			aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		inplace_transform_arr(vec_vec_div_ftor<T>(), lhs, rhs);
		return lhs;
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline aview2d<T, TOrd, LIndexer0, LIndexer1>& operator /= (aview2d<T, TOrd, LIndexer0, LIndexer1>& lhs, const T& rhs)
	{
		inplace_transform_arr(vec_sca_div_ftor<T>(rhs), lhs);
		return lhs;
	}


	// negation

	template<typename T, class RIndexer>
	inline array1d<T> operator - (const caview1d<T, RIndexer>& rhs)
	{
		return transform_arr(vec_neg_ftor<T>(), rhs);
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> operator - (const caview2d<T, TOrd, RIndexer0, RIndexer1>& rhs)
	{
		return transform_arr(vec_neg_ftor<T>(), rhs);
	}

	template<typename T, class RIndexer>
	inline aview1d<T, RIndexer>& neg_ip(aview1d<T, RIndexer>& a)
	{
		inplace_transform_arr(vec_neg_ftor<T>(), a);
		return a;
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	inline aview2d<T, TOrd, RIndexer0, RIndexer1>& neg_ip(aview2d<T, TOrd, RIndexer0, RIndexer1>& a)
	{
		inplace_transform_arr(vec_neg_ftor<T>(), a);
		return a;
	}


	// absolute value

	template<typename T, class TIndexer>
	inline array1d<T> abs(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_abs_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> abs(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_abs_ftor<T>(), a);
	}

	template<typename T, class TIndexer>
	inline aview1d<T, TIndexer>& abs_ip(aview1d<T, TIndexer>& a)
	{
		inplace_transform_arr(vec_abs_ftor<T>(), a);
		return a;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline aview2d<T, TOrd, TIndexer0, TIndexer1>& abs_ip(aview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		inplace_transform_arr(vec_abs_ftor<T>(), a);
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
	inline array1d<T> sqr(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_sqr_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> sqr(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_sqr_ftor<T>(), a);
	}

	// sqrt

	template<typename T, class TIndexer>
	inline array1d<T> sqrt(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_sqrt_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> sqrt(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_sqrt_ftor<T>(), a);
	}

	// rcp

	template<typename T, class TIndexer>
	inline array1d<T> rcp(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_rcp_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> rcp(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_rcp_ftor<T>(), a);
	}

	// rsqrt

	template<typename T, class TIndexer>
	inline array1d<T> rsqrt(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_rsqrt_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> rsqrt(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_rsqrt_ftor<T>(), a);
	}

	// pow

	template<typename T, class LIndexer, class RIndexer>
	inline array1d<T> pow(const caview1d<T, LIndexer>& a, const caview1d<T, RIndexer>& e)
	{
		return transform_arr(vec_pow_ftor<T>(), a, e);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> pow(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& a,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& e)
	{
		return transform_arr(vec_pow_ftor<T>(), a, e);
	}

	// pow with constant

	template<typename T, class LIndexer>
	inline array1d<T> pow(const caview1d<T, LIndexer>& a, const T& e)
	{
		return transform_arr(vec_sca_pow_ftor<T>(e), a);
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	inline array2d<T, TOrd> pow(const caview2d<T, TOrd, LIndexer0, LIndexer1>& a, const T& e)
	{
		return transform_arr(vec_sca_pow_ftor<T>(e), a);
	}

	// exponential and logarithm functions

	// exp

	template<typename T, class TIndexer>
	inline array1d<T> exp(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_exp_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> exp(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_exp_ftor<T>(), a);
	}


	// log

	template<typename T, class TIndexer>
	inline array1d<T> log(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_log_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> log(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_log_ftor<T>(), a);
	}

	// log10

	template<typename T, class TIndexer>
	inline array1d<T> log10(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_log10_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> log10(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_log10_ftor<T>(), a);
	}


	// rounding functions

	// floor

	template<typename T, class TIndexer>
	inline array1d<T> floor(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_floor_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> floor(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_floor_ftor<T>(), a);
	}

	// ceil

	template<typename T, class TIndexer>
	inline array1d<T> ceil(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_ceil_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> ceil(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_ceil_ftor<T>(), a);
	}

	// round

	template<typename T, class TIndexer>
	inline array1d<T> round(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_round_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> round(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_round_ftor<T>(), a);
	}


	// trigonometric functions

	// sin

	template<typename T, class TIndexer>
	inline array1d<T> sin(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_sin_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> sin(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_sin_ftor<T>(), a);
	}

	// cos

	template<typename T, class TIndexer>
	inline array1d<T> cos(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_cos_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> cos(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_cos_ftor<T>(), a);
	}

	// tan

	template<typename T, class TIndexer>
	inline array1d<T> tan(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_tan_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> tan(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_tan_ftor<T>(), a);
	}

	// asin

	template<typename T, class TIndexer>
	inline array1d<T> asin(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_asin_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> asin(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_asin_ftor<T>(), a);
	}

	// acos

	template<typename T, class TIndexer>
	inline array1d<T> acos(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_acos_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> acos(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_acos_ftor<T>(), a);
	}

	// atan

	template<typename T, class TIndexer>
	inline array1d<T> atan(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_atan_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> atan(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_atan_ftor<T>(), a);
	}

	// atan2

	template<typename T, class TIndexer>
	inline array1d<T> atan2(const caview1d<T, TIndexer>& a, const caview1d<T, TIndexer>& b)
	{
		return transform_arr(vec_atan2_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> atan2(
			const caview2d<T, TOrd, TIndexer0, TIndexer1>& a,
			const caview2d<T, TOrd, TIndexer0, TIndexer1>& b)
	{
		return transform_arr(vec_atan2_ftor<T>(), a, b);
	}


	// hyperbolic functions

	// sinh

	template<typename T, class TIndexer>
	inline array1d<T> sinh(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_sinh_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> sinh(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_sinh_ftor<T>(), a);
	}

	// cosh

	template<typename T, class TIndexer>
	inline array1d<T> cosh(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_cosh_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> cosh(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_cosh_ftor<T>(), a);
	}

	// tanh

	template<typename T, class TIndexer>
	inline array1d<T> tanh(const caview1d<T, TIndexer>& a)
	{
		return transform_arr(vec_tanh_ftor<T>(), a);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> tanh(const caview2d<T, TOrd, TIndexer0, TIndexer1>& a)
	{
		return transform_arr(vec_tanh_ftor<T>(), a);
	}


	// accuracy-preserving functions

	// hypot

	template<typename T, class TIndexer>
	inline array1d<T> hypot(const caview1d<T, TIndexer>& a, const caview1d<T, TIndexer>& b)
	{
		return transform_arr(vec_hypot_ftor<T>(), a, b);
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	inline array2d<T, TOrd> hypot(
			const caview2d<T, TOrd, TIndexer0, TIndexer1>& a,
			const caview2d<T, TOrd, TIndexer0, TIndexer1>& b)
	{
		return transform_arr(vec_hypot_ftor<T>(), a, b);
	}

}

#endif 
