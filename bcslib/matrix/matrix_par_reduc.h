/**
 * @file matrix_par_reduc.h
 *
 * Partial reduction on matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_PAR_REDUC_H_
#define BCSLIB_MATRIX_PAR_REDUC_H_

#include <bcslib/matrix/slicewise_proxy.h>
#include <bcslib/matrix/matrix_par_reduc_expr.h>
#include <bcslib/engine/reduction_functors.h>


namespace bcs
{

	// sum

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<sum_reductor<T>, Mat>
	sum(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(sum_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<sum_reductor<T>, Mat>
	sum(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(sum_reductor<T>(), proxy.ref());
	}


	// mean

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<mean_reductor<T>, Mat>
	mean(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(mean_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<mean_reductor<T>, Mat>
	mean(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(mean_reductor<T>(), proxy.ref());
	}


	// min

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<min_reductor<T>, Mat>
	min_val(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(min_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<min_reductor<T>, Mat>
	min_val(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(min_reductor<T>(), proxy.ref());
	}


	// max

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<max_reductor<T>, Mat>
	max_val(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(max_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<max_reductor<T>, Mat>
	max_val(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(max_reductor<T>(), proxy.ref());
	}

	// L1norm

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<L1norm_reductor<T>, Mat>
	L1norm(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(L1norm_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<L1norm_reductor<T>, Mat>
	L1norm(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(L1norm_reductor<T>(), proxy.ref());
	}


	// sqL2norm

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<sqL2norm_reductor<T>, Mat>
	sqL2norm(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(sqL2norm_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<sqL2norm_reductor<T>, Mat>
	sqL2norm(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(sqL2norm_reductor<T>(), proxy.ref());
	}

	// L2norm

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<L2norm_reductor<T>, Mat>
	L2norm(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(L2norm_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<L2norm_reductor<T>, Mat>
	L2norm(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(L2norm_reductor<T>(), proxy.ref());
	}


	// Linfnorm

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_colwise_reduction_expr<Linfnorm_reductor<T>, Mat>
	Linfnorm(const_colwise_proxy<Mat, T> proxy)
	{
		return colwise_reduce(Linfnorm_reductor<T>(), proxy.ref());
	}

	template<typename T, class Mat>
	BCS_ENSURE_INLINE
	unary_rowwise_reduction_expr<Linfnorm_reductor<T>, Mat>
	Linfnorm(const_rowwise_proxy<Mat, T> proxy)
	{
		return rowwise_reduce(Linfnorm_reductor<T>(), proxy.ref());
	}



	// dot

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<dot_reductor<T>, LMat, RMat>
	dot(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(dot_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<dot_reductor<T>, LMat, RMat>
	dot(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(dot_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	// L1norm_diff

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<L1diffnorm_reductor<T>, LMat, RMat>
	L1norm_diff(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(L1diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<L1diffnorm_reductor<T>, LMat, RMat>
	L1norm_diff(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(L1diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	// sqL2norm_diff

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<sqL2diffnorm_reductor<T>, LMat, RMat>
	sqL2norm_diff(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(sqL2diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<sqL2diffnorm_reductor<T>, LMat, RMat>
	sqL2norm_diff(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(sqL2diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	// L2norm_diff

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<L2diffnorm_reductor<T>, LMat, RMat>
	L2norm_diff(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(L2diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<L2diffnorm_reductor<T>, LMat, RMat>
	L2norm_diff(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(L2diffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}


	// Linfnorm_diff

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_colwise_reduction_expr<Linfdiffnorm_reductor<T>, LMat, RMat>
	Linfnorm_diff(const_colwise_proxy<LMat, T> lproxy, const_colwise_proxy<RMat, T> rproxy)
	{
		return colwise_reduce(Linfdiffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

	template<typename T, class LMat, class RMat>
	BCS_ENSURE_INLINE
	binary_rowwise_reduction_expr<Linfdiffnorm_reductor<T>, LMat, RMat>
	Linfnorm_diff(const_rowwise_proxy<LMat, T> lproxy, const_rowwise_proxy<RMat, T> rproxy)
	{
		return rowwise_reduce(Linfdiffnorm_reductor<T>(), lproxy.ref(), rproxy.ref());
	}

}


#endif /* MATRIX_PAR_REDUC_H_ */


