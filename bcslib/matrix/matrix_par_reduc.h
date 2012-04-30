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


}


#endif /* MATRIX_PAR_REDUC_H_ */


