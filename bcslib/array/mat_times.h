/**
 * @file mat_times.h
 *
 * Matrix/Vector multiplication (in Linear algebraic sense)
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_MAT_TIMES_H_
#define BCSLIB_MAT_TIMES_H_

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/array/generic_blas.h>

namespace bcs
{
	// matrix-vector product


	// a * x

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer>
	inline array1d<T> mtimes(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& a,
			const caview1d<T, RIndexer>& x)
	{
		check_arg(a.dim1() == x.dim0(), "mtimes: inconsistent dimensions");

		array1d<T> y(a.nrows());
		blas::gemv(a, x, y, 'N');

		return y;
	}

	// a^T * x

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer>
	inline array1d<T> mtimes(
			const _detail::_arr2d_transpose_t<T, TOrd, LIndexer0, LIndexer1>& at,
			const caview1d<T, RIndexer>& x)
	{
		const caview2d<T, TOrd, LIndexer0, LIndexer1>& a = at.get();
		check_arg(a.dim0() == x.dim0(), "mtimes: inconsistent dimensions");

		array1d<T> y(a.ncolumns());
		blas::gemv(a, x, y, 'T');

		return y;
	}

	// x * a

	template<typename T, typename TOrd, class LIndexer, class RIndexer0, class RIndexer1>
	inline array1d<T> mtimes(
			const caview1d<T, LIndexer>& x,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& a)
	{
		check_arg(x.dim0() == a.dim0(), "mtimes: inconsistent dimensions");

		array1d<T> y(a.ncolumns());
		blas::gemv(a, x, y,'T');

		return y;
	}

	// x * a^T

	template<typename T, typename TOrd, class LIndexer, class RIndexer0, class RIndexer1>
	inline array1d<T> mtimes(
			const caview1d<T, LIndexer>& x,
			const _detail::_arr2d_transpose_t<T, TOrd, RIndexer0, RIndexer1>& at)
	{
		const caview2d<T, TOrd, RIndexer0, RIndexer1>& a = at.get();
		check_arg(x.dim0() == a.dim1(), "mtimes: inconsistent dimensions");

		array1d<T> y(a.nrows());
		blas::gemv(a, x, y, 'N');

		return y;
	}


	// matrix-matrix product

	// a * b

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> mtimes(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& a,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		check_arg(a.dim1() == b.dim0(), "mtimes: inconsistent dimensions");

		array2d<T, TOrd> c(a.nrows(), b.ncolumns());
		blas::gemm(a, b, c, 'N', 'N');

		return c;
	}

	// a * b^T

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> mtimes(
			const caview2d<T, TOrd, LIndexer0, LIndexer1>& a,
			const _detail::_arr2d_transpose_t<T, TOrd, RIndexer0, RIndexer1>& bt)
	{
		const caview2d<T, TOrd, RIndexer0, RIndexer1>& b = bt.get();
		check_arg(a.dim1() == b.dim1(), "mtimes: inconsistent dimensions");

		array2d<T, TOrd> c(a.nrows(), b.nrows());
		blas::gemm(a, b, c, 'N', 'T');

		return c;
	}

	// a^T * b

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> mtimes(
			const _detail::_arr2d_transpose_t<T, TOrd, LIndexer0, LIndexer1>& at,
			const caview2d<T, TOrd, RIndexer0, RIndexer1>& b)
	{
		const caview2d<T, TOrd, LIndexer0, LIndexer1>& a = at.get();
		check_arg(a.dim0() == b.dim0(), "mtimes: inconsistent dimensions");

		array2d<T, TOrd> c(a.ncolumns(), b.ncolumns());
		blas::gemm(a, b, c, 'T', 'N');

		return c;
	}

	// a^T * b^T

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	inline array2d<T, TOrd> mtimes(
			const _detail::_arr2d_transpose_t<T, TOrd, LIndexer0, LIndexer1>& at,
			const _detail::_arr2d_transpose_t<T, TOrd, RIndexer0, RIndexer1>& bt)
	{
		const caview2d<T, TOrd, LIndexer0, LIndexer1>& a = at.get();
		const caview2d<T, TOrd, RIndexer0, RIndexer1>& b = bt.get();
		check_arg(a.dim0() == b.dim1(), "mtimes: inconsistent dimensions");

		array2d<T, TOrd> c(a.ncolumns(), b.nrows());
		blas::gemm(a, b, c, 'T', 'T');

		return c;
	}

}

#endif 



