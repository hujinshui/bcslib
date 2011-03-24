/**
 * @file array_calc.h
 *
 * Vectorized calculation on arrays
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_CALC_H
#define BCSLIB_ARRAY_CALC_H

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <bcslib/veccomp/veccalc.h>

namespace bcs
{

	// generic functions

	template<typename T, class TIndexer, typename TVecFunc>
	inline array1d<T> array_calc(const const_aview1d<T, TIndexer>& x, TVecFunc vecfunc)
	{
		size_t n = x.nelems();
		array1d<T> y(n);

		if (is_dense_view(x))
		{
			vecfunc(n, x.pbase(), y.pbase());
		}
		else
		{
			array1d<T> xc = make_copy(x);
			vecfunc(n, xc.pbase(), y.pbase());
		}
		return y;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename TVecFunc>
	inline array2d<T, TOrd> array_calc(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x, TVecFunc vecfunc)
	{
		array2d<T, TOrd> y(x.dim0(), x.dim1());

		if (is_dense_view(x))
		{
			vecfunc(x.nelems(), x.pbase(), y.pbase());
		}
		else
		{
			array2d<T, TOrd> xc = make_copy(x);
			vecfunc(xc.nelems(), xc.pbase(), y.pbase());
		}
		return y;
	}


	template<typename T, class LIndexer, class RIndexer, typename TVecFunc>
	inline array1d<T> array_calc(const const_aview1d<T, LIndexer>& x1, const const_aview1d<T, RIndexer>& x2, TVecFunc vecfunc)
	{
		if ( !is_same_shape(x1, x2) )
		{
			throw array_size_mismatch();
		}

		size_t n = x1.nelems();
		array1d<T> y(n);

		if (is_dense_view(x1))
		{
			if (is_dense_view(x2))
			{
				vecfunc(n, x1.pbase(), x2.pbase(), y.pbase());
			}
			else
			{
				array1d<T> x2c = make_copy(x2);
				vecfunc(n, x1.pbase(), x2c.pbase(), y.pbase());
			}
		}
		else
		{
			array1d<T> x1c = make_copy(x1);
			if (is_dense_view(x2))
			{
				vecfunc(n, x1c.pbase(), x2.pbase(), y.pbase());
			}
			else
			{
				array1d<T> x2c = make_copy(x2);
				vecfunc(n, x1c.pbase(), x2c.pbase(), y.pbase());
			}
		}

		return y;
	}


	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1, typename TVecFunc>
	inline array2d<T, TOrd> array_calc(
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2,
			TVecFunc vecfunc)
	{
		if ( !is_same_shape(x1, x2) )
		{
			throw array_size_mismatch();
		}

		array2d<T, TOrd> y(x1.dim0(), x1.dim1());
		size_t n = x1.nelems();

		if (is_dense_view(x1))
		{
			if (is_dense_view(x2))
			{
				vecfunc(n, x1.pbase(), x2.pbase(), y.pbase());
			}
			else
			{
				array2d<T, TOrd> x2c = make_copy(x2);
				vecfunc(n, x1.pbase(), x2c.pbase(), y.pbase());
			}
		}
		else
		{
			array2d<T, TOrd> x1c = make_copy(x1);
			if (is_dense_view(x2))
			{
				vecfunc(n, x1c.pbase(), x2.pbase(), y.pbase());
			}
			else
			{
				array2d<T, TOrd> x2c = make_copy(x2);
				vecfunc(n, x1c.pbase(), x2c.pbase(), y.pbase());
			}
		}

		return y;
	}


	template<typename T, class LIndexer, typename TVecFunc>
	inline array1d<T> array_calc(const const_aview1d<T, LIndexer>& x1, const T& x2, TVecFunc vecfunc)
	{
		size_t n = x1.nelems();
		array1d<T> y(n);

		if (is_dense_view(x1))
		{
			vecfunc(n, x1.pbase(), x2, y.pbase());
		}
		else
		{
			array1d<T> x1c = make_copy(x1);
			vecfunc(n, x1c.pbase(), x2, y.pbase());
		}

		return y;
	}


	template<typename T, class RIndexer, typename TVecFunc>
	inline array1d<T> array_calc(const T& x1, const const_aview1d<T, RIndexer>& x2, TVecFunc vecfunc)
	{
		size_t n = x2.nelems();
		array1d<T> y(n);

		if (is_dense_view(x2))
		{
			vecfunc(n, x1, x2.pbase(), y.pbase());
		}
		else
		{
			array1d<T> x2c = make_copy(x2);
			vecfunc(n, x1, x2c.pbase(), y.pbase());
		}

		return y;
	}


	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, typename TVecFunc>
	inline array2d<T, TOrd> array_calc(const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1, const T& x2, TVecFunc vecfunc)
	{
		array2d<T, TOrd> y(x1.dim0(), x1.dim1());
		size_t n = x1.nelems();

		if (is_dense_view(x1))
		{
			vecfunc(n, x1.pbase(), x2, y.pbase());
		}
		else
		{
			array2d<T, TOrd> x1c = make_copy(x1);
			vecfunc(n, x1c.pbase(), x2, y.pbase());
		}
		return y;
	}


	template<typename T, typename TOrd, class RIndexer0, class RIndexer1, typename TVecFunc>
	inline array2d<T, TOrd> array_calc(const T& x1, const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2, TVecFunc vecfunc)
	{
		array2d<T, TOrd> y(x2.dim0(), x2.dim1());
		size_t n = x2.nelems();

		if (is_dense_view(x2))
		{
			vecfunc(n, x1, x2.pbase(), y.pbase());
		}
		else
		{
			array2d<T, TOrd> x2c = make_copy(x2);
			vecfunc(n, x1, x2c.pbase(), y.pbase());
		}
		return y;
	}

	template<typename T, class TIndexer, typename TVecFunc>
	inline void array_calc_inplace(aview1d<T, TIndexer>& x, TVecFunc vecfunc)
	{
		size_t n = x.nelems();

		if (is_dense_view(x))
		{
			vecfunc(n, x.pbase());
		}
		else
		{
			array1d<T> xc = make_copy(x);
			vecfunc(n, xc.pbase());
			x << xc;
		}
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename TVecFunc>
	inline void array_calc_inplace(aview2d<T, TOrd, TIndexer0, TIndexer1>& x, TVecFunc vecfunc)
	{
		size_t n = x.nelems();

		if (is_dense_view(x))
		{
			vecfunc(n, x.pbase());
		}
		else
		{
			array2d<T, TOrd> xc = make_copy(x);
			vecfunc(n, xc.pbase());
			x << xc;
		}
	}


	template<typename T, class YIndexer, class XIndexer, typename TVecFunc>
	inline void array_calc_inplace(aview1d<T, YIndexer>& y, const const_aview1d<T, XIndexer>& x, TVecFunc vecfunc)
	{
		if (! is_same_shape(y, x) )
		{
			throw array_size_mismatch();
		}

		size_t n = y.nelems();

		if (is_dense_view(y))
		{
			if (is_dense_view(x))
			{
				vecfunc(n, y.pbase(), x.pbase());
			}
			else
			{
				array1d<T> xc = make_copy(x);
				vecfunc(n, y.pbase(), xc.pbase());
			}
		}
		else
		{
			array1d<T> yc = make_copy(y);
			if (is_dense_view(x))
			{
				vecfunc(n, yc.pbase(), x.pbase());
			}
			else
			{
				array1d<T> xc = make_copy(x);
				vecfunc(n, yc.pbase(), xc.pbase());
			}
			y << yc;
		}
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1, class XIndexer0, class XIndexer1, typename TVecFunc>
	inline void array_calc_inplace(
			aview2d<T, TOrd, YIndexer0, YIndexer1>& y,
			const const_aview2d<T, TOrd, XIndexer0, XIndexer1>& x,
			TVecFunc vecfunc)
	{
		if (! is_same_shape(y, x) )
		{
			throw array_size_mismatch();
		}

		size_t n = y.nelems();

		if (is_dense_view(y))
		{
			if (is_dense_view(x))
			{
				vecfunc(n, y.pbase(), x.pbase());
			}
			else
			{
				array2d<T, TOrd> xc = make_copy(x);
				vecfunc(n, y.pbase(), xc.pbase());
			}
		}
		else
		{
			array2d<T, TOrd> yc = make_copy(y);
			if (is_dense_view(x))
			{
				vecfunc(n, yc.pbase(), x.pbase());
			}
			else
			{
				array2d<T, TOrd> xc = make_copy(x);
				vecfunc(n, yc.pbase(), xc.pbase());
			}
			y << yc;
		}
	}


	template<typename T, class YIndexer, typename TVecFunc>
	inline void array_calc_inplace(aview1d<T, YIndexer>& y, const T& x, TVecFunc vecfunc)
	{
		size_t n = y.nelems();

		if (is_dense_view(y))
		{
			vecfunc(n, y.pbase(), x);
		}
		else
		{
			array1d<T> yc = make_copy(y);
			vecfunc(n, yc.pbase(), x);
			y << yc;
		}
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1, typename TVecFunc>
	inline void array_calc_inplace(aview2d<T, TOrd, YIndexer0, YIndexer1>& y, const T& x, TVecFunc vecfunc)
	{
		size_t n = y.nelems();

		if (is_dense_view(y))
		{
			vecfunc(n, y.pbase(), x);
		}
		else
		{
			array2d<T, TOrd> yc = make_copy(y);
			vecfunc(n, yc.pbase(), x);
			y << yc;
		}
	}


	// Calculation operators overloading: add, sub, mul, div, negate

	// add

	template<typename T>
	struct vec_add_fun
	{
		void operator() (size_t n, const T *x1, const T *x2, T *y)
		{
			vec_add(n, x1, x2, y);
		}

		void operator() (size_t n, const T *x1, const T& x2, T *y)
		{
			vec_add(n, x1, x2, y);
		}

		void operator() (size_t n, T *y, const T* x)
		{
			vec_add_inplace(n, y, x);
		}

		void operator() (size_t n, T *y, const T& x)
		{
			vec_add_inplace(n, y, x);
		}
	};


	template<typename T, class LIndexer, class RIndexer>
	array1d<T> operator + (const const_aview1d<T, LIndexer>& x1, const const_aview1d<T, RIndexer>& x2)
	{
		return array_calc(x1, x2, vec_add_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> operator + (
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2)
	{
		return array_calc(x1, x2, vec_add_fun<T>());
	}

	template<typename T, class LIndexer>
	array1d<T> operator + (const const_aview1d<T, LIndexer>& x1, const T& x2)
	{
		return array_calc(x1, x2, vec_add_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	array2d<T, TOrd> operator + (const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1, const T& x2)
	{
		return array_calc(x1, x2, vec_add_fun<T>());
	}

	template<typename T, class RIndexer>
	array1d<T> operator + (const T& x1, const const_aview1d<T, RIndexer>& x2)
	{
		return array_calc(x2, x1, vec_add_fun<T>());
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> operator + (const T& x1, const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2)
	{
		return array_calc(x2, x1, vec_add_fun<T>());
	}

	template<typename T, class YIndexer, class XIndexer>
	aview1d<T, YIndexer>& operator += (aview1d<T, YIndexer>& y, const const_aview1d<T, XIndexer>& x)
	{
		array_calc_inplace(y, x, vec_add_fun<T>());
		return y;
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1, class XIndexer0, class XIndexer1>
	aview2d<T, TOrd, YIndexer0, YIndexer1>& operator += (
			aview2d<T, TOrd, YIndexer0, YIndexer1>& y,
			const const_aview2d<T, TOrd, XIndexer0, XIndexer1>& x)
	{
		array_calc_inplace(y, x, vec_add_fun<T>());
		return y;
	}

	template<typename T, class YIndexer>
	aview1d<T, YIndexer>& operator += (aview1d<T, YIndexer>& y, const T& x)
	{
		array_calc_inplace(y, x, vec_add_fun<T>());
		return y;
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1>
	aview2d<T, TOrd, YIndexer0, YIndexer1>& operator += (aview2d<T, TOrd, YIndexer0, YIndexer1>& y, const T& x)
	{
		array_calc_inplace(y, x, vec_add_fun<T>());
		return y;
	}


	// sub

	template<typename T>
	struct vec_sub_fun
	{
		void operator() (size_t n, const T *x1, const T *x2, T *y)
		{
			vec_sub(n, x1, x2, y);
		}

		void operator() (size_t n, const T *x1, const T& x2, T *y)
		{
			vec_sub(n, x1, x2, y);
		}

		void operator() (size_t n, const T& x1, const T *x2, T *y)
		{
			vec_sub(n, x1, x2, y);
		}

		void operator() (size_t n, T *y, const T* x)
		{
			vec_sub_inplace(n, y, x);
		}

		void operator() (size_t n, T *y, const T& x)
		{
			vec_sub_inplace(n, y, x);
		}
	};

	template<typename T>
	struct vec_rev_sub_fun
	{
		void operator() (size_t n, T *y, const T& x)
		{
			vec_sub_inplace(n, x, y);
		}
	};


	template<typename T, class LIndexer, class RIndexer>
	array1d<T> operator - (const const_aview1d<T, LIndexer>& x1, const const_aview1d<T, RIndexer>& x2)
	{
		return array_calc(x1, x2, vec_sub_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> operator - (
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2)
	{
		return array_calc(x1, x2, vec_sub_fun<T>());
	}

	template<typename T, class LIndexer>
	array1d<T> operator - (const const_aview1d<T, LIndexer>& x1, const T& x2)
	{
		return array_calc(x1, x2, vec_sub_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	array2d<T, TOrd> operator - (const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1, const T& x2)
	{
		return array_calc(x1, x2, vec_sub_fun<T>());
	}

	template<typename T, class RIndexer>
	array1d<T> operator - (const T& x1, const const_aview1d<T, RIndexer>& x2)
	{
		return array_calc(x1, x2, vec_sub_fun<T>());
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> operator - (const T& x1, const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2)
	{
		return array_calc(x1, x2, vec_sub_fun<T>());
	}

	template<typename T, class YIndexer, class XIndexer>
	aview1d<T, YIndexer>& operator -= (aview1d<T, YIndexer>& y, const const_aview1d<T, XIndexer>& x)
	{
		array_calc_inplace(y, x, vec_sub_fun<T>());
		return y;
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1, class XIndexer0, class XIndexer1>
	aview2d<T, TOrd, YIndexer0, YIndexer1>& operator -= (
			aview2d<T, TOrd, YIndexer0, YIndexer1>& y,
			const const_aview2d<T, TOrd, XIndexer0, XIndexer1>& x)
	{
		array_calc_inplace(y, x, vec_sub_fun<T>());
		return y;
	}

	template<typename T, class YIndexer>
	aview1d<T, YIndexer>& operator -= (aview1d<T, YIndexer>& y, const T& x)
	{
		array_calc_inplace(y, x, vec_sub_fun<T>());
		return y;
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1>
	aview2d<T, TOrd, YIndexer0, YIndexer1>& operator -= (aview2d<T, TOrd, YIndexer0, YIndexer1>& y, const T& x)
	{
		array_calc_inplace(y, x, vec_sub_fun<T>());
		return y;
	}


	template<typename T, class YIndexer>
	void be_subtracted(const T& x, aview1d<T, YIndexer>& y)
	{
		array_calc_inplace(y, x, vec_rev_sub_fun<T>());
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1>
	void be_subtracted(const T& x, aview2d<T, TOrd, YIndexer0, YIndexer1>& y)
	{
		array_calc_inplace(y, x, vec_rev_sub_fun<T>());
	}


	// mul

	template<typename T>
	struct vec_mul_fun
	{
		void operator() (size_t n, const T *x1, const T *x2, T *y)
		{
			vec_mul(n, x1, x2, y);
		}

		void operator() (size_t n, const T *x1, const T& x2, T *y)
		{
			vec_mul(n, x1, x2, y);
		}

		void operator() (size_t n, T *y, const T* x)
		{
			vec_mul_inplace(n, y, x);
		}

		void operator() (size_t n, T *y, const T& x)
		{
			vec_mul_inplace(n, y, x);
		}
	};


	template<typename T, class LIndexer, class RIndexer>
	array1d<T> operator * (const const_aview1d<T, LIndexer>& x1, const const_aview1d<T, RIndexer>& x2)
	{
		return array_calc(x1, x2, vec_mul_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> operator * (
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2)
	{
		return array_calc(x1, x2, vec_mul_fun<T>());
	}

	template<typename T, class LIndexer>
	array1d<T> operator * (const const_aview1d<T, LIndexer>& x1, const T& x2)
	{
		return array_calc(x1, x2, vec_mul_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	array2d<T, TOrd> operator * (const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1, const T& x2)
	{
		return array_calc(x1, x2, vec_mul_fun<T>());
	}

	template<typename T, class RIndexer>
	array1d<T> operator * (const T& x1, const const_aview1d<T, RIndexer>& x2)
	{
		return array_calc(x2, x1, vec_mul_fun<T>());
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> operator * (const T& x1, const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2)
	{
		return array_calc(x2, x1, vec_mul_fun<T>());
	}

	template<typename T, class YIndexer, class XIndexer>
	aview1d<T, YIndexer>& operator *= (aview1d<T, YIndexer>& y, const const_aview1d<T, XIndexer>& x)
	{
		array_calc_inplace(y, x, vec_mul_fun<T>());
		return y;
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1, class XIndexer0, class XIndexer1>
	aview2d<T, TOrd, YIndexer0, YIndexer1>& operator *= (
			aview2d<T, TOrd, YIndexer0, YIndexer1>& y,
			const const_aview2d<T, TOrd, XIndexer0, XIndexer1>& x)
	{
		array_calc_inplace(y, x, vec_mul_fun<T>());
		return y;
	}

	template<typename T, class YIndexer>
	aview1d<T, YIndexer>& operator *= (aview1d<T, YIndexer>& y, const T& x)
	{
		array_calc_inplace(y, x, vec_mul_fun<T>());
		return y;
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1>
	aview2d<T, TOrd, YIndexer0, YIndexer1>& operator *= (aview2d<T, TOrd, YIndexer0, YIndexer1>& y, const T& x)
	{
		array_calc_inplace(y, x, vec_mul_fun<T>());
		return y;
	}


	// div

	template<typename T>
	struct vec_div_fun
	{
		void operator() (size_t n, const T *x1, const T *x2, T *y)
		{
			vec_div(n, x1, x2, y);
		}

		void operator() (size_t n, const T *x1, const T& x2, T *y)
		{
			vec_div(n, x1, x2, y);
		}

		void operator() (size_t n, const T& x1, const T *x2, T *y)
		{
			vec_div(n, x1, x2, y);
		}

		void operator() (size_t n, T *y, const T* x)
		{
			vec_div_inplace(n, y, x);
		}

		void operator() (size_t n, T *y, const T& x)
		{
			vec_div_inplace(n, y, x);
		}
	};

	template<typename T>
	struct vec_rev_div_fun
	{
		void operator() (size_t n, T *y, const T& x)
		{
			vec_div_inplace(n, x, y);
		}
	};


	template<typename T, class LIndexer, class RIndexer>
	array1d<T> operator / (const const_aview1d<T, LIndexer>& x1, const const_aview1d<T, RIndexer>& x2)
	{
		return array_calc(x1, x2, vec_div_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> operator / (
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2)
	{
		return array_calc(x1, x2, vec_div_fun<T>());
	}

	template<typename T, class LIndexer>
	array1d<T> operator / (const const_aview1d<T, LIndexer>& x1, const T& x2)
	{
		return array_calc(x1, x2, vec_div_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1>
	array2d<T, TOrd> operator / (const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1, const T& x2)
	{
		return array_calc(x1, x2, vec_div_fun<T>());
	}

	template<typename T, class RIndexer>
	array1d<T> operator / (const T& x1, const const_aview1d<T, RIndexer>& x2)
	{
		return array_calc(x1, x2, vec_div_fun<T>());
	}

	template<typename T, typename TOrd, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> operator / (const T& x1, const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2)
	{
		return array_calc(x1, x2, vec_div_fun<T>());
	}

	template<typename T, class YIndexer, class XIndexer>
	aview1d<T, YIndexer>& operator /= (aview1d<T, YIndexer>& y, const const_aview1d<T, XIndexer>& x)
	{
		array_calc_inplace(y, x, vec_div_fun<T>());
		return y;
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1, class XIndexer0, class XIndexer1>
	aview2d<T, TOrd, YIndexer0, YIndexer1>& operator /= (
			aview2d<T, TOrd, YIndexer0, YIndexer1>& y,
			const const_aview2d<T, TOrd, XIndexer0, XIndexer1>& x)
	{
		array_calc_inplace(y, x, vec_div_fun<T>());
		return y;
	}

	template<typename T, class YIndexer>
	aview1d<T, YIndexer>& operator /= (aview1d<T, YIndexer>& y, const T& x)
	{
		array_calc_inplace(y, x, vec_div_fun<T>());
		return y;
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1>
	aview2d<T, TOrd, YIndexer0, YIndexer1>& operator /= (aview2d<T, TOrd, YIndexer0, YIndexer1>& y, const T& x)
	{
		array_calc_inplace(y, x, vec_div_fun<T>());
		return y;
	}


	template<typename T, class YIndexer>
	void be_divided(const T& x, aview1d<T, YIndexer>& y)
	{
		array_calc_inplace(y, x, vec_rev_div_fun<T>());
	}

	template<typename T, typename TOrd, class YIndexer0, class YIndexer1>
	void be_divided(const T& x, aview2d<T, TOrd, YIndexer0, YIndexer1>& y)
	{
		array_calc_inplace(y, x, vec_rev_div_fun<T>());
	}


	// negate

	template<typename T>
	struct vec_neg_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_negate(n, x, y);
		}

		void operator() (size_t n, T *x)
		{
			vec_negate(n, x);
		}
	};


	template<typename T, class TIndexer>
	array1d<T> operator - (const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_neg_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> operator - (const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_neg_fun<T>());
	}

	template<typename T, class TIndexer>
	void be_negated(aview1d<T, TIndexer>& x)
	{
		array_calc_inplace(x, vec_neg_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	void be_negated(aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		array_calc_inplace(x, vec_neg_fun<T>());
	}


}

#endif 
