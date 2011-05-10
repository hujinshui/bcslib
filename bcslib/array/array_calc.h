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

	/******************************************************
	 *
	 *  Generic functions
	 *
	 ******************************************************/


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
			vecfunc(n, x.begin(), y.pbase());
		}
		return y;
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename TVecFunc>
	inline array2d<T, TOrd> array_calc(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x, TVecFunc vecfunc)
	{
		array2d<T, TOrd> y(x.nrows(), x.ncolumns());

		if (is_dense_view(x))
		{
			vecfunc(x.nelems(), x.pbase(), y.pbase());
		}
		else
		{
			vecfunc(x.nelems(), x.begin(), y.pbase());
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
				vecfunc(n, x1.pbase(), x2.begin(), y.pbase());
			}
		}
		else
		{
			if (is_dense_view(x2))
			{
				vecfunc(n, x1.begin(), x2.pbase(), y.pbase());
			}
			else
			{
				vecfunc(n, x1.begin(), x2.begin(), y.pbase());
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

		array2d<T, TOrd> y(x1.nrows(), x1.ncolumns());
		size_t n = x1.nelems();

		if (is_dense_view(x1))
		{
			if (is_dense_view(x2))
			{
				vecfunc(n, x1.pbase(), x2.pbase(), y.pbase());
			}
			else
			{
				vecfunc(n, x1.pbase(), x2.begin(), y.pbase());
			}
		}
		else
		{
			if (is_dense_view(x2))
			{
				vecfunc(n, x1.begin(), x2.pbase(), y.pbase());
			}
			else
			{
				vecfunc(n, x1.begin(), x2.begin(), y.pbase());
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
			vecfunc(n, x1.begin(), x2, y.pbase());
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
			vecfunc(n, x1, x2.begin(), y.pbase());
		}

		return y;
	}


	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, typename TVecFunc>
	inline array2d<T, TOrd> array_calc(const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1, const T& x2, TVecFunc vecfunc)
	{
		array2d<T, TOrd> y(x1.nrows(), x1.ncolumns());
		size_t n = x1.nelems();

		if (is_dense_view(x1))
		{
			vecfunc(n, x1.pbase(), x2, y.pbase());
		}
		else
		{
			vecfunc(n, x1.begin(), x2, y.pbase());
		}
		return y;
	}


	template<typename T, typename TOrd, class RIndexer0, class RIndexer1, typename TVecFunc>
	inline array2d<T, TOrd> array_calc(const T& x1, const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2, TVecFunc vecfunc)
	{
		array2d<T, TOrd> y(x2.nrows(), x2.ncolumns());
		size_t n = x2.nelems();

		if (is_dense_view(x2))
		{
			vecfunc(n, x1, x2.pbase(), y.pbase());
		}
		else
		{
			vecfunc(n, x1, x2.begin(), y.pbase());
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
			vecfunc(n, x.begin());
		}
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1, typename TVecFunc>
	inline void array_calc_inplace(aview2d<T, TOrd, TIndexer0, TIndexer1>& y, TVecFunc vecfunc)
	{
		size_t n = y.nelems();

		if (is_dense_view(y))
		{
			vecfunc(n, y.pbase());
		}
		else
		{
			vecfunc(n, y.begin());
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
				vecfunc(n, y.pbase(), x.begin());
			}
		}
		else
		{
			if (is_dense_view(x))
			{
				vecfunc(n, y.begin(), x.pbase());
			}
			else
			{
				vecfunc(n, y.begin(), x.begin());
			}
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
				vecfunc(n, y.pbase(), x.begin());
			}
		}
		else
		{
			if (is_dense_view(x))
			{
				vecfunc(n, y.begin(), x.pbase());
			}
			else
			{
				vecfunc(n, y.begin(), x.begin());
			}
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
			vecfunc(n, y.begin(), x);
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
			vecfunc(n, y.begin(), x);
		}
	}


	/******************************************************
	 *
	 * Calculation operators overloading:
	 *
	 *  add, sub, mul, div, negate
	 *
	 ******************************************************/

	// add

	template<typename T>
	struct vec_add_fun
	{
		void operator() (size_t n, const T *x1, const T *x2, T *y)
		{
			vec_add(n, x1, x2, y);
		}

		template<typename TIterX1, typename TIterX2>
		void operator() (size_t n, TIterX1 x1, TIterX2 x2, T* y)
		{
			for (size_t i = 0; i < n; ++i, ++x1, ++x2)
			{
				*(y++) = *x1 + *x2;
			}
		}


		void operator() (size_t n, const T *x1, const T& x2, T *y)
		{
			vec_add(n, x1, x2, y);
		}

		template<typename TIterX1>
		void operator() (size_t n, TIterX1 x1, const T& x2, T *y)
		{
			for (size_t i = 0; i < n; ++i, ++x1)
			{
				*(y++) = *x1 + x2;
			}
		}


		void operator() (size_t n, T *y, const T* x)
		{
			vec_add_inplace(n, y, x);
		}

		template<typename TIterY, typename TIterX>
		void operator() (size_t n, TIterY y, TIterX x)
		{
			for (size_t i = 0; i < n; ++i, ++y, ++x)
			{
				*y += *x;
			}
		}


		void operator() (size_t n, T *y, const T& x)
		{
			vec_add_inplace(n, y, x);
		}

		template<typename TIterY>
		void operator() (size_t n, TIterY y, const T& x)
		{
			for (size_t i = 0; i < n; ++i, ++y)
			{
				*y += x;
			}
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

		template<typename TIterX1, typename TIterX2>
		void operator() (size_t n, TIterX1 x1, TIterX2 x2, T* y)
		{
			for (size_t i = 0; i < n; ++i, ++x1, ++x2)
			{
				*(y++) = *x1 - *x2;
			}
		}


		void operator() (size_t n, const T *x1, const T& x2, T *y)
		{
			vec_sub(n, x1, x2, y);
		}

		template<typename TIterX1>
		void operator() (size_t n, TIterX1 x1, const T& x2, T *y)
		{
			for (size_t i = 0; i < n; ++i, ++x1)
			{
				*(y++) = *x1 - x2;
			}
		}

		void operator() (size_t n, const T& x1, const T *x2, T *y)
		{
			vec_sub(n, x1, x2, y);
		}

		template<typename TIterX2>
		void operator() (size_t n, const T& x1, TIterX2 x2, T *y)
		{
			for (size_t i = 0; i < n; ++i, ++x2)
			{
				*(y++) = x1 - *x2;
			}
		}


		void operator() (size_t n, T *y, const T* x)
		{
			vec_sub_inplace(n, y, x);
		}

		template<typename TIterY, typename TIterX>
		void operator() (size_t n, TIterY y, TIterX x)
		{
			for (size_t i = 0; i < n; ++i, ++y, ++x)
			{
				*y -= *x;
			}
		}


		void operator() (size_t n, T *y, const T& x)
		{
			vec_sub_inplace(n, y, x);
		}

		template<typename TIterY>
		void operator() (size_t n, TIterY y, const T& x)
		{
			for (size_t i = 0; i < n; ++i, ++y)
			{
				*y -= x;
			}
		}
	};


	template<typename T>
	struct vec_rev_sub_fun
	{
		void operator() (size_t n, T *y, const T& x)
		{
			vec_sub_inplace(n, x, y);
		}

		template<typename TIterY>
		void operator() (size_t n, TIterY y, const T& x)
		{
			for (size_t i = 0; i < n; ++i, ++y)
			{
				*y = x - *y;
			}
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

		template<typename TIterX1, typename TIterX2>
		void operator() (size_t n, TIterX1 x1, TIterX2 x2, T* y)
		{
			for (size_t i = 0; i < n; ++i, ++x1, ++x2)
			{
				*(y++) = *x1 * *x2;
			}
		}


		void operator() (size_t n, const T *x1, const T& x2, T *y)
		{
			vec_mul(n, x1, x2, y);
		}

		template<typename TIterX1>
		void operator() (size_t n, TIterX1 x1, const T& x2, T *y)
		{
			for (size_t i = 0; i < n; ++i, ++x1)
			{
				*(y++) = *x1 * x2;
			}
		}


		void operator() (size_t n, T *y, const T* x)
		{
			vec_mul_inplace(n, y, x);
		}

		template<typename TIterY, typename TIterX>
		void operator() (size_t n, TIterY y, TIterX x)
		{
			for (size_t i = 0; i < n; ++i, ++y, ++x)
			{
				*y *= *x;
			}
		}


		void operator() (size_t n, T *y, const T& x)
		{
			vec_mul_inplace(n, y, x);
		}

		template<typename TIterY>
		void operator() (size_t n, TIterY y, const T& x)
		{
			for (size_t i = 0; i < n; ++i, ++y)
			{
				*y *= x;
			}
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

		template<typename TIterX1, typename TIterX2>
		void operator() (size_t n, TIterX1 x1, TIterX2 x2, T* y)
		{
			for (size_t i = 0; i < n; ++i, ++x1, ++x2)
			{
				*(y++) = *x1 / *x2;
			}
		}


		void operator() (size_t n, const T *x1, const T& x2, T *y)
		{
			vec_div(n, x1, x2, y);
		}

		template<typename TIterX1>
		void operator() (size_t n, TIterX1 x1, const T& x2, T *y)
		{
			for (size_t i = 0; i < n; ++i, ++x1)
			{
				*(y++) = *x1 / x2;
			}
		}

		void operator() (size_t n, const T& x1, const T *x2, T *y)
		{
			vec_div(n, x1, x2, y);
		}

		template<typename TIterX2>
		void operator() (size_t n, const T& x1, TIterX2 x2, T *y)
		{
			for (size_t i = 0; i < n; ++i, ++x2)
			{
				*(y++) = x1 / *x2;
			}
		}


		void operator() (size_t n, T *y, const T* x)
		{
			vec_div_inplace(n, y, x);
		}

		template<typename TIterY, typename TIterX>
		void operator() (size_t n, TIterY y, TIterX x)
		{
			for (size_t i = 0; i < n; ++i, ++y, ++x)
			{
				*y /= *x;
			}
		}


		void operator() (size_t n, T *y, const T& x)
		{
			vec_div_inplace(n, y, x);
		}

		template<typename TIterY>
		void operator() (size_t n, TIterY y, const T& x)
		{
			for (size_t i = 0; i < n; ++i, ++y)
			{
				*y /= x;
			}
		}
	};


	template<typename T>
	struct vec_rev_div_fun
	{
		void operator() (size_t n, T *y, const T& x)
		{
			vec_div_inplace(n, x, y);
		}

		template<typename TIterY>
		void operator() (size_t n, TIterY y, const T& x)
		{
			for (size_t i = 0; i < n; ++i, ++y)
			{
				*y = x / *y;
			}
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

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = - *x;
			}
		}

		void operator() (size_t n, T *y)
		{
			vec_negate(n, y);
		}

		template<typename TIterY>
		void operator() (size_t n, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++y)
			{
				*y = -*y;
			}
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


	/******************************************************
	 *
	 *  Elementary functions
	 *
	 ******************************************************/


	// abs

	template<typename T>
	struct vec_abs_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_abs(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::abs(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> abs(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_abs_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> abs(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_abs_fun<T>());
	}


	// sqr

	template<typename T>
	struct vec_sqr_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_sqr(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = sqr(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> sqr(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_sqr_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> sqr(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_sqr_fun<T>());
	}


	// sqrt

	template<typename T>
	struct vec_sqrt_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_sqrt(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::sqrt(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> sqrt(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_sqrt_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> sqrt(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_sqrt_fun<T>());
	}


	// pow

	template<typename T>
	struct vec_pow_fun
	{
		void operator() (size_t n, const T *x, const T *e, T *y)
		{
			vec_pow(n, x, e, y);
		}

		template<typename TIterX, typename TIterE, typename TIterY>
		void operator() (size_t n, TIterX x, TIterE e, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++e, ++y)
			{
				*y = std::pow(*x, *e);
			}
		}

		void operator() (size_t n, const T *x, const T &e, T *y)
		{
			vec_pow(n, x, e, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, const T& e, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::pow(*x, e);
			}
		}
	};

	template<typename T>
	struct vec_pow_n_fun
	{
		int e;
		vec_pow_n_fun(int e_) : e(e_) { }

		void operator() (size_t n, const T *x, T *y)
		{
			vec_pow_n(n, x, e, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::pow(*x, e);
			}
		}
	};

	template<typename T, class LIndexer, class RIndexer>
	array1d<T> pow(const const_aview1d<T, LIndexer>& x, const const_aview1d<T, RIndexer>& e)
	{
		return array_calc(x, e, vec_pow_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> pow(
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& e)
	{
		return array_calc(x, e, vec_pow_fun<T>());
	}


	template<typename T, class TIndexer>
	array1d<T> pow(const const_aview1d<T, TIndexer>& x, const T& e)
	{
		return array_calc(x, e, vec_pow_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> pow(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x, const T &e)
	{
		return array_calc(x, e, vec_pow_fun<T>());
	}

	template<typename T, class TIndexer>
	array1d<T> pow_n(const const_aview1d<T, TIndexer>& x, const int& e)
	{
		return array_calc(x, vec_pow_n_fun<T>(e));
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> pow_n(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x, const int& e)
	{
		return array_calc(x, vec_pow_n_fun<T>(e));
	}


	// exp

	template<typename T>
	struct vec_exp_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_exp(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::exp(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> exp(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_exp_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> exp(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_exp_fun<T>());
	}


	// log

	template<typename T>
	struct vec_log_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_log(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::log(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> log(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_log_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> log(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_log_fun<T>());
	}


	// log10

	template<typename T>
	struct vec_log10_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_log10(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::log10(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> log10(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_log10_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> log10(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_log10_fun<T>());
	}


	// ceil

	template<typename T>
	struct vec_ceil_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_ceil(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::ceil(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> ceil(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_ceil_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> ceil(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_ceil_fun<T>());
	}


	// floor

	template<typename T>
	struct vec_floor_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_floor(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::floor(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> floor(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_floor_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> floor(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_floor_fun<T>());
	}


	// sin

	template<typename T>
	struct vec_sin_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_sin(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::sin(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> sin(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_sin_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> sin(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_sin_fun<T>());
	}


	// cos

	template<typename T>
	struct vec_cos_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_cos(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::cos(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> cos(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_cos_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> cos(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_cos_fun<T>());
	}


	// tan

	template<typename T>
	struct vec_tan_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_tan(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::tan(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> tan(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_tan_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> tan(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_tan_fun<T>());
	}


	// asin

	template<typename T>
	struct vec_asin_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_asin(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::asin(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> asin(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_asin_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> asin(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_asin_fun<T>());
	}


	// acos

	template<typename T>
	struct vec_acos_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_acos(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::acos(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> acos(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_acos_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> acos(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_acos_fun<T>());
	}


	// atan

	template<typename T>
	struct vec_atan_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_atan(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::atan(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> atan(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_atan_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> atan(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_atan_fun<T>());
	}


	// atan2

	template<typename T>
	struct vec_atan2_fun
	{
		void operator() (size_t n, const T *x1, const T *x2, T *y)
		{
			vec_atan2(n, x1, x2, y);
		}

		template<typename TIterX1, typename TIterX2, typename TIterY>
		void operator() (size_t n, TIterX1 x1, TIterX2 x2, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x1, ++x2, ++y)
			{
				*y = std::atan2(*x1, *x2);
			}
		}
	};

	template<typename T, class LIndexer, class RIndexer>
	array1d<T> atan2(const const_aview1d<T, LIndexer>& x1, const const_aview1d<T, RIndexer>& x2)
	{
		return array_calc(x1, x2, vec_atan2_fun<T>());
	}

	template<typename T, typename TOrd, class LIndexer0, class LIndexer1, class RIndexer0, class RIndexer1>
	array2d<T, TOrd> atan2(
			const const_aview2d<T, TOrd, LIndexer0, LIndexer1>& x1,
			const const_aview2d<T, TOrd, RIndexer0, RIndexer1>& x2)
	{
		return array_calc(x1, x2, vec_atan2_fun<T>());
	}


	// sinh

	template<typename T>
	struct vec_sinh_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_sinh(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::sinh(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> sinh(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_sinh_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> sinh(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_sinh_fun<T>());
	}


	// cosh

	template<typename T>
	struct vec_cosh_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_cosh(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::cosh(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> cosh(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_cosh_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> cosh(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_cosh_fun<T>());
	}


	// tanh

	template<typename T>
	struct vec_tanh_fun
	{
		void operator() (size_t n, const T *x, T *y)
		{
			vec_tanh(n, x, y);
		}

		template<typename TIterX, typename TIterY>
		void operator() (size_t n, TIterX x, TIterY y)
		{
			for (size_t i = 0; i < n; ++i, ++x, ++y)
			{
				*y = std::tanh(*x);
			}
		}
	};

	template<typename T, class TIndexer>
	array1d<T> tanh(const const_aview1d<T, TIndexer>& x)
	{
		return array_calc(x, vec_tanh_fun<T>());
	}

	template<typename T, typename TOrd, class TIndexer0, class TIndexer1>
	array2d<T, TOrd> tanh(const const_aview2d<T, TOrd, TIndexer0, TIndexer1>& x)
	{
		return array_calc(x, vec_tanh_fun<T>());
	}


}

#endif 
