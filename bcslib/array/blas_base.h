/**
 * @file blas.h
 *
 * The basis of BLAS incorporation
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_BLAS_BASE_H_
#define BCSLIB_BLAS_BASE_H_

#include <bcslib/array/array_base.h>
#include <bcslib/array/generic_array_functions.h>

#include <bcslib/base/arg_check.h>
#include <bcslib/extern/blas_select.h>


namespace bcs
{
	namespace blas
	{

		/**************************************************
		 *
		 *  Helper types
		 *
		 **************************************************/

		template<typename T>
		struct cvec
		{
			int n;
			const T *data;
		};

		template<typename T>
		inline cvec<T> make_cvec(int n, const T *data)
		{
			cvec<T> v;
			v.n = n;
			v.data = data;
			return v;
		}

		template<typename T>
		struct vec
		{
			int n;
			T *data;
		};

		template<typename T>
		inline vec<T> make_vec(int n, T *data)
		{
			vec<T> v;
			v.n = n;
			v.data = data;
			return v;
		}


		template<typename T, typename TOrd>
		struct cmat
		{
			int m;
			int n;
			const T *data;
			bool trans;
		};


		template<typename T, typename TOrd>
		inline cmat<T, typename TOrd> make_cvec(int m, int n, const T *data, bool trans, TOrd)
		{
			cmat<T, TOrd> a;
			a.m = m;
			a.n = n;
			a.data = data;
			a.trans = trans;
			return a;
		}


		template<typename T, typename TOrd>
		struct mat
		{
			int m;
			int n;
			T *data;
		};

		template<typename T, typename TOrd>
		inline mat<T, TOrd> make_cvec(int m, int n, T *data, TOrd)
		{
			mat<T, TOrd> a;
			a.m = m;
			a.n = n;
			a.data = data;
			return a;
		}


		/**************************************************
		 *
		 *  generic blas function
		 *  (wrapped with a easier interface)
		 *
		 **************************************************/

		// BLAS Level 1

		inline double asum(const cvec<double>& x)
		{
			double s = 0;
			if (x.n > 0)
			{
				BCS_DASUM(&(x.n), x.data, &s);
			}
			return s;
		}

		inline float asum(const vec<float>& x)
		{
			float s = 0;
			if (x.n > 0)
			{
				BCS_SASUM(&(x.n), x.data, &s);
			}
			return s;
		}

	}

}

#endif 
