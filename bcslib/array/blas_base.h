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
		inline cmat<T, TOrd> make_cmat(int m, int n, const T *data, bool trans, TOrd)
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
		inline mat<T, TOrd> make_mat(int m, int n, T *data, TOrd)
		{
			mat<T, TOrd> a;
			a.m = m;
			a.n = n;
			a.data = data;
			return a;
		}


		/**************************************************
		 *
		 *  base blas function
		 *  (wrapped with a easier interface)
		 *
		 **************************************************/

		// BLAS Level 1

		// asum

		inline double _asum(cvec<double> x)
		{
			double s = 0;
			int incx = 1;
			if (x.n > 0)
			{
				s = BCS_DASUM(&(x.n), x.data, &incx);
			}
			return s;
		}

		inline float _asum(cvec<float> x)
		{
			float s = 0;
			int incx = 1;
			if (x.n > 0)
			{
				s = BCS_SASUM(&(x.n), x.data, &incx);
			}
			return s;
		}

		// axpy

		inline void _axpy(double alpha, cvec<double> x, vec<double> y)
		{
			check_arg(x.n == y.n, "blas::_axpy: inconsistent dimensions");

			int n = x.n;
			int incx = 1;
			int incy = 1;

			BCS_DAXPY(&n, &alpha, x.data, &incx, y.data, &incy);
		}


		inline void _axpy(float alpha, cvec<float> x, vec<float> y)
		{
			check_arg(x.n == y.n, "blas::_axpy: inconsistent dimensions");

			int n = x.n;
			int incx = 1;
			int incy = 1;

			BCS_SAXPY(&n, &alpha, x.data, &incx, y.data, &incy);
		}

		// dot

		inline double _dot(cvec<double> x, cvec<double> y)
		{
			check_arg(x.n == y.n, "blas::_dot: inconsistent dimensions");

			int n = x.n;
			int incx = 1;
			int incy = 1;

			return BCS_DDOT(&n, x.data, &incx, y.data, &incy);
		}

		inline float _dot(cvec<float> x, cvec<float> y)
		{
			check_arg(x.n == y.n, "blas::_dot: inconsistent dimensions");

			int n = x.n;
			int incx = 1;
			int incy = 1;

			return BCS_SDOT(&n, x.data, &incx, y.data, &incy);
		}

		// nrm2

		inline double _nrm2(cvec<double> x)
		{
			double s = 0;
			int incx = 1;
			if (x.n > 0)
			{
				s = BCS_DNRM2(&(x.n), x.data, &incx);
			}
			return s;
		}

		inline float _nrm2(cvec<float> x)
		{
			float s = 0;
			int incx = 1;
			if (x.n > 0)
			{
				s = BCS_SNRM2(&(x.n), x.data, &incx);
			}
			return s;
		}

		// rot

		inline void _rot(vec<double> x, vec<double> y, double c, double s)
		{
			check_arg(x.n == y.n, "blas::_dot: inconsistent dimensions");

			int n = x.n;
			int incx = 1;
			int incy = 1;

			BCS_DROT(&n, x.data, &incx, y.data, &incy, &c, &s);
		}

		inline void _rot(vec<float> x, vec<float> y, float c, float s)
		{
			check_arg(x.n == y.n, "blas::_dot: inconsistent dimensions");

			int n = x.n;
			int incx = 1;
			int incy = 1;

			BCS_SROT(&n, x.data, &incx, y.data, &incy, &c, &s);
		}

	}

}

#endif 
