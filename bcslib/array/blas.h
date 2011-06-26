/**
 * @file blas.h
 *
 * Classes that wrap the BLAS functions
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_BLAS_H_
#define BCSLIB_BLAS_H_

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


		template<typename T>
		struct cmat
		{
			int m;
			int n;
			const T *data;
			bool trans;
		};

		template<typename T>
		inline cmat<T> make_cvec(int m, int n, const T *data, bool trans)
		{
			cmat<T> a;
			a.m = m;
			a.n = n;
			a.data = data;
			a.trans = trans;
			return a;
		}


		template<typename T>
		struct mat
		{
			int m;
			int n;
			T *data;
		};

		template<typename T>
		inline mat<T> make_cvec(int m, int n, T *data)
		{
			mat<T> a;
			a.m = m;
			a.n = n;
			a.data = data;
			return a;
		}



		/**************************************************
		 *
		 *  Referenced implementation
		 *
		 **************************************************/

		// BLAS Level 1

		template<typename T>
		T asum_rimp(const cvec<T>& x)
		{
			T s = 0;
			for (int i = 0; i < x.n; ++i)
			{
				s += std::abs(x.data[i]);
			}
			return s;
		}

		template<typename T>
		void axpy_rimp(const cvec<T>& x, const T& a, vec<T>& y)
		{
			check_arg(x.n == y.n, "blas::axpy: inconsistent dimensions");

			for (int i = 0; i < x.n; ++i)
			{
				y.data[i] += a * x.data[i];
			}
		}

		template<typename T>
		T dot_rimp(const cvec<T>& x, const cvec<T>& y)
		{
			check_arg(x.n == y.n, "blas::dot: inconsistent dimensions");

			T s = 0;
			for (int i = 0; i < x.n; ++i)
			{
				s += x.data[i] * y.data[i];
			}
			return s;
		}

		template<typename T>
		T nrm2_rimp(const cvec<T>& x)
		{
			T s = 0;
			for (int i = 0; i < x.n; ++i)
			{
				T c = x.data[i];
				s += c * c;
			}
			return std::sqrt(s);
		}

		template<typename T>
		void rot_rimp(vec<T>& x, vec<T>& y, const T& c, const T& s)
		{
			check_arg(x.n == y.n, "blas::rot: inconsistent dimensions");

			for (int i = 0; i < x.n; ++i)
			{
				T r1 = c * x.data[i] + s * y.data[i];
				T r2 = c * y.data[i] - s * x.data[i];

				x.data[i] = r1;
				y.data[i] = r2;
			}
		}


		// BLAS Level 2

		namespace _detail
		{
			template<typename T>
			BCS_FORCE_INLINE void _vdot(int n, const T *x, int incx, const T *y, int incy)
			{
				T s = 0;

				for (int i = 0; i < n; ++i, x += incx, y += incy)
				{
					s += (*x) * (*y);
				}
			}
		}

		template<typename T>
		void gemv_rimp(const cmat<T>& a, const cvec<T>& x, vec<T>& y, const T& alpha, const T& beta)
		{
			int m = a.m;
			int n = a.n;

			if (a.trans)
			{
				check_arg(m == x.n && n == y.n, "blas::gemv: inconsistent dimensions");
			}
			else
			{
				check_arg(n == x.n && m == y.n, "blas::gemv: inconsistent dimensions");
			}

			if (beta == 0)
			{
				if (a.trans)
				{
					for (int i = 0; i < n; ++i)
					{
						y.data[i] = alpha * _detail::_vdot(m, a.data + i * m, 1, x.data, 1);
					}
				}
				else
				{
					for (int i = 0; i < m; ++i)
					{
						y.data[i] = alpha * _detail::_vdot(n, a.data + i, m, x.data, 1);
					}
				}
			}
			else
			{
				if (a.trans)
				{
					for (int i = 0; i < n; ++i)
					{
						y.data[i] = alpha * _detail::_vdot(m, a.data + i * m, 1, x.data, 1) + beta * y.data[i];
					}
				}
				else
				{
					for (int i = 0; i < m; ++i)
					{
						y.data[i] = alpha * _detail::_vdot(n, a.data + i, m, x.data, 1) + beta * y.data[i];
					}
				}
			}
		}

		template<typename T>
		void ger_rimp(mat<T>& a, const cvec<T>& x, const cvec<T>& y, const T& alpha)
		{
			check_arg(a.m == x.n && a.n == y.n, "blas::ger: inconsistent dimensions");

			int m = a.m;
			int n = a.n;
			T *pd = a.data;

			for (int j = 0; j < n; ++j)
			{
				for (int i = 0; i < m; ++i)
				{
					*(pd++) += x.data[i] * y.data[j];
				}
			}
		}


		template<typename T>
		inline void symv_rimp(const cmat<T>& a, const cvec<T>& x, vec<T>& y, const T& alpha, const T& beta)
		{
			gemv_rimp(a, x, y, alpha, beta);
		}

		template<typename T>
		inline void syr_rimp(mat<T>& a, const cvec<T>& x, const T& alpha)
		{
			syr_rimp(a, x, x, alpha);
		}


		// BLAS Level 3


	}

}

#endif 
