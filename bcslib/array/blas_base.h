/**
 * @file blas.h
 *
 * The basis of BLAS incorporation
 * 
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BLAS_BASE_H_
#define BCSLIB_BLAS_BASE_H_

#include <bcslib/array/array_base.h>

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
			char trans;
		};


		template<typename T, typename TOrd>
		inline cmat<T, TOrd> make_cmat(int m, int n, const T *data, char trans, TOrd)
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

		inline void _axpy(cvec<double> x, vec<double> y, double alpha)
		{
			check_arg(x.n == y.n, "blas::_axpy: inconsistent dimensions");

			int n = x.n;
			if (n > 0)
			{
				int incx = 1;
				int incy = 1;
				BCS_DAXPY(&n, &alpha, x.data, &incx, y.data, &incy);
			}
		}


		inline void _axpy(cvec<float> x, vec<float> y, float alpha)
		{
			check_arg(x.n == y.n, "blas::_axpy: inconsistent dimensions");

			int n = x.n;
			if (n > 0)
			{
				int incx = 1;
				int incy = 1;
				BCS_SAXPY(&n, &alpha, x.data, &incx, y.data, &incy);
			}
		}

		// dot

		inline double _dot(cvec<double> x, cvec<double> y)
		{
			check_arg(x.n == y.n, "blas::_dot: inconsistent dimensions");

			double s = 0;
			int n = x.n;
			if (n > 0)
			{
				int incx = 1;
				int incy = 1;
				s = BCS_DDOT(&n, x.data, &incx, y.data, &incy);
			}
			return s;
		}

		inline float _dot(cvec<float> x, cvec<float> y)
		{
			check_arg(x.n == y.n, "blas::_dot: inconsistent dimensions");

			float s = 0;
			int n = x.n;
			if (n > 0)
			{
				int incx = 1;
				int incy = 1;
				s = BCS_SDOT(&n, x.data, &incx, y.data, &incy);
			}
			return s;
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
			check_arg(x.n == y.n, "blas::_rot: inconsistent dimensions");

			int n = x.n;
			if (n > 0)
			{
				int incx = 1;
				int incy = 1;
				BCS_DROT(&n, x.data, &incx, y.data, &incy, &c, &s);
			}
		}

		inline void _rot(vec<float> x, vec<float> y, float c, float s)
		{
			check_arg(x.n == y.n, "blas::_rot: inconsistent dimensions");

			int n = x.n;
			if (n > 0)
			{
				int incx = 1;
				int incy = 1;
				BCS_SROT(&n, x.data, &incx, y.data, &incy, &c, &s);
			}
		}


		// BLAS Level 2

		namespace _detail
		{
			template<typename T>
			inline cmat<T, column_major_t> fortranize(const cmat<T, row_major_t>& a)
			{
				cmat<T, column_major_t> t;
				t.m = a.n;
				t.n = a.m;
				t.data = a.data;
				t.trans = (a.trans == 'N' ? 'T' : 'N');
				return t;
			}

			template<typename T>
			inline cmat<T, column_major_t> fortranize_trans(const cmat<T, row_major_t>& a)
			{
				cmat<T, column_major_t> t;
				t.m = a.n;
				t.n = a.m;
				t.data = a.data;
				t.trans = a.trans;
				return t;
			}

			template<typename T>
			inline mat<T, column_major_t> fortranize(const mat<T, row_major_t>& a)
			{
				mat<T, column_major_t> t;
				t.m = a.n;
				t.n = a.m;
				t.data = a.data;
				return t;
			}
		}

		inline char check_trans(char trans)
		{
			if (trans == 'N' || trans == 'n') return 'N';
			else if (trans == 'T' || trans == 't') return 'T';
			else throw std::invalid_argument("blas: invalid character for trans");
		}

		// gemv

		inline void _gemv(cmat<double, column_major_t> a, cvec<double> x, vec<double> y, double alpha, double beta)
		{
			char trans = check_trans(a.trans);
			int inner_dim = (a.trans == 'N' ? a.n : a.m);
			check_arg(inner_dim == x.n, "blas::_gemv: inconsistent dimensions");

			if (a.m > 0 && a.n > 0)
			{
				int incx = 1;
				int incy = 1;
				BCS_DGEMV(&trans, &(a.m), &(a.n), &alpha, a.data, &(a.m), x.data, &incx, &beta, y.data, &incy);
			}
		}

		inline void _gemv(cmat<float, column_major_t> a, cvec<float> x, vec<float> y, float alpha, float beta)
		{
			char trans = check_trans(a.trans);
			int inner_dim = (a.trans == 'N' ? a.n : a.m);
			check_arg(inner_dim == x.n, "blas::_gemv: inconsistent dimensions");

			if (a.m > 0 && a.n > 0)
			{
				int incx = 1;
				int incy = 1;
				BCS_SGEMV(&trans, &(a.m), &(a.n), &alpha, a.data, &(a.m), x.data, &incx, &beta, y.data, &incy);
			}
		}

		inline void _gemv(cmat<double, row_major_t> a, cvec<double> x, vec<double> y, double alpha, double beta)
		{
			_gemv(_detail::fortranize(a), x, y, alpha, beta);
		}

		inline void _gemv(cmat<float, row_major_t> a, cvec<float> x, vec<float> y, float alpha, float beta)
		{
			_gemv(_detail::fortranize(a), x, y, alpha, beta);
		}


		// ger

		inline void _ger(mat<double, column_major_t> a, cvec<double> x, cvec<double> y, double alpha)
		{
			check_arg(a.m == x.n && a.n == y.n, "blas::_ger: inconsistent dimensions");

			if (a.m > 0 && a.n > 0)
			{
				int incx = 1;
				int incy = 1;
				BCS_DGER(&(a.m), &(a.n), &alpha, x.data, &incx, y.data, &incy, a.data, &(a.m));
			}
		}

		inline void _ger(mat<float, column_major_t> a, cvec<float> x, cvec<float> y, float alpha)
		{
			check_arg(a.m == x.n && a.n == y.n, "blas::_ger: inconsistent dimensions");

			if (a.m > 0 && a.n > 0)
			{
				int incx = 1;
				int incy = 1;
				BCS_SGER(&(a.m), &(a.n), &alpha, x.data, &incx, y.data, &incy, a.data, &(a.m));
			}
		}

		inline void _ger(mat<double, row_major_t> a, cvec<double> x, cvec<double> y, double alpha)
		{
			_ger(_detail::fortranize(a), y, x, alpha);
		}

		inline void _ger(mat<float, row_major_t> a, cvec<float> x, cvec<float> y, float alpha)
		{
			_ger(_detail::fortranize(a), y, x, alpha);
		}

		// symv

		inline void _symv(cmat<double, column_major_t> a, cvec<double> x, vec<double> y, double alpha, double beta)
		{
			int n = a.m;
			check_arg(n == a.n, "blas::_symv: a must be a square matrix");
			check_arg(n == x.n, "blas::_symv: inconsistent dimensions");

			if (n > 0)
			{
				char uplo = 'U';
				int incx = 1;
				int incy = 1;
				BCS_DSYMV(&uplo, &n, &alpha, a.data, &n, x.data, &incx, &beta, y.data, &incy);
			}
		}

		inline void _symv(cmat<float, column_major_t> a, cvec<float> x, vec<float> y, float alpha, float beta)
		{
			int n = a.m;
			check_arg(n == a.n, "blas::_symv: a must be a square matrix");
			check_arg(n == x.n, "blas::_symv: inconsistent dimensions");

			if (n > 0)
			{
				char uplo = 'U';
				int incx = 1;
				int incy = 1;
				BCS_SSYMV(&uplo, &n, &alpha, a.data, &n, x.data, &incx, &beta, y.data, &incy);
			}
		}

		inline void _symv(cmat<double, row_major_t> a, cvec<double> x, vec<double> y, double alpha, double beta)
		{
			_symv(_detail::fortranize(a), x, y, alpha, beta);
		}

		inline void _symv(cmat<float, row_major_t> a, cvec<float> x, vec<float> y, float alpha, float beta)
		{
			_symv(_detail::fortranize(a), x, y, alpha, beta);
		}


		// BLAS Level 3

		// gemm

		inline void _gemm(cmat<double, column_major_t> a, cmat<double, column_major_t> b, mat<double, column_major_t> c,
				double alpha, double beta)
		{
			char transa = check_trans(a.trans);
			char transb = check_trans(b.trans);

			int m, n, k, kb;
			if (transa == 'N') { m = a.m; k = a.n; } else { m = a.n; k = a.m; }
			if (transb == 'N') { n = b.n; kb = b.m; } else { n = b.m; kb = b.n; }
			check_arg(k == kb, "blas::_gemm: inconsistent dimensions");
			check_arg(m == c.m && n == c.n, "blas::_gemm: inconsistent dimensions");

			if (m > 0 && n > 0 && k > 0)
			{
				BCS_DGEMM(&transa, &transb, &m, &n, &k, &alpha, a.data, &(a.m), b.data, &(b.m), &beta, c.data, &(c.m));
			}
		}

		inline void _gemm(cmat<float, column_major_t> a, cmat<float, column_major_t> b, mat<float, column_major_t> c,
				float alpha, float beta)
		{
			char transa = check_trans(a.trans);
			char transb = check_trans(b.trans);

			int m, n, k, kb;
			if (transa == 'N') { m = a.m; k = a.n; } else { m = a.n; k = a.m; }
			if (transb == 'N') { n = b.n; kb = b.m; } else { n = b.m; kb = b.n; }
			check_arg(k == kb, "blas::_gemm: inconsistent dimensions");
			check_arg(m == c.m && n == c.n, "blas::_gemm: inconsistent dimensions");

			if (m > 0 && n > 0 && k > 0)
			{
				BCS_SGEMM(&transa, &transb, &m, &n, &k, &alpha, a.data, &(a.m), b.data, &(b.m), &beta, c.data, &(c.m));
			}
		}

		inline void _gemm(cmat<double, row_major_t> a, cmat<double, row_major_t> b, mat<double, row_major_t> c,
				double alpha, double beta)
		{
			_gemm(_detail::fortranize_trans(b), _detail::fortranize_trans(a), _detail::fortranize(c), alpha, beta);
		}

		inline void _gemm(cmat<float, row_major_t> a, cmat<float, row_major_t> b, mat<float, row_major_t> c,
				float alpha, float beta)
		{
			_gemm(_detail::fortranize_trans(b), _detail::fortranize_trans(a), _detail::fortranize(c), alpha, beta);
		}

	}

}

#endif 
