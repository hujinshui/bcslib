/**
 * @file blas.h
 *
 * Wrapped BLAS functions
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BLAS_H_
#define BCSLIB_BLAS_H_

#include <bcslib/core/basic_defs.h>
#include <bcslib/engine/blas_extern.h>

namespace bcs { namespace engine {

	/********************************************
	 *
	 *  Level 1 functors
	 *
	 ********************************************/

	// asum

	template<typename T, int N> struct asum;

	template<int N>
	struct asum<double, N>
	{
		BCS_ENSURE_INLINE
		static double eval(const int n, const double *x)
		{
			const int incx = 1;
			return BCS_DASUM(&n, x, &incx);
		}
	};

	template<int N>
	struct asum<float, N>
	{
		BCS_ENSURE_INLINE
		static float eval(const int n, const float *x)
		{
			const int incx = 1;
			return BCS_SASUM(&n, x, &incx);
		}
	};

	// axpy

	template<typename T, int N> struct axpy;

	template<int N>
	struct axpy<double, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const int n, double a, const double *x, double *y)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_DAXPY(&n, &a, x, &incx, y, &incy);
		}
	};

	template<int N>
	struct axpy<float, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const int n, float a, const float *x, float *y)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_SAXPY(&n, &a, x, &incx, y, &incy);
		}
	};

	// dot

	template<typename T, int N> struct dot;

	template<int N>
	struct dot<double, N>
	{
		BCS_ENSURE_INLINE
		static double eval(const int n, const double *x, const double *y)
		{
			const int incx = 1;
			const int incy = 1;

			return BCS_DDOT(&n, x, &incx, y, &incy);
		}
	};

	template<int N>
	struct dot<float, N>
	{
		BCS_ENSURE_INLINE
		static float eval(const int n, const float *x, const float *y)
		{
			const int incx = 1;
			const int incy = 1;

			return BCS_SDOT(&n, x, &incx, y, &incy);
		}
	};

	// nrm2

	template<typename T, int N> struct nrm2;

	template<int N>
	struct nrm2<double, N>
	{
		BCS_ENSURE_INLINE
		static double eval(const int n, const double *x)
		{
			const int incx = 1;
			return BCS_DNRM2(&n, x, &incx);
		}
	};

	template<int N>
	struct nrm2<float, N>
	{
		BCS_ENSURE_INLINE
		static float eval(const int n, const float *x)
		{
			const int incx = 1;
			return BCS_SNRM2(&n, x, &incx);
		}
	};

	// rot

	template<typename T, int N> struct rot;

	template<int N>
	struct rot<double, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const int n, double *x, double *y, double c, double s)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_DROT(&n, x, &incx, y, &incy, &c, &s);
		}
	};

	template<int N>
	struct rot<float, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const int n, float *x, float *y, float c, float s)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_SROT(&n, x, &incx, y, &incy, &c, &s);
		}
	};


	/********************************************
	 *
	 *  Level 2 functors
	 *
	 ********************************************/

	// gemv

	template<typename T, int M, int N> struct gemv;
	template<typename T, int M, int N> struct gemv_ex;

	template<int M, int N>
	struct gemv<double, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(char trans, const int m, const int n,
				const double alpha, const double *a, const int lda, const double *x,
				const double beta, double *y)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_DGEMV(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
		}
	};

	template<int M, int N>
	struct gemv<float, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(char trans, const int m, const int n,
				const float alpha, const float *a, int lda, const float *x,
				const float beta, float *y)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_SGEMV(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
		}
	};

	template<int M, int N>
	struct gemv_ex<double, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(char trans, const int m, const int n,
				const double alpha, const double *a, const int lda, const double *x, const int incx,
				const double beta, double *y, const int incy)
		{
			BCS_DGEMV(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
		}
	};

	template<int M, int N>
	struct gemv_ex<float, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(char trans, const int m, const int n,
				const float alpha, const float *a, int lda, const float *x, const int incx,
				const float beta, float *y, const int incy)
		{
			BCS_SGEMV(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
		}
	};



	// ger

	template<typename T, int M, int N> struct ger;

	template<int M, int N>
	struct ger<double, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const int m, const int n,
				const double alpha, const double *x, const double *y,
				double *a, const int lda)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_DGER(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
		}
	};


	template<int M, int N>
	struct ger<float, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const int m, const int n,
				const float alpha, const float *x, const float *y,
				float *a, const int lda)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_SGER(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
		}
	};


	// symv

	template<typename T, int N> struct symv;

	template<int N>
	struct symv<double, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char uplo, int n,
				const double alpha, const double *a, const int lda, const double *x,
				const double beta, double *y)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_DSYMV(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
		}
	};


	template<int N>
	struct symv<float, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char uplo, int n,
				const float alpha, const float *a, const int lda, const float *x,
				const float beta, float *y)
		{
			const int incx = 1;
			const int incy = 1;

			BCS_SSYMV(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
		}
	};


	// symv_ex

	template<typename T, int N> struct symv_ex;

	template<int N>
	struct symv_ex<double, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char uplo, int n,
				const double alpha, const double *a, const int lda, const double *x, const int incx,
				const double beta, double *y, const int incy)
		{
			BCS_DSYMV(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
		}
	};

	template<int N>
	struct symv_ex<float, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char uplo, int n,
				const float alpha, const float *a, const int lda, const float *x, const int incx,
				const float beta, float *y, const int incy)
		{
			BCS_SSYMV(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
		}
	};



	// trmv

	template<typename T, int N> struct trmv;

	template<int N>
	struct trmv<double, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char uplo, const char transa, const char diag, int n,
				const double *a, const int lda, double *b)
		{
			const int incx = 1;
			BCS_DTRMV(&uplo, &transa, &diag, &n, a, &lda, b, &incx);
		}
	};

	template<int N>
	struct trmv<float, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char uplo, const char transa, const char diag, int n,
				const float *a, const int lda, float *b)
		{
			const int incx = 1;
			BCS_STRMV(&uplo, &transa, &diag, &n, a, &lda, b, &incx);
		}
	};


	// trsv

	template<typename T, int N> struct trsv;

	template<int N>
	struct trsv<double, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char uplo, const char trans, const char diag, int n,
				const double *a, const int lda, double *x)
		{
			const int incx = 1;
			BCS_DTRSV(&uplo, &trans, &diag, &n, a, &lda, x, &incx);
		}
	};


	template<int N>
	struct trsv<float, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char uplo, const char trans, const char diag, int n,
				const float *a, const int lda, float *x)
		{
			const int incx = 1;
			BCS_STRSV(&uplo, &trans, &diag, &n, a, &lda, x, &incx);
		}
	};


	/********************************************
	 *
	 *  Level 3 functors
	 *
	 ********************************************/

	// gemm

	template<typename T, int M, int N, int K> struct gemm;

	template<int M, int N, int K>
	struct gemm<double, M, N, K>
	{
		BCS_ENSURE_INLINE
		static void eval(const char transa, const char transb,
				const int m, const int n, const int k,
				const double alpha, const double *a, const int lda, const double *b, const int ldb,
				const double beta, double *c, const int ldc)
		{
			BCS_DGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
		}
	};

	template<int M, int N, int K>
	struct gemm<float, M, N, K>
	{
		BCS_ENSURE_INLINE
		static void eval(const char transa, const char transb,
				const int m, const int n, const int k,
				const float alpha, const float *a, const int lda, const float *b, const int ldb,
				const float beta, float *c, const int ldc)
		{
			BCS_SGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
		}
	};


	// symm

	template<typename T, int M, int N> struct symm;

	template<int M, int N>
	struct symm<double, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char side, const char uplo, const int m, const int n,
				const double alpha, const double *a, const int lda, const double *b, const int ldb,
				const double beta, double *c, const int ldc)
		{
			BCS_DSYMM(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
		}
	};

	template<int M, int N>
	struct symm<float, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char side, const char uplo, const int m, const int n,
				const float alpha, const float *a, const int lda, const float *b, const int ldb,
				const float beta, float *c, const int ldc)
		{
			BCS_SSYMM(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
		}
	};

	// trmm

	template<typename T, int M, int N> struct trmm;

	template<int M, int N>
	struct trmm<double, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char side, const char uplo, const char transa, const char diag,
				const int m, const int n,
				const double alpha, const double *a, const int lda,
				const double beta, double *b, const int ldb)
		{
			BCS_DTRMM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
		}
	};

	template<int M, int N>
	struct trmm<float, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char side, const char uplo, const char transa, const char diag,
				const int m, const int n,
				const float alpha, const float *a, const int lda,
				const float beta, float *b, const int ldb)
		{
			BCS_STRMM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
		}
	};

	// trsm

	template<typename T, int M, int N> struct trsm;

	template<int M, int N>
	struct trsm<double, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char side, const char uplo, const char transa, const char diag,
				const int m, const int n,
				const double alpha, const double *a, const int lda,
				const double beta, double *b, const int ldb)
		{
			BCS_DTRSM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
		}
	};

	template<int M, int N>
	struct trsm<float, M, N>
	{
		BCS_ENSURE_INLINE
		static void eval(const char side, const char uplo, const char transa, const char diag,
				const int m, const int n,
				const float alpha, const float *a, const int lda,
				const float beta, float *b, const int ldb)
		{
			BCS_STRSM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
		}
	};



} }


#endif /* BLAS_H_ */
