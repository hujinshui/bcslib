/**
 * @file blas_refimpl.h
 *
 * Referenced implementation for BLAS
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BLAS_REFIMPL_H_
#define BCSLIB_BLAS_REFIMPL_H_

#include <bcslib/core/basic_defs.h>

namespace bcs { namespace engine {

	// auxiliary

	template<typename T>
	inline void inplace_prepare_dst1d(const index_t N, const T* y, const index_t incy, const T beta)
	{
		if (incy == 1)
		{
			if (beta == 0) for (index_t i = 0; i < N; ++i) y[i] = T(0);
			else if (beta != 1) for (index_t i = 0; i < N; ++i) y[i] *= beta;
		}
		else
		{
			if (beta == 0) for (index_t i = 0; i < N; ++i) y[i * incy] = T(0);
			else if (beta != 1) for (index_t i = 0; i < N; ++i) y[i * incy] *= beta;
		}
	}

	template<typename T>
	inline void inplace_prepare_dst2d(const index_t M, const index_t N, const T* a, const index_t lda, const T beta)
	{
		if (beta == 0)
		{
			for (index_t j = 0; j < N; ++j)
				for (index_t i = 0; i < M; ++i) a[i + j * lda] = 0;
		}
		else if (beta != 1)
		{
			for (index_t j = 0; j < N; ++j)
				for (index_t i = 0; i < M; ++i) a[i + j * lda] *= beta;
		}
	}


	/********************************************
	 *
	 *  Level 1
	 *
	 ********************************************/

	// dot

	template<typename T>
	inline T ref_dot_(const index_t N, const T* __restrict__ x, const T* __restrict__ y)
	{
		T s(0);
		for (int i = 0; i < N; ++i) s += x[i] * y[i];
		return s;
	}

	template<typename T>
	inline T ref_dot_(const index_t N, const T* __restrict__ x, const index_t incx, const T* __restrict__ y)
	{
		T s(0);
		for (int i = 0; i < N; ++i) s += x[i * incx] * y[i];
		return s;
	}

	template<typename T>
	inline T ref_dot_(const index_t N, const T* __restrict__ x, const T* __restrict__ y, const index_t incy)
	{
		T s(0);
		for (int i = 0; i < N; ++i) s += x[i] * y[i * incy];
		return s;
	}

	template<typename T>
	inline T ref_dot_(const index_t N, const T* __restrict__ x, const index_t incx, const T* __restrict__ y, const index_t incy)
	{
		T s(0);
		for (int i = 0; i < N; ++i) s += x[i * incx] * y[i * incy];
		return s;
	}


	template<typename T>
	inline T ref_dot(const index_t N, const T* __restrict__ x, const index_t incx, const T* __restrict__ y, const index_t incy)
	{
		T s(0);

		if (incx == 1)
		{
			if (incy == 1) ref_dot_(N, x, y);
			else ref_dot_(N, x, y, incy);
		}
		else
		{
			if (incy == 1) ref_dot_(N, x, incx, y);
			else ref_dot_(N, x, incx, y, incy);
		}

		return s;
	}


	// axpy

	template<typename T>
	inline void ref_axpy_(const index_t N, const T a, const T* __restrict__ x, const T* __restrict__ y)
	{
		for (int i = 0; i < N; ++i) y[i] += a * x[i];
	}

	template<typename T>
	inline void ref_axpy_(const index_t N, const T a, const T* __restrict__ x, const index_t incx, const T* __restrict__ y)
	{
		for (int i = 0; i < N; ++i) y[i] += a * x[i * incx];
	}

	template<typename T>
	inline void ref_axpy_(const index_t N, const T a, const T* __restrict__ x, const T* __restrict__ y, const index_t incy)
	{
		for (int i = 0; i < N; ++i) y[i * incy] += a * x[i];
	}

	template<typename T>
	inline void ref_axpy_(const index_t N, const T a, const T* __restrict__ x, const index_t incx, const T* __restrict__ y, const index_t incy)
	{
		for (int i = 0; i < N; ++i) y[i * incy] += a * x[i * incx];
	}


	template<typename T>
	inline void ref_axpy(const index_t N, const T a, const T* __restrict__ x, const index_t incx, const T* __restrict__ y, const index_t incy)
	{
		if (incx == 1)
		{
			if (incy == 1) ref_axpy_(N, a, x, y);
			else ref_axpy_(N, x, y, incy);
		}
		else
		{
			if (incy == 1) ref_axpy_(N, x, incx, y);
			else ref_axpy_(N, x, incx, y, incy);
		}
	}


	/********************************************
	 *
	 *  Level 2
	 *
	 ********************************************/

	// gemv_n

	template<typename T>
	inline void ref_gemv_n_(const index_t M, const index_t N,
			const T alpha, const T* __restrict__ a, const index_t lda,
			const T* __restrict__ x,
			const T beta, T* __restrict__ y)
	{
		for (index_t j = 0; j < N; ++j)
		{
			ref_axpy_(M, x[j], a + j * lda, y);
		}
	}

	template<typename T>
	inline void ref_gemv_n_(const index_t M, const index_t N,
			const T alpha, const T* __restrict__ a, const index_t lda,
			const T* __restrict__ x, const index_t incx,
			const T beta, T* __restrict__ y)
	{
		for (index_t j = 0; j < N; ++j)
		{
			ref_axpy_(M, x[j * incx], a + j * lda, y);
		}
	}

	template<typename T>
	inline void ref_gemv_n_(const index_t M, const index_t N,
			const T alpha, const T* __restrict__ a, const index_t lda,
			const T* __restrict__ x,
			const T beta, T* __restrict__ y, const index_t incy)
	{
		for (index_t j = 0; j < N; ++j)
		{
			ref_axpy_(M, x[j], a + j * lda, y, incy);
		}
	}

	template<typename T>
	inline void ref_gemv_n_(const index_t M, const index_t N,
			const T alpha, const T* __restrict__ a, const index_t lda,
			const T* __restrict__ x, const index_t incx,
			const T beta, T* __restrict__ y, const index_t incy)
	{
		for (index_t j = 0; j < N; ++j)
		{
			ref_axpy_(M, x[j * incx], a + j * lda, y, incy);
		}
	}

	template<typename T>
	inline void ref_gemv_n(const index_t M, const index_t N,
			const T alpha, const T* __restrict__ a, const index_t lda,
			const T* __restrict__ x, const index_t incx,
			const T beta, T* __restrict__ y, const index_t incy)
	{
		inplace_prepare_dst1d(M, y, incy, beta);

		if (incx == 1)
		{
			if (incy == 1) ref_gemv_n_(M, N, alpha, a, lda, x, beta, y);
			else ref_gemv_n_(M, N, alpha, a, lda, x, beta, y, incy);
		}
		else
		{
			if (incy == 1) ref_gemv_n_(M, N, alpha, a, lda, x, incx, beta, y);
			else ref_gemv_n_(M, N, alpha, a, lda, x, incx, beta, y, incy);
		}
	}


	/********************************************
	 *
	 *  Level 3
	 *
	 ********************************************/

	// gemm_nn

	template<typename T>
	inline void ref_gemm_nn(const index_t M, const index_t N, const index_t K,
			const T alpha, const T* a, const index_t lda, const T* b, const index_t ldb,
			const T beta, T *c, const index_t ldc)
	{
		inplace_prepare_dst2d(M, N, c, ldc, beta);

		for (index_t j = 0; j < N; ++j)
		{
			ref_gemv_n_(M, K, alpha, a, lda, b + j * ldb, beta, c + j * ldc);
		}
	};

} }


#endif /* BLAS_REFIMPL_H_ */
