/*
 * @file smallmat_mul.cpp
 *
 * Compare the performance of small matrix multiplication
 *
 * @author Dahua Lin
 */


#include <bcslib/matrix.h>
#include <bcslib/linalg.h>
#include <bcslib/utils/timer.h>
#include <cstdio>

using namespace bcs;

template<typename T, index_t M, index_t K, index_t N>
struct mm0
{
	static void run(const T alpha, const T beta,
			const T* a, const T* b, T *c)
	{
		if (beta == 0)
		{
			for (index_t j = 0; j < N; ++j)
			{
				for (index_t i = 0; i < M; ++i)
				{
					T s(0);

					for (index_t k = 0; k < K; ++k)
						s += a[i + k * M] * b[k + j * K];

					c[i + j * M] = alpha * s;
				}
			}
		}
		else
		{
			if (beta != 1)
			{
				for (index_t i = 0; i < M * N; ++i) c[i] *= beta;
			}

			for (index_t j = 0; j < N; ++j)
			{
				for (index_t i = 0; i < M; ++i)
				{
					T s(0);

					for (index_t k = 0; k < K; ++k)
						s += a[i + k * M] * b[k + j * K];

					c[i + j * M] += alpha * s;
				}
			}
		}

	}
};


template<typename T>
struct smallmm_111
{
	BCS_ENSURE_INLINE
	static void eval(const T alpha, const T beta,
			const T* __restrict__ a, const index_t lda,
			const T* __restrict__ b, const index_t ldb,
			T* __restrict__ c, const index_t ldc)
	{
		if (beta == 0)
		{
			c[0] = alpha * a[0] * b[0];
		}
		else
		{
			c[0] = alpha * a[0] * b[0] + beta * c[0];
		}
	}
};


template<typename T, int N>
struct smallmm_11N
{
	BCS_ENSURE_INLINE
	static void eval(const T alpha, const T beta,
			const T* __restrict__ a, const index_t lda,
			const T* __restrict__ b, const index_t ldb,
			T* __restrict__ c, const index_t ldc)
	{
		const T aa = alpha * a[0];

		if (beta == 0)
		{
			for (int j = 0; j < N; ++j) c[j * ldc] = aa * b[j * ldb];
		}
		else if (beta == 1)
		{
			for (int j = 0; j < N; ++j) c[j * ldc] += aa * b[j * ldb];
		}
		else
		{
			for (int j = 0; j < N; ++j) c[j * ldc] = aa * b[j * ldb] + beta * c[j * ldc];
		}
	}
};


template<typename T, int K>
struct smallmm_1K1
{
	BCS_ENSURE_INLINE
	static void eval(const T alpha, const T beta,
			const T* __restrict__ a, const index_t lda,
			const T* __restrict__ b, const index_t ldb,
			T* __restrict__ c, const index_t ldc)
	{
		T s(0);

		for (int k = 0; k < K; ++k) s += a[k * lda] * b[k];

		if (beta == 0) c[0] = alpha * s;
		else c[0] = alpha * s + beta * c[0];
	}
};


template<typename T, int K, int N>
struct smallmm_1KN
{
	inline
	static void eval(const T alpha, const T beta,
			const T* __restrict__ a, const index_t lda,
			const T* __restrict__ b, const index_t ldb,
			T* __restrict__ c, const index_t ldc)
	{
		if (beta == 0)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
				{
					const T *bj = b + ldb * j;
					T s(0);
					for (int k = 0; k < K; ++k) s += a[k * lda] * bj[k];
					c[j * ldc] = s;
				}
			}
			else
			{
				for (int j = 0; j < N; ++j)
				{
					const T *bj = b + ldb * j;
					T s(0);
					for (int k = 0; k < K; ++k) s += a[k * lda] * bj[k];
					c[j * ldc] = alpha * s;
				}
			}
		}
		else
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
				{
					const T *bj = b + ldb * j;
					T s(0);
					for (int k = 0; k < K; ++k) s += a[k * lda] * bj[k];
					c[j * ldc] = s + beta * c[j * ldc];
				}
			}
			else
			{
				for (int j = 0; j < N; ++j)
				{
					const T *bj = b + ldb * j;
					T s(0);
					for (int k = 0; k < K; ++k) s += a[k * lda] * bj[k];
					c[j * ldc] = alpha * s + beta * c[j * ldc];
				}
			}
		}
	}
};


template<typename T, int M>
struct smallmm_M11
{
	BCS_ENSURE_INLINE
	static void eval(const T alpha, const T beta,
			const T* __restrict__ a, const index_t lda,
			const T* __restrict__ b, const index_t ldb,
			T* __restrict__ c, const index_t ldc)
	{
		const T ab = alpha * b[0];

		if (beta == 0)
		{
			for (int i = 0; i < M; ++i) c[i] = ab * a[i];
		}
		else if (beta == 1)
		{
			for (int i = 0; i < M; ++i) c[i] += ab * a[i];
		}
		else
		{
			for (int i = 0; i < M; ++i) c[i] = ab * a[i] + beta * c[i];
		}
	}
};


template<typename T, int M, int N>
struct smallmv
{
	BCS_ENSURE_INLINE
	static void eval_b0(const T alpha,
			const T* __restrict__ a, const index_t lda, const T* __restrict__ b,
			T* __restrict__ c)
	{
		if (alpha == 1)
		{
			const T b0 = b[0];
			for (int i = 0; i < M; ++i) c[i] = a[i] * b0;

			for (int j = 1; j < N; ++j)
			{
				const T *aj = a + lda * j;
				const T bj = b[j];

				for (int i = 0; i < M; ++i) c[i] += aj[i] * bj;
			}
		}
		else
		{
			const T b0 = b[0] * alpha;
			for (int i = 0; i < M; ++i) c[i] = a[i] * b0;

			for (int j = 1; j < N; ++j)
			{
				const T *aj = a + lda * j;
				const T bj = b[j] * alpha;

				for (int i = 0; i < M; ++i) c[i] += aj[i] * bj;
			}
		}
	}


	BCS_ENSURE_INLINE
	static void eval_b1(const T alpha,
			const T* __restrict__ a, const index_t lda, const T* __restrict__ b,
			T* __restrict__ c)
	{
		if (alpha == 1)
		{
			for (int j = 0; j < N; ++j)
			{
				const T *aj = a + lda * j;
				const T bj = b[j];

				for (int i = 0; i < M; ++i) c[i] += aj[i] * bj;
			}
		}
		else
		{
			for (int j = 0; j < N; ++j)
			{
				const T *aj = a + lda * j;
				const T bj = b[j] * alpha;

				for (int i = 0; i < M; ++i) c[i] += aj[i] * bj;
			}
		}
	}
};



template<typename T, int M, int N>
struct smallmm_M1N
{
	inline
	static void eval(const T alpha, const T beta,
			const T* __restrict__ a, const index_t lda,
			const T* __restrict__ b, const index_t ldb,
			T* __restrict__ c, const index_t ldc)
	{
		if (beta == 0)
		{
			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
				{
					T *cj = c + j * ldc;
					const T bj = b[j * ldb];

					for (int i = 0; i < M; ++i) cj[i] = a[i] * bj;
				}
			}
			else
			{
				for (int j = 0; j < N; ++j)
				{
					T *cj = c + j * ldc;
					const T bj = b[j * ldb] * alpha;

					for (int i = 0; i < M; ++i) cj[i] = a[i] * bj;
				}
			}
		}
		else
		{
			if (beta != 1)
			{
				for (int j = 0; j < N; ++j)
				{
					T *cj = c + j * ldc;
					for (int i = 0; i < M; ++i) cj[i] *= beta;
				}
			}

			if (alpha == 1)
			{
				for (int j = 0; j < N; ++j)
				{
					T *cj = c + j * ldc;
					const T bj = b[j * ldb];

					for (int i = 0; i < M; ++i) cj[i] += a[i] * bj;
				}
			}
			else
			{
				for (int j = 0; j < N; ++j)
				{
					T *cj = c + j * ldc;
					const T bj = b[j * ldb] * alpha;

					for (int i = 0; i < M; ++i) cj[i] += a[i] * bj;
				}
			}
		}
	}
};


template<typename T, int M, int K>
struct smallmm_MK1
{
	inline
	static void eval(const T alpha, const T beta,
			const T* __restrict__ a, const index_t lda,
			const T* __restrict__ b, const index_t ldb,
			T* __restrict__ c, const index_t ldc)
	{
		if (beta == 0)
		{
			smallmv<T, M, K>::eval_b0(alpha, a, lda, b, c);
		}
		else
		{
			if (beta != 1)
			{
				for (int i = 0; i < M; ++i) c[i] *= beta;
			}

			smallmv<T, M, K>::eval_b1(alpha, a, lda, b, c);
		}
	}
};


template<typename T, int M, int K, int N>
struct smallmm_MKN
{
	inline
	static void eval(const T alpha, const T beta,
			const T* __restrict__ a, const index_t lda,
			const T* __restrict__ b, const index_t ldb,
			T* __restrict__ c, const index_t ldc)
	{
		if (beta == 0)
		{
			smallmv<T, M, K>::eval_b0(alpha, a, lda, b, c);

			for (int j = 1; j < N; ++j)
			{
				smallmv<T, M, K>::eval_b0(alpha, a, lda, b + j * ldb, c + j * ldc);
			}
		}
		else
		{
			if (beta != 1)
			{
				for (int j = 0; j < N; ++j)
				{
					T *cj = c + j * ldc;
					for (int i = 0; i < M; ++i) cj[i] *= beta;
				}
			}

			for (int j = 0; j < N; ++j)
			{
				smallmv<T, M, K>::eval_b1(alpha, a, lda, b + j * ldb, c + j * ldc);
			}
		}
	}
};

template<typename T, int M, int K, int N>
struct smallmm
{
	typedef typename select_type<M == 1,
			// M = 1
				typename select_type<K == 1,
					// M = 1 & K = 1
					typename select_type<N == 1,
						smallmm_111<T>,
						smallmm_11N<T, N>
					>::type,
					// M = 1 & K > 1
					typename select_type<N == 1,
						smallmm_1K1<T, K>,
						smallmm_1KN<T, K, N>
					>::type
				>::type,
			// M > 1
				typename select_type<K == 1,
					// M > 1 & K = 1
					typename select_type<N == 1,
						smallmm_M11<T, M>,
						smallmm_M1N<T, M, N>
					>::type,
					// M > 1 & K > 1
					typename select_type<N == 1,
						smallmm_MK1<T, M, K>,
						smallmm_MKN<T, M, K, N>
					>::type
				>::type
			>::type impl_t;

	BCS_ENSURE_INLINE
	static void eval(const T alpha, const T beta,
			const T* __restrict__ a, const index_t lda,
			const T* __restrict__ b, const index_t ldb,
			T* __restrict__ c, const index_t ldc)
	{
		impl_t::eval(alpha, beta, a, lda, b, ldb, c, ldc);
	}

};



typedef double real;
typedef dense_matrix<real> real_mat;

void init_mat(real_mat& a)
{
	const index_t N = a.nelems();

	for (int i = 0; i < N; ++i) a[i] = real(i + 1);
}


template<int M, int K, int N>
void verify_case(const real alpha, const real beta,
		const real_mat& a, const real_mat& b, real_mat& c0, real_mat& c)
{
	init_mat(c0);
	init_mat(c);

	mm0<real, M, K, N>::run(alpha, beta, a.ptr_data(), b.ptr_data(), c0.ptr_data());

	smallmm<real, M, K, N>::eval(alpha, beta,
			a.ptr_data(), a.lead_dim(), b.ptr_data(), b.lead_dim(), c.ptr_data(), c.lead_dim());

	if ( !is_equal(c, c0) )
	{
		std::printf("... V-FAIL @ M = %d, N = %d, K = %d, alpha = %g, beta = %g\n",
				M, N, K, alpha, beta);

		std::printf("a = \n"); printf_mat("%6g ", a);
		std::printf("b = \n"); printf_mat("%6g ", b);

		std::printf("c0 = \n"); printf_mat("%6g ", c0);
		std::printf("c = \n"); printf_mat("%6g ", c);

		std::printf("\n");
	}
}


template<index_t M, index_t K, index_t N>
void verify_mm()
{
	real_mat a(M, K);
	real_mat b(K, N);
	real_mat c0(M, N);
	real_mat c1(M, N);

	init_mat(a);
	init_mat(b);

	verify_case<M, K, N>(real(1), real(0), a, b, c0, c1);
	verify_case<M, K, N>(real(1), real(1), a, b, c0, c1);
	verify_case<M, K, N>(real(1), real(2), a, b, c0, c1);

	verify_case<M, K, N>(real(2), real(0), a, b, c0, c1);
	verify_case<M, K, N>(real(2), real(1), a, b, c0, c1);
	verify_case<M, K, N>(real(2), real(2), a, b, c0, c1);
}


template<int M>
void verify_all()
{
	verify_mm<M, 1, 1>();
	verify_mm<M, 1, 4>();

	verify_mm<M, 2, 1>();
	verify_mm<M, 2, 4>();

	verify_mm<M, 4, 1>();
	verify_mm<M, 4, 4>();

	verify_mm<M, 8, 1>();
	verify_mm<M, 8, 4>();

	std::printf("verification for M = %d done\n", M);
}


void report_res(int m, int k, int n,
		const int rt_a, const double s_a,
		const int rt_b, const double s_b,
		const int rt_mkl, const double s_mkl)
{
	double g_a = (double(2 * m * n * k) * double(rt_a) * 1.0e-9) / s_a;
	double g_b = (double(2 * m * n * k) * double(rt_b) * 1.0e-9) / s_b;
	double g_m = (double(2 * m * n * k) * double(rt_mkl) * 1.0e-9) / s_mkl;

	std::printf("%3d, %3d, %3d,   %7.4f, %7.4f, %7.4f\n",
			m, k, n, g_a, g_b, g_m);
}


template<int M, int K, int N>
void bench_mm()
{
	typedef double real;

	dense_matrix<real> a(M, K);
	dense_matrix<real> b(K, N);
	dense_matrix<real> c(M, N);

	init_mat(a);
	init_mat(b);
	init_mat(c);

	int rt0 = int(double(2e8) / double(M * N * K));

	// test mm_a

	const int rt_a = rt0;

	timer tm_a;
	tm_a.start();

	for (int i = 0; i < rt_a; ++i)
	{
		mm0<real, M, K, N>::run(real(1), real(0), a.ptr_data(), b.ptr_data(), c.ptr_data());
	}

	double s_a = tm_a.elapsed_secs();

	// test mm_b

	const int rt_b = rt0;

	timer tm_b;
	tm_b.start();

	for (int i = 0; i < rt_b; ++i)
	{
		#pragma ivdep
		smallmm<real, M, K, N>::eval(real(1), real(0),
				a.ptr_data(), a.lead_dim(), b.ptr_data(), b.lead_dim(),
				c.ptr_data(), c.lead_dim());
	}

	double s_b = tm_b.elapsed_secs();

	// test mkl

	const int rt_mkl = 1000000;

	for (int i = 0; i < 10; ++i) c = mm(a, b); // warming

	timer tm_mkl;
	tm_mkl.start();

	for (int i = 0; i < rt_mkl; ++i)
	{
		c = mm(a, b);
	}

	double s_mkl = tm_mkl.elapsed_secs();

	report_res(M, K, N, rt_a, s_a, rt_b, s_b, rt_mkl, s_mkl);
}


int main(int argc, char *argv[])
{
	bench_mm<4, 4, 4>();
}






