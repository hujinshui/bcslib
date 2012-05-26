/*
 * @file test_small_blasL2.cpp
 *
 * Unit testing of small matrix BLAS (Level 1)
 *
 * @author Dahua Lin
 */


#include <gtest/gtest.h>
#include <bcslib/engine/small_blasL2.h>

#include <bcslib/matrix.h>
#include <cstdio>

using namespace bcs;

template<typename T>
void L2_init_vec(int n, T *v, T b)
{
	for (int i = 0; i < n; ++i) v[i] = T(i+b);
}

template<typename T>
void L2_prn_vec(const char *name, int n, const T *x, const int inc = 1)
{
	std::printf("%s: ", name);
	for (int i = 0; i < n; ++i) std::printf("%6g ", x[i * inc]);
	std::printf("\n");
}


/************************************************
 *
 *  gemv
 *
 ************************************************/

template<typename T>
bool gemv_n_check(const ref_matrix_ex<T>& a,
		const T alpha, const T beta,
		const T *x, int incx, const T* y0, const T *y, int incy)
{
	const index_t m = a.nrows();
	const index_t n = a.ncolumns();

	dense_matrix<T> yt(m, 1);

	for (index_t i = 0; i < m; ++i)
	{
		T s(0);
		for (index_t j = 0; j < n; ++j) s += a(i, j) * x[j * incx];
		yt[i] = y0[i * incy] * beta + s * alpha;
	}

	bool pass = true;
	for (index_t i = 0; i < m; ++i) if (yt[i] != y[i * incy]) pass = false;

	if (!pass)
	{
		std::printf("gemv_n-dump [m = %ld, n = %ld]:\n", m, n);
		std::printf("   lda = %ld, incx = %d, incy = %d, alpha = %g, beta = %g\n",
				a.lead_dim(), incx, incy, alpha, beta);

		std::printf("  a = \n"); printf_mat("%6g ", a);
		L2_prn_vec("  x ", (int)n, x, incx);
		L2_prn_vec("  y0", (int)m, y0, incy);
		L2_prn_vec("  yt", (int)m, yt.ptr_data(), 1);
		L2_prn_vec("  y ", (int)m, y, incy);

		return false;
	}

	return pass;
}

template<typename T>
bool gemv_t_check(const ref_matrix_ex<T>& a,
		const T alpha, const T beta,
		const T *x, int incx, const T* y0, const T *y, int incy)
{
	const index_t m = a.nrows();
	const index_t n = a.ncolumns();

	dense_matrix<T> yt(1, n);

	for (index_t j = 0; j < n; ++j)
	{
		T s(0);
		for (index_t i = 0; i < m; ++i) s += a(i, j) * x[i * incx];
		yt[j] = y0[j * incy] * beta + s * alpha;
	}

	bool pass = true;
	for (index_t j = 0; j < n; ++j) if (yt[j] != y[j * incy]) pass = false;

	if (!pass)
	{
		std::printf("gemv_t-dump [m = %ld, n = %ld]:\n", m, n);
		std::printf("   lda = %ld, incx = %d, incy = %d, alpha = %g, beta = %g\n",
				a.lead_dim(), incx, incy, alpha, beta);

		std::printf("  a = \n"); printf_mat("%6g ", a);
		L2_prn_vec("  x ", (int)m, x, incx);
		L2_prn_vec("  y0", (int)n, y0, incy);
		L2_prn_vec("  yt", (int)n, yt.ptr_data(), 1);
		L2_prn_vec("  y ", (int)n, y, incy);

		return false;
	}

	return pass;
}



template<typename T, int M, int N>
void gemv_n_test()
{
	const index_t max_lda = M + 2;
	const index_t src_size = max_lda * N;

	scoped_block<T> a_src(src_size);
	T *pa = a_src.ptr_begin();
	L2_init_vec((int)src_size, pa, T(1));

	T x[4 * N];
	T y0[4 * M];
	T y[4 * M];

	L2_init_vec(4 * N, x, T(1));
	L2_init_vec(4 * M, y0, T(2));

	int incxs[2] = {1, 2};
	const int n_incxs = 2;

	int incys[2] = {1, 3};
	const int n_incys = 2;

	T alphas[2] = {T(1), T(2)};
	const int n_alphas = 2;

	T betas[3] = {T(0), T(1), T(2)};
	const int n_betas = 3;

	index_t ldas[2] = {M, max_lda};
	const int n_ldas = 2;

	for (int u = 0; u < n_incxs; ++u)
	for (int v = 0; v < n_incys; ++v)
	for (int i = 0; i < n_ldas; ++i)
	for (int j = 0; j < n_betas; ++j)
	for (int k = 0; k < n_alphas; ++k)
	{
		const int incx = incxs[u];
		const int incy = incys[v];

		const index_t lda = ldas[i];
		const T beta = betas[j];
		const T alpha = alphas[k];

		ref_matrix_ex<T> a(pa, (index_t)M, (index_t)N, lda);

		L2_init_vec(4 * M, y, T(2));
		engine::small_gemv_n<T, M, N>::eval(alpha, pa, (int)lda, x, incx, beta, y, incy);

		bool pass = gemv_n_check(a, alpha, beta, x, incx, y0, y, incy);
		ASSERT_TRUE( pass );
	}

}


template<typename T, int M, int N>
void gemv_t_test()
{
	const index_t max_lda = M + 2;
	const index_t src_size = max_lda * N;

	scoped_block<T> a_src(src_size);
	T *pa = a_src.ptr_begin();
	L2_init_vec((int)src_size, pa, T(1));

	T x[4 * M];
	T y0[4 * N];
	T y[4 * N];

	L2_init_vec(4 * M, x, T(1));
	L2_init_vec(4 * N, y0, T(2));

	int incxs[2] = {1, 2};
	const int n_incxs = 2;

	int incys[2] = {1, 3};
	const int n_incys = 2;

	T alphas[2] = {T(1), T(2)};
	const int n_alphas = 2;

	T betas[3] = {T(0), T(1), T(2)};
	const int n_betas = 3;

	index_t ldas[2] = {M, max_lda};
	const int n_ldas = 2;

	for (int u = 0; u < n_incxs; ++u)
	for (int v = 0; v < n_incys; ++v)
	for (int i = 0; i < n_ldas; ++i)
	for (int j = 0; j < n_betas; ++j)
	for (int k = 0; k < n_alphas; ++k)
	{
		const int incx = incxs[u];
		const int incy = incys[v];

		const index_t lda = ldas[i];
		const T beta = betas[j];
		const T alpha = alphas[k];

		ref_matrix_ex<T> a(pa, (index_t)M, (index_t)N, lda);

		L2_init_vec(4 * N, y, T(2));
		engine::small_gemv_t<T, M, N>::eval(alpha, pa, (int)lda, x, incx, beta, y, incy);

		bool pass = gemv_t_check(a, alpha, beta, x, incx, y0, y, incy);
		ASSERT_TRUE( pass );
	}

}


TEST( SmallBlasL2, GemvN_11d )
{
	gemv_n_test<double, 1, 1>();
}

TEST( SmallBlasL2, GemvT_11d )
{
	gemv_t_test<double, 1, 1>();
}

TEST( SmallBlasL2, GemvN_12d )
{
	gemv_n_test<double, 1, 2>();
}

TEST( SmallBlasL2, GemvT_12d )
{
	gemv_t_test<double, 1, 2>();
}

TEST( SmallBlasL2, GemvN_13d )
{
	gemv_n_test<double, 1, 3>();
}

TEST( SmallBlasL2, GemvT_13d )
{
	gemv_t_test<double, 1, 3>();
}

TEST( SmallBlasL2, GemvN_14d )
{
	gemv_n_test<double, 1, 4>();
}

TEST( SmallBlasL2, GemvT_14d )
{
	gemv_t_test<double, 1, 4>();
}

TEST( SmallBlasL2, GemvN_16d )
{
	gemv_n_test<double, 1, 6>();
}

TEST( SmallBlasL2, GemvT_16d )
{
	gemv_t_test<double, 1, 6>();
}


TEST( SmallBlasL2, GemvN_21d )
{
	gemv_n_test<double, 2, 1>();
}

TEST( SmallBlasL2, GemvT_21d )
{
	gemv_t_test<double, 2, 1>();
}

TEST( SmallBlasL2, GemvN_22d )
{
	gemv_n_test<double, 2, 2>();
}

TEST( SmallBlasL2, GemvT_22d )
{
	gemv_t_test<double, 2, 2>();
}

TEST( SmallBlasL2, GemvN_23d )
{
	gemv_n_test<double, 2, 3>();
}

TEST( SmallBlasL2, GemvT_23d )
{
	gemv_t_test<double, 2, 3>();
}

TEST( SmallBlasL2, GemvN_24d )
{
	gemv_n_test<double, 2, 4>();
}

TEST( SmallBlasL2, GemvT_24d )
{
	gemv_t_test<double, 2, 4>();
}

TEST( SmallBlasL2, GemvN_26d )
{
	gemv_n_test<double, 2, 6>();
}

TEST( SmallBlasL2, GemvT_26d )
{
	gemv_t_test<double, 2, 6>();
}


TEST( SmallBlasL2, GemvN_31d )
{
	gemv_n_test<double, 3, 1>();
}

TEST( SmallBlasL2, GemvT_31d )
{
	gemv_t_test<double, 3, 1>();
}

TEST( SmallBlasL2, GemvN_32d )
{
	gemv_n_test<double, 3, 2>();
}

TEST( SmallBlasL2, GemvT_32d )
{
	gemv_t_test<double, 3, 2>();
}

TEST( SmallBlasL2, GemvN_33d )
{
	gemv_n_test<double, 3, 3>();
}

TEST( SmallBlasL2, GemvT_33d )
{
	gemv_t_test<double, 3, 3>();
}

TEST( SmallBlasL2, GemvN_34d )
{
	gemv_n_test<double, 3, 4>();
}

TEST( SmallBlasL2, GemvT_34d )
{
	gemv_t_test<double, 3, 4>();
}

TEST( SmallBlasL2, GemvN_36d )
{
	gemv_n_test<double, 3, 6>();
}

TEST( SmallBlasL2, GemvT_36d )
{
	gemv_t_test<double, 3, 6>();
}


TEST( SmallBlasL2, GemvN_41d )
{
	gemv_n_test<double, 4, 1>();
}

TEST( SmallBlasL2, GemvT_41d )
{
	gemv_t_test<double, 4, 1>();
}

TEST( SmallBlasL2, GemvN_42d )
{
	gemv_n_test<double, 4, 2>();
}

TEST( SmallBlasL2, GemvT_42d )
{
	gemv_t_test<double, 4, 2>();
}

TEST( SmallBlasL2, GemvN_43d )
{
	gemv_n_test<double, 4, 3>();
}

TEST( SmallBlasL2, GemvT_43d )
{
	gemv_t_test<double, 4, 3>();
}

TEST( SmallBlasL2, GemvN_44d )
{
	gemv_n_test<double, 4, 4>();
}

TEST( SmallBlasL2, GemvT_44d )
{
	gemv_t_test<double, 4, 4>();
}

TEST( SmallBlasL2, GemvN_46d )
{
	gemv_n_test<double, 4, 6>();
}

TEST( SmallBlasL2, GemvT_46d )
{
	gemv_t_test<double, 4, 6>();
}


TEST( SmallBlasL2, GemvN_61d )
{
	gemv_n_test<double, 6, 1>();
}

TEST( SmallBlasL2, GemvT_61d )
{
	gemv_t_test<double, 6, 1>();
}

TEST( SmallBlasL2, GemvN_62d )
{
	gemv_n_test<double, 6, 2>();
}

TEST( SmallBlasL2, GemvT_62d )
{
	gemv_t_test<double, 6, 2>();
}

TEST( SmallBlasL2, GemvN_63d )
{
	gemv_n_test<double, 6, 3>();
}

TEST( SmallBlasL2, GemvT_63d )
{
	gemv_t_test<double, 6, 3>();
}

TEST( SmallBlasL2, GemvN_64d )
{
	gemv_n_test<double, 6, 4>();
}

TEST( SmallBlasL2, GemvT_64d )
{
	gemv_t_test<double, 6, 4>();
}

TEST( SmallBlasL2, GemvN_66d )
{
	gemv_n_test<double, 6, 6>();
}

TEST( SmallBlasL2, GemvT_66d )
{
	gemv_t_test<double, 6, 6>();
}



