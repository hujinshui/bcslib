/**
 * @file test_small_gemm.cpp
 *
 * Unit testing of small matrix GEMM
 *
 * @author Dahua Lin
 */

#include <gtest/gtest.h>
#include <bcslib/engine/small_blasL3.h>

#include <bcslib/matrix.h>
#include <cstdio>

using namespace bcs;

template<typename T>
void L3_init_mat(dense_matrix<T>& a, T b)
{
	for (int i = 0; i < a.nelems(); ++i) a[i] = T(i+b);
}


template<typename T, class MatA, class MatB, class MatC>
dense_matrix<T> naive_gemm(const T alpha, const T beta,
		const IDenseMatrix<MatA, T>& a,
		const IDenseMatrix<MatB, T>& b,
		const IDenseMatrix<MatC, T>& c)
{
	const index_t m = a.nrows();
	const index_t k = a.ncolumns();
	const index_t n = b.ncolumns();

	dense_matrix<T> r(m, n);

	for (index_t j = 0; j < n; ++j)
	{
		for (index_t i = 0; i < m; ++i)
		{
			T s(0);
			for (index_t u = 0; u < k; ++u) s += a(i, u) * b(u, j);
			r(i, j) = beta * c(i, j) + alpha * s;
		}
	}

	return r;
}


template<typename T, class MatA, class MatB, class MatC0, class MatC>
bool check_gemm_nn(const T alpha, const T beta, const MatA& a, const MatB& b,
		const MatC0& c0, const MatC& c)
{
	dense_matrix<T> r = naive_gemm(alpha, beta, a, b, c0);

	bool pass = is_equal(r, c);
	if (!pass)
	{
		const index_t m = a.nrows();
		const index_t n = b.ncolumns();
		const index_t k = a.ncolumns();

		std::printf("gemm_nn-dump [m = %ld, n = %ld, k = %ld]\n", m, n, k);
		std::printf("  alpha = %g,  beta = %g, cont_a = %d, cont_b = %d, cont_c = %d\n",
				alpha, beta,
				int(a.lead_dim() == a.nrows()),
				int(b.lead_dim() == b.nrows()),
				int(c.lead_dim() == c.nrows()));

		std::printf("a = \n"); printf_mat("%6g ", a);
		std::printf("b = \n"); printf_mat("%6g ", b);
		std::printf("c0 = \n"); printf_mat("%6g ", c0);
		std::printf("r = \n"); printf_mat("%6g ", r);
		std::printf("c = \n"); printf_mat("%6g ", c);
		std::printf("\n");
	}

	return pass;
}

template<typename T, class MatA, class MatB, class MatC0, class MatC>
bool check_gemm_nt(const T alpha, const T beta, const MatA& a, const MatB& b,
		const MatC0& c0, const MatC& c)
{
	dense_matrix<T> bt = b.trans();
	dense_matrix<T> r = naive_gemm(alpha, beta, a, bt, c0);

	bool pass = is_equal(r, c);
	if (!pass)
	{
		const index_t m = a.nrows();
		const index_t n = b.nrows();
		const index_t k = a.ncolumns();

		std::printf("gemm_nt-dump [m = %ld, n = %ld, k = %ld]\n", m, n, k);
		std::printf("  alpha = %g,  beta = %g, cont_a = %d, cont_b = %d, cont_c = %d\n",
				alpha, beta,
				int(a.lead_dim() == a.nrows()),
				int(b.lead_dim() == b.nrows()),
				int(c.lead_dim() == c.nrows()));

		std::printf("a = \n"); printf_mat("%6g ", a);
		std::printf("b = \n"); printf_mat("%6g ", b);
		std::printf("c0 = \n"); printf_mat("%6g ", c0);
		std::printf("r = \n"); printf_mat("%6g ", r);
		std::printf("c = \n"); printf_mat("%6g ", c);
		std::printf("\n");
	}

	return pass;
}


template<typename T, class MatA, class MatB, class MatC0, class MatC>
bool check_gemm_tn(const T alpha, const T beta, const MatA& a, const MatB& b,
		const MatC0& c0, const MatC& c)
{
	dense_matrix<T> at = a.trans();
	dense_matrix<T> r = naive_gemm(alpha, beta, at, b, c0);

	bool pass = is_equal(r, c);
	if (!pass)
	{
		const index_t m = a.ncolumns();
		const index_t n = b.ncolumns();
		const index_t k = a.nrows();

		std::printf("gemm_tn-dump [m = %ld, n = %ld, k = %ld]\n", m, n, k);
		std::printf("  alpha = %g,  beta = %g, cont_a = %d, cont_b = %d, cont_c = %d\n",
				alpha, beta,
				int(a.lead_dim() == a.nrows()),
				int(b.lead_dim() == b.nrows()),
				int(c.lead_dim() == c.nrows()));

		std::printf("a = \n"); printf_mat("%6g ", a);
		std::printf("b = \n"); printf_mat("%6g ", b);
		std::printf("c0 = \n"); printf_mat("%6g ", c0);
		std::printf("r = \n"); printf_mat("%6g ", r);
		std::printf("c = \n"); printf_mat("%6g ", c);
		std::printf("\n");
	}

	return pass;
}


template<typename T, class MatA, class MatB, class MatC0, class MatC>
bool check_gemm_tt(const T alpha, const T beta, const MatA& a, const MatB& b,
		const MatC0& c0, const MatC& c)
{
	dense_matrix<T> at = a.trans();
	dense_matrix<T> bt = b.trans();
	dense_matrix<T> r = naive_gemm(alpha, beta, at, bt, c0);

	bool pass = is_equal(r, c);
	if (!pass)
	{
		const index_t m = a.ncolumns();
		const index_t n = b.nrows();
		const index_t k = a.nrows();

		std::printf("gemm_tt-dump [m = %ld, n = %ld, k = %ld]\n", m, n, k);
		std::printf("  alpha = %g,  beta = %g, cont_a = %d, cont_b = %d, cont_c = %d\n",
				alpha, beta,
				int(a.lead_dim() == a.nrows()),
				int(b.lead_dim() == b.nrows()),
				int(c.lead_dim() == c.nrows()));

		std::printf("a = \n"); printf_mat("%6g ", a);
		std::printf("b = \n"); printf_mat("%6g ", b);
		std::printf("c0 = \n"); printf_mat("%6g ", c0);
		std::printf("r = \n"); printf_mat("%6g ", r);
		std::printf("c = \n"); printf_mat("%6g ", c);
		std::printf("\n");
	}

	return pass;
}


template<typename T, int M, int N, int K>
void gemm_test_nn()
{
	const int ld_add = 2;

	const int ma = M; const int na = K;
	const int mb = K; const int nb = N;
	const int mc = M; const int nc = N;

	const int lda0 = ma + ld_add;
	const int ldb0 = mb + ld_add;
	const int ldc0 = mc + ld_add;

	dense_matrix<T> a0(lda0, na);
	dense_matrix<T> b0(ldb0, nb);
	dense_matrix<T> c0(ldc0, nc);

	L3_init_mat(a0, T(1));
	L3_init_mat(b0, T(2));
	L3_init_mat(c0, T(3));

	const int lda_s[2] = {ma, lda0};
	const int ldb_s[2] = {mb, ldb0};
	const int ldc_s[2] = {mc, ldc0};

	const T alpha_s[2] = {T(1), T(2)};
	const T beta_s[3] = {T(0), T(1), T(2)};

	for (int ia = 0; ia < 2; ++ia)
	for (int ib = 0; ib < 2; ++ib)
	for (int ic = 0; ic < 2; ++ic)
	for (int ja = 0; ja < 2; ++ja)
	for (int jb = 0; jb < 3; ++jb)
	{
		const int lda = lda_s[ia];
		const int ldb = ldb_s[ib];
		const int ldc = ldc_s[ic];

		const T alpha = alpha_s[ja];
		const T beta = beta_s[jb];

		ref_matrix_ex<T> a(a0.ptr_data(), ma, na, lda);
		ref_matrix_ex<T> b(b0.ptr_data(), mb, nb, ldb);
		ref_matrix_ex<T> c_bas(c0.ptr_data(), mc, nc, ldc);

		dense_matrix<T> c_cp(c0);
		ref_matrix_ex<T> c(c_cp.ptr_data(), mc, nc, ldc);

		engine::small_gemm<T, M, N, K>::eval_nn(
				alpha, a.ptr_data(), lda, b.ptr_data(), ldb,
				beta, c.ptr_data(), ldc);

		bool pass = check_gemm_nn(alpha, beta, a, b, c_bas, c);
		ASSERT_TRUE( pass );
	}
}

template<typename T, int M, int N, int K>
void gemm_test_nt()
{
	const int ld_add = 2;

	const int ma = M; const int na = K;
	const int mb = N; const int nb = K;
	const int mc = M; const int nc = N;

	const int lda0 = ma + ld_add;
	const int ldb0 = mb + ld_add;
	const int ldc0 = mc + ld_add;

	dense_matrix<T> a0(lda0, na);
	dense_matrix<T> b0(ldb0, nb);
	dense_matrix<T> c0(ldc0, nc);

	L3_init_mat(a0, T(1));
	L3_init_mat(b0, T(2));
	L3_init_mat(c0, T(3));

	const int lda_s[2] = {ma, lda0};
	const int ldb_s[2] = {mb, ldb0};
	const int ldc_s[2] = {mc, ldc0};

	const T alpha_s[2] = {T(1), T(2)};
	const T beta_s[3] = {T(0), T(1), T(2)};

	for (int ia = 0; ia < 2; ++ia)
	for (int ib = 0; ib < 2; ++ib)
	for (int ic = 0; ic < 2; ++ic)
	for (int ja = 0; ja < 2; ++ja)
	for (int jb = 0; jb < 3; ++jb)
	{
		const int lda = lda_s[ia];
		const int ldb = ldb_s[ib];
		const int ldc = ldc_s[ic];

		const T alpha = alpha_s[ja];
		const T beta = beta_s[jb];

		ref_matrix_ex<T> a(a0.ptr_data(), ma, na, lda);
		ref_matrix_ex<T> b(b0.ptr_data(), mb, nb, ldb);
		ref_matrix_ex<T> c_bas(c0.ptr_data(), mc, nc, ldc);

		dense_matrix<T> c_cp(c0);
		ref_matrix_ex<T> c(c_cp.ptr_data(), mc, nc, ldc);

		engine::small_gemm<T, M, N, K>::eval_nt(
				alpha, a.ptr_data(), lda, b.ptr_data(), ldb,
				beta, c.ptr_data(), ldc);

		bool pass = check_gemm_nt(alpha, beta, a, b, c_bas, c);
		ASSERT_TRUE( pass );
	}
}


template<typename T, int M, int N, int K>
void gemm_test_tn()
{
	const int ld_add = 2;

	const int ma = K; const int na = M;
	const int mb = K; const int nb = N;
	const int mc = M; const int nc = N;

	const int lda0 = ma + ld_add;
	const int ldb0 = mb + ld_add;
	const int ldc0 = mc + ld_add;

	dense_matrix<T> a0(lda0, na);
	dense_matrix<T> b0(ldb0, nb);
	dense_matrix<T> c0(ldc0, nc);

	L3_init_mat(a0, T(1));
	L3_init_mat(b0, T(2));
	L3_init_mat(c0, T(3));

	const int lda_s[2] = {ma, lda0};
	const int ldb_s[2] = {mb, ldb0};
	const int ldc_s[2] = {mc, ldc0};

	const T alpha_s[2] = {T(1), T(2)};
	const T beta_s[3] = {T(0), T(1), T(2)};

	for (int ia = 0; ia < 2; ++ia)
	for (int ib = 0; ib < 2; ++ib)
	for (int ic = 0; ic < 2; ++ic)
	for (int ja = 0; ja < 2; ++ja)
	for (int jb = 0; jb < 3; ++jb)
	{
		const int lda = lda_s[ia];
		const int ldb = ldb_s[ib];
		const int ldc = ldc_s[ic];

		const T alpha = alpha_s[ja];
		const T beta = beta_s[jb];

		ref_matrix_ex<T> a(a0.ptr_data(), ma, na, lda);
		ref_matrix_ex<T> b(b0.ptr_data(), mb, nb, ldb);
		ref_matrix_ex<T> c_bas(c0.ptr_data(), mc, nc, ldc);

		dense_matrix<T> c_cp(c0);
		ref_matrix_ex<T> c(c_cp.ptr_data(), mc, nc, ldc);

		engine::small_gemm<T, M, N, K>::eval_tn(
				alpha, a.ptr_data(), lda, b.ptr_data(), ldb,
				beta, c.ptr_data(), ldc);

		bool pass = check_gemm_tn(alpha, beta, a, b, c_bas, c);
		ASSERT_TRUE( pass );
	}
}

template<typename T, int M, int N, int K>
void gemm_test_tt()
{
	const int ld_add = 2;

	const int ma = K; const int na = M;
	const int mb = N; const int nb = K;
	const int mc = M; const int nc = N;

	const int lda0 = ma + ld_add;
	const int ldb0 = mb + ld_add;
	const int ldc0 = mc + ld_add;

	dense_matrix<T> a0(lda0, na);
	dense_matrix<T> b0(ldb0, nb);
	dense_matrix<T> c0(ldc0, nc);

	L3_init_mat(a0, T(1));
	L3_init_mat(b0, T(2));
	L3_init_mat(c0, T(3));

	const int lda_s[2] = {ma, lda0};
	const int ldb_s[2] = {mb, ldb0};
	const int ldc_s[2] = {mc, ldc0};

	const T alpha_s[2] = {T(1), T(2)};
	const T beta_s[3] = {T(0), T(1), T(2)};

	for (int ia = 0; ia < 2; ++ia)
	for (int ib = 0; ib < 2; ++ib)
	for (int ic = 0; ic < 2; ++ic)
	for (int ja = 0; ja < 2; ++ja)
	for (int jb = 0; jb < 3; ++jb)
	{
		const int lda = lda_s[ia];
		const int ldb = ldb_s[ib];
		const int ldc = ldc_s[ic];

		const T alpha = alpha_s[ja];
		const T beta = beta_s[jb];

		ref_matrix_ex<T> a(a0.ptr_data(), ma, na, lda);
		ref_matrix_ex<T> b(b0.ptr_data(), mb, nb, ldb);
		ref_matrix_ex<T> c_bas(c0.ptr_data(), mc, nc, ldc);

		dense_matrix<T> c_cp(c0);
		ref_matrix_ex<T> c(c_cp.ptr_data(), mc, nc, ldc);

		engine::small_gemm<T, M, N, K>::eval_tt(
				alpha, a.ptr_data(), lda, b.ptr_data(), ldb,
				beta, c.ptr_data(), ldc);

		bool pass = check_gemm_tt(alpha, beta, a, b, c_bas, c);
		ASSERT_TRUE( pass );
	}
}


template<typename T, int M, int N, int K>
void gemm_test()
{
	gemm_test_nn<T, M, N, K>();
	gemm_test_nt<T, M, N, K>();
	gemm_test_tn<T, M, N, K>();
	gemm_test_tt<T, M, N, K>();
}

TEST( SmallBlasL3, Gemm_111d )
{
	gemm_test<double, 1, 1, 1>();
}

TEST( SmallBlasL3, Gemm_112d )
{
	gemm_test<double, 1, 1, 2>();
}

TEST( SmallBlasL3, Gemm_114d )
{
	gemm_test<double, 1, 1, 4>();
}

TEST( SmallBlasL3, Gemm_116d )
{
	gemm_test<double, 1, 1, 6>();
}

TEST( SmallBlasL3, Gemm_121d )
{
	gemm_test<double, 1, 2, 1>();
}

TEST( SmallBlasL3, Gemm_122d )
{
	gemm_test<double, 1, 2, 2>();
}

TEST( SmallBlasL3, Gemm_124d )
{
	gemm_test<double, 1, 2, 4>();
}

TEST( SmallBlasL3, Gemm_126d )
{
	gemm_test<double, 1, 2, 6>();
}

TEST( SmallBlasL3, Gemm_141d )
{
	gemm_test<double, 1, 4, 1>();
}

TEST( SmallBlasL3, Gemm_142d )
{
	gemm_test<double, 1, 4, 2>();
}

TEST( SmallBlasL3, Gemm_144d )
{
	gemm_test<double, 1, 4, 4>();
}

TEST( SmallBlasL3, Gemm_146d )
{
	gemm_test<double, 1, 4, 6>();
}

TEST( SmallBlasL3, Gemm_161d )
{
	gemm_test<double, 1, 6, 1>();
}

TEST( SmallBlasL3, Gemm_162d )
{
	gemm_test<double, 1, 6, 2>();
}

TEST( SmallBlasL3, Gemm_164d )
{
	gemm_test<double, 1, 6, 4>();
}

TEST( SmallBlasL3, Gemm_166d )
{
	gemm_test<double, 1, 6, 6>();
}

TEST( SmallBlasL3, Gemm_211d )
{
	gemm_test<double, 2, 1, 1>();
}

TEST( SmallBlasL3, Gemm_212d )
{
	gemm_test<double, 2, 1, 2>();
}

TEST( SmallBlasL3, Gemm_214d )
{
	gemm_test<double, 2, 1, 4>();
}

TEST( SmallBlasL3, Gemm_216d )
{
	gemm_test<double, 2, 1, 6>();
}

TEST( SmallBlasL3, Gemm_221d )
{
	gemm_test<double, 2, 2, 1>();
}

TEST( SmallBlasL3, Gemm_222d )
{
	gemm_test<double, 2, 2, 2>();
}

TEST( SmallBlasL3, Gemm_224d )
{
	gemm_test<double, 2, 2, 4>();
}

TEST( SmallBlasL3, Gemm_226d )
{
	gemm_test<double, 2, 2, 6>();
}

TEST( SmallBlasL3, Gemm_241d )
{
	gemm_test<double, 2, 4, 1>();
}

TEST( SmallBlasL3, Gemm_242d )
{
	gemm_test<double, 2, 4, 2>();
}

TEST( SmallBlasL3, Gemm_244d )
{
	gemm_test<double, 2, 4, 4>();
}

TEST( SmallBlasL3, Gemm_246d )
{
	gemm_test<double, 2, 4, 6>();
}

TEST( SmallBlasL3, Gemm_261d )
{
	gemm_test<double, 2, 6, 1>();
}

TEST( SmallBlasL3, Gemm_262d )
{
	gemm_test<double, 2, 6, 2>();
}

TEST( SmallBlasL3, Gemm_264d )
{
	gemm_test<double, 2, 6, 4>();
}

TEST( SmallBlasL3, Gemm_266d )
{
	gemm_test<double, 2, 6, 6>();
}

TEST( SmallBlasL3, Gemm_411d )
{
	gemm_test<double, 4, 1, 1>();
}

TEST( SmallBlasL3, Gemm_412d )
{
	gemm_test<double, 4, 1, 2>();
}

TEST( SmallBlasL3, Gemm_414d )
{
	gemm_test<double, 4, 1, 4>();
}

TEST( SmallBlasL3, Gemm_416d )
{
	gemm_test<double, 4, 1, 6>();
}

TEST( SmallBlasL3, Gemm_421d )
{
	gemm_test<double, 4, 2, 1>();
}

TEST( SmallBlasL3, Gemm_422d )
{
	gemm_test<double, 4, 2, 2>();
}

TEST( SmallBlasL3, Gemm_424d )
{
	gemm_test<double, 4, 2, 4>();
}

TEST( SmallBlasL3, Gemm_426d )
{
	gemm_test<double, 4, 2, 6>();
}

TEST( SmallBlasL3, Gemm_441d )
{
	gemm_test<double, 4, 4, 1>();
}

TEST( SmallBlasL3, Gemm_442d )
{
	gemm_test<double, 4, 4, 2>();
}

TEST( SmallBlasL3, Gemm_444d )
{
	gemm_test<double, 4, 4, 4>();
}

TEST( SmallBlasL3, Gemm_446d )
{
	gemm_test<double, 4, 4, 6>();
}

TEST( SmallBlasL3, Gemm_461d )
{
	gemm_test<double, 4, 6, 1>();
}

TEST( SmallBlasL3, Gemm_462d )
{
	gemm_test<double, 4, 6, 2>();
}

TEST( SmallBlasL3, Gemm_464d )
{
	gemm_test<double, 4, 6, 4>();
}

TEST( SmallBlasL3, Gemm_466d )
{
	gemm_test<double, 4, 6, 6>();
}

TEST( SmallBlasL3, Gemm_611d )
{
	gemm_test<double, 6, 1, 1>();
}

TEST( SmallBlasL3, Gemm_612d )
{
	gemm_test<double, 6, 1, 2>();
}

TEST( SmallBlasL3, Gemm_614d )
{
	gemm_test<double, 6, 1, 4>();
}

TEST( SmallBlasL3, Gemm_616d )
{
	gemm_test<double, 6, 1, 6>();
}

TEST( SmallBlasL3, Gemm_621d )
{
	gemm_test<double, 6, 2, 1>();
}

TEST( SmallBlasL3, Gemm_622d )
{
	gemm_test<double, 6, 2, 2>();
}

TEST( SmallBlasL3, Gemm_624d )
{
	gemm_test<double, 6, 2, 4>();
}

TEST( SmallBlasL3, Gemm_626d )
{
	gemm_test<double, 6, 2, 6>();
}

TEST( SmallBlasL3, Gemm_641d )
{
	gemm_test<double, 6, 4, 1>();
}

TEST( SmallBlasL3, Gemm_642d )
{
	gemm_test<double, 6, 4, 2>();
}

TEST( SmallBlasL3, Gemm_644d )
{
	gemm_test<double, 6, 4, 4>();
}

TEST( SmallBlasL3, Gemm_646d )
{
	gemm_test<double, 6, 4, 6>();
}

TEST( SmallBlasL3, Gemm_661d )
{
	gemm_test<double, 6, 6, 1>();
}

TEST( SmallBlasL3, Gemm_662d )
{
	gemm_test<double, 6, 6, 2>();
}

TEST( SmallBlasL3, Gemm_664d )
{
	gemm_test<double, 6, 6, 4>();
}

TEST( SmallBlasL3, Gemm_666d )
{
	gemm_test<double, 6, 6, 6>();
}




