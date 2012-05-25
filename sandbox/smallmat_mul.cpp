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
void verify_M()
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


void verify_all()
{
	verify_M<1>();
	verify_M<2>();
	verify_M<4>();
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

/*
template<int M>
void do_bench()
{
	bench_mm<M, 1, 1>();
	bench_mm<M, 1, 2>();
	bench_mm<M, 1, 3>();
	bench_mm<M, 1, 4>();
	bench_mm<M, 1, 6>();
	bench_mm<M, 1, 8>();

	bench_mm<M, 2, 1>();
	bench_mm<M, 2, 2>();
	bench_mm<M, 2, 3>();
	bench_mm<M, 2, 4>();
	bench_mm<M, 2, 6>();
	bench_mm<M, 2, 8>();

	bench_mm<M, 3, 1>();
	bench_mm<M, 3, 2>();
	bench_mm<M, 3, 3>();
	bench_mm<M, 3, 4>();
	bench_mm<M, 3, 6>();
	bench_mm<M, 3, 8>();

	bench_mm<M, 4, 1>();
	bench_mm<M, 4, 2>();
	bench_mm<M, 4, 3>();
	bench_mm<M, 4, 4>();
	bench_mm<M, 4, 6>();
	bench_mm<M, 4, 8>();

	bench_mm<M, 6, 1>();
	bench_mm<M, 6, 2>();
	bench_mm<M, 6, 3>();
	bench_mm<M, 6, 4>();
	bench_mm<M, 6, 6>();
	bench_mm<M, 6, 8>();

	bench_mm<M, 8, 1>();
	bench_mm<M, 8, 2>();
	bench_mm<M, 8, 3>();
	bench_mm<M, 8, 4>();
	bench_mm<M, 8, 6>();
	bench_mm<M, 8, 8>();
}

void do_all_bench()
{
	std::printf("M, K, N, mm0, smm, mkl\n");

	do_bench<1>();
	do_bench<2>();
	do_bench<3>();
	do_bench<4>();
	do_bench<6>();
	do_bench<8>();
}
*/


int main(int argc, char *argv[])
{
	// do_all_bench();
	verify_all();
}






