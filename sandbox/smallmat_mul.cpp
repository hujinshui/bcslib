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


template<index_t M, index_t N, index_t K>
struct mm_a
{
	template<typename T>
	static void run(const T* a, const T* b, T *c)
	{
		#pragma ivdep
		for (index_t j = 0; j < N; ++j)
		{
			for (index_t i = 0; i < M; ++i)
			{
				T s(0);

				for (index_t k = 0; k < K; ++k)
					s += a[i + k * M] * b[k + j * K];

				c[i + j * M] += s;
			}
		}
	}
};

template<index_t M, index_t N, index_t K>
struct mm_b
{
	template<typename T>
	static void run(const T* a, const T* b, T *c)
	{
		#pragma ivdep
		for (index_t k = 0; k < K; ++k)
		{
			const T *ak = a + k * M;

			for (index_t j = 0; j < N; ++j)
			{
				const T bv = b[k + j * K];

				for (index_t i = 0; i < M; ++i)
				{
					c[i + j * M] += ak[i] * bv;
				}
			}
		}
	}
};



void report_res(int m, int n, int k,
		const int rt_a, const double s_a,
		const int rt_b, const double s_b,
		const int rt_mkl, const double s_mkl)
{
	double g_a = (double(2 * m * n * k) * double(rt_a) * 1.0e-9) / s_a;
	double g_b = (double(2 * m * n * k) * double(rt_b) * 1.0e-9) / s_b;
	double g_m = (double(2 * m * n * k) * double(rt_mkl) * 1.0e-9) / s_mkl;

	std::printf("%3d, %3d, %3d,   %7.4f, %7.4f, %7.4f\n",
			m, n, k, g_a, g_b, g_m);
}


template<index_t M, index_t N, index_t K>
void test_main()
{
	typedef double real;

	dense_matrix<real> a(M, K);
	dense_matrix<real> b(K, N);
	dense_matrix<real> c(M, N, 0.0);

	for (int i = 0; i < M * K; ++i) a[i] = real(i + 1);
	for (int i = 0; i < K * N; ++i) b[i] = real(i + 1);
	for (int i = 0; i < M * N; ++i) c[i] = real(2 * i + 1);

	int rt0 = int(double(2e8) / double(M * N * K));

	// test mm_a

	const int rt_a = rt0;

	timer tm_a;
	tm_a.start();

	for (int i = 0; i < rt_a; ++i)
	{
		mm_a<M, N, K>::run(a.ptr_data(), b.ptr_data(), c.ptr_data());
	}

	double s_a = tm_a.elapsed_secs();

	// test mm_b

	const int rt_b = rt0;

	timer tm_b;
	tm_b.start();

	for (int i = 0; i < rt_b; ++i)
	{
		mm_b<M, N, K>::run(a.ptr_data(), b.ptr_data(), c.ptr_data());
	}

	double s_b = tm_b.elapsed_secs();

	// test mkl

	const int rt_mkl = 1000000;

	for (int i = 0; i < 10; ++i) c = mm(a, b); // warming

	timer tm_mkl;
	tm_mkl.start();

	for (int i = 0; i < rt_mkl; ++i)
	{
		c += mm(a, b);
	}

	double s_mkl = tm_mkl.elapsed_secs();

	report_res(M, N, K, rt_a, s_a, rt_b, s_b, rt_mkl, s_mkl);

}



int main(int argc, char *argv[])
{
	std::printf("M, N, K, mm_a, mm_b, mkl\n");

	test_main<2, 2, 2>();
	test_main<2, 2, 3>();
	test_main<2, 2, 4>();
	test_main<2, 2, 6>();
	test_main<2, 2, 8>();

	test_main<2, 3, 2>();
	test_main<2, 3, 3>();
	test_main<2, 3, 4>();
	test_main<2, 3, 6>();
	test_main<2, 3, 8>();

	test_main<2, 4, 2>();
	test_main<2, 4, 3>();
	test_main<2, 4, 4>();
	test_main<2, 4, 6>();
	test_main<2, 4, 8>();

	test_main<2, 6, 2>();
	test_main<2, 6, 3>();
	test_main<2, 6, 4>();
	test_main<2, 6, 6>();
	test_main<2, 6, 8>();

	test_main<2, 8, 2>();
	test_main<2, 8, 3>();
	test_main<2, 8, 4>();
	test_main<2, 8, 6>();
	test_main<2, 8, 8>();

	test_main<3, 2, 2>();
	test_main<3, 2, 3>();
	test_main<3, 2, 4>();
	test_main<3, 2, 6>();
	test_main<3, 2, 8>();

	test_main<3, 3, 2>();
	test_main<3, 3, 3>();
	test_main<3, 3, 4>();
	test_main<3, 3, 6>();
	test_main<3, 3, 8>();

	test_main<3, 4, 2>();
	test_main<3, 4, 3>();
	test_main<3, 4, 4>();
	test_main<3, 4, 6>();
	test_main<3, 4, 8>();

	test_main<3, 6, 2>();
	test_main<3, 6, 3>();
	test_main<3, 6, 4>();
	test_main<3, 6, 6>();
	test_main<3, 6, 8>();

	test_main<3, 8, 2>();
	test_main<3, 8, 3>();
	test_main<3, 8, 4>();
	test_main<3, 8, 6>();
	test_main<3, 8, 8>();

	test_main<4, 2, 2>();
	test_main<4, 2, 3>();
	test_main<4, 2, 4>();
	test_main<4, 2, 6>();
	test_main<4, 2, 8>();

	test_main<4, 3, 2>();
	test_main<4, 3, 3>();
	test_main<4, 3, 4>();
	test_main<4, 3, 6>();
	test_main<4, 3, 8>();

	test_main<4, 4, 2>();
	test_main<4, 4, 3>();
	test_main<4, 4, 4>();
	test_main<4, 4, 6>();
	test_main<4, 4, 8>();

	test_main<4, 6, 2>();
	test_main<4, 6, 3>();
	test_main<4, 6, 4>();
	test_main<4, 6, 6>();
	test_main<4, 6, 8>();

	test_main<4, 8, 2>();
	test_main<4, 8, 3>();
	test_main<4, 8, 4>();
	test_main<4, 8, 6>();
	test_main<4, 8, 8>();

	test_main<6, 2, 2>();
	test_main<6, 2, 3>();
	test_main<6, 2, 4>();
	test_main<6, 2, 6>();
	test_main<6, 2, 8>();

	test_main<6, 3, 2>();
	test_main<6, 3, 3>();
	test_main<6, 3, 4>();
	test_main<6, 3, 6>();
	test_main<6, 3, 8>();

	test_main<6, 4, 2>();
	test_main<6, 4, 3>();
	test_main<6, 4, 4>();
	test_main<6, 4, 6>();
	test_main<6, 4, 8>();

	test_main<6, 6, 2>();
	test_main<6, 6, 3>();
	test_main<6, 6, 4>();
	test_main<6, 6, 6>();
	test_main<6, 6, 8>();

	test_main<6, 8, 2>();
	test_main<6, 8, 3>();
	test_main<6, 8, 4>();
	test_main<6, 8, 6>();
	test_main<6, 8, 8>();

	test_main<8, 2, 2>();
	test_main<8, 2, 3>();
	test_main<8, 2, 4>();
	test_main<8, 2, 6>();
	test_main<8, 2, 8>();

	test_main<8, 3, 2>();
	test_main<8, 3, 3>();
	test_main<8, 3, 4>();
	test_main<8, 3, 6>();
	test_main<8, 3, 8>();

	test_main<8, 4, 2>();
	test_main<8, 4, 3>();
	test_main<8, 4, 4>();
	test_main<8, 4, 6>();
	test_main<8, 4, 8>();

	test_main<8, 6, 2>();
	test_main<8, 6, 3>();
	test_main<8, 6, 4>();
	test_main<8, 6, 6>();
	test_main<8, 6, 8>();

	test_main<8, 8, 2>();
	test_main<8, 8, 3>();
	test_main<8, 8, 4>();
	test_main<8, 8, 6>();
	test_main<8, 8, 8>();
}






