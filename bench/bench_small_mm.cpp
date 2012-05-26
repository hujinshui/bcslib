/**
 * @file bench_small_mm.cpp
 *
 * Benchmark of small fixed-size matrix product
 *
 * @author Dahua Lin
 */


#include "bench_tools.h"
#include <bcslib/matrix.h>
#include <bcslib/engine/small_blasL3.h>

#include <cstdio>
#include <cstdlib>

using namespace bcs;

template<typename T> struct type_nam;

template<> struct type_nam<float>  { static const char *get() { return "single"; } };
template<> struct type_nam<double> { static const char *get() { return "double"; } };


template<typename T, int M_, int N_, int K_>
struct SmallMM
{
	typedef T value_type;
	static const int M = M_;
	static const int N = N_;
	static const int K = K_;

	SmallMM() : a(M, K), b(K, N), c(M, N)
	{
		for (int i = 0; i < M * K; ++i) a[i] = T(std::rand()) / T(RAND_MAX);
		for (int i = 0; i < K * N; ++i) b[i] = T(std::rand()) / T(RAND_MAX);
	}

	void run()
	{
		engine::small_gemm_ker<T, M, N, K>::eval_nn_b0(T(1),
				a.ptr_data(), a.lead_dim(),
				b.ptr_data(), b.lead_dim(),
				c.ptr_data(), c.lead_dim());
	}

	int size() const
	{
		return M * N * K * 2;
	}

	dense_matrix<T> a;
	dense_matrix<T> b;
	dense_matrix<T> c;
};


template<class Task>
void run(Task& tsk, long ntimes)
{
	const int M = Task::M;
	const int N = Task::N;
	const int K = Task::K;

	bench_stats bst = run_benchmark(tsk, 10, ntimes);

	std::printf("%s, %d, %d, %d,  %.6f, %.4f\n",
			type_nam<typename Task::value_type>::get(),
			M, N, K, bst.GPS(), bst.elapsed_secs * 1.0e3);
}


long calc_times(int M, int N, int K)
{
	long base = 200000000;
	return base / long(M * N * K * 2);
}


template<typename T, int M, int N>
void run_on_size()
{
	SmallMM<T, M, N, 1> t1;
	SmallMM<T, M, N, 2> t2;
	SmallMM<T, M, N, 3> t3;
	SmallMM<T, M, N, 4> t4;
	SmallMM<T, M, N, 6> t6;
	SmallMM<T, M, N, 8> t8;

	run(t1, calc_times(M, N, 1));
	run(t2, calc_times(M, N, 2));
	run(t3, calc_times(M, N, 3));
	run(t4, calc_times(M, N, 4));
	run(t6, calc_times(M, N, 6));
	run(t8, calc_times(M, N, 8));
}


template<typename T>
void run_all()
{
	run_on_size<T, 1, 1>();
	run_on_size<T, 1, 2>();
	run_on_size<T, 1, 3>();
	run_on_size<T, 1, 4>();
	run_on_size<T, 1, 6>();
	run_on_size<T, 1, 8>();

	run_on_size<T, 2, 1>();
	run_on_size<T, 2, 2>();
	run_on_size<T, 2, 3>();
	run_on_size<T, 2, 4>();
	run_on_size<T, 2, 6>();
	run_on_size<T, 2, 8>();

	run_on_size<T, 3, 1>();
	run_on_size<T, 3, 2>();
	run_on_size<T, 3, 3>();
	run_on_size<T, 3, 4>();
	run_on_size<T, 3, 6>();
	run_on_size<T, 3, 8>();

	run_on_size<T, 4, 1>();
	run_on_size<T, 4, 2>();
	run_on_size<T, 4, 3>();
	run_on_size<T, 4, 4>();
	run_on_size<T, 4, 6>();
	run_on_size<T, 4, 8>();

	run_on_size<T, 6, 1>();
	run_on_size<T, 6, 2>();
	run_on_size<T, 6, 3>();
	run_on_size<T, 6, 4>();
	run_on_size<T, 6, 6>();
	run_on_size<T, 6, 8>();

	run_on_size<T, 8, 1>();
	run_on_size<T, 8, 2>();
	run_on_size<T, 8, 3>();
	run_on_size<T, 8, 4>();
	run_on_size<T, 8, 6>();
	run_on_size<T, 8, 8>();
}



int main(int argc, char *argv[])
{
	std::printf("type, M, N, K, GFlops, elapsed (ms)\n");
	run_all<float>();
	run_all<double>();
}







