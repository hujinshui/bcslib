/**
 * @file bench_ewise_calc.cpp
 *
 * Benchmark of element-wise calculation
 *
 * @author Dahua Lin
 */


#include "bench_tools.h"
#include <bcslib/matrix.h>
#include <cstdio>
#include <cstdlib>

using namespace bcs;


struct ECalcTaskBase
{
	dense_matrix<double> x;
	dense_matrix<double> y;
	dense_matrix<double> z;

	dense_matrix<double> res;

	ECalcTaskBase(index_t m, index_t n)
	: x(m, n), y(m, n), z(m, n)
	, res(m, n)
	{
		index_t len = m * n;
		for (index_t i = 0; i < len; ++i) x[i] = double(std::rand()) / RAND_MAX;
		for (index_t i = 0; i < len; ++i) y[i] = double(std::rand()) / RAND_MAX;
		for (index_t i = 0; i < len; ++i) z[i] = double(std::rand()) / RAND_MAX;
	}

	index_t size() const { return x.nelems(); }
};


struct DirectCalc : public ECalcTaskBase
{
	DirectCalc(index_t m, index_t n) : ECalcTaskBase(m, n) { }

	void run()
	{
		index_t len = x.nelems();

		#pragma simd
		for (index_t i = 0; i < len; ++i)
		{
			res[i] = std::log(std::exp(x[i] - y[i]) + z[i]);
		}
	}
};


struct MatrixCalc : public ECalcTaskBase
{
	MatrixCalc(index_t m, index_t n) : ECalcTaskBase(m, n) { }

	void run()
	{
		res = log(exp(x + y) + z);
	}
};


template<class Task>
void run(Task& tsk, const char *name, int ntimes)
{
	bench_stats bst = run_benchmark(tsk, 1, ntimes);
	std::printf("%16s:  %.2f M /sec\n", name, bst.MPS());
}


int main(int argc, char *argv[])
{
	std::printf("Element-wise evaluation benchmark\n");
	std::printf("======================================\n");

	DirectCalc direct_calc(1000, 1000);
	run(direct_calc, "direct-calc", 100);

	DirectCalc matrix_calc(1000, 1000);
	run(matrix_calc, "matrix-calc", 100);

	std::printf("\n");
}



