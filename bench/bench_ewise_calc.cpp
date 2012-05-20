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
		const index_t len = x.nelems();

#ifdef __INTEL_COMPILER
		#pragma simd
#endif
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
#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		res = log(exp(x - y) + z);
	}
};

struct PerColCalc : public ECalcTaskBase
{
	PerColCalc(index_t m, index_t n) : ECalcTaskBase(m, n) { }

	void run()
	{
		const index_t n = x.ncolumns();

#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		for (index_t j = 0; j < n; ++j)
		{
			res.column(j) = log(exp(x.column(j) - y.column(j)) + z.column(j));
		}
	}
};


struct BsxCalcTaskBase
{
	dense_col<double> x;
	dense_row<double> y;
	dense_matrix<double> z;

	dense_matrix<double> res;

	BsxCalcTaskBase(index_t m, index_t n)
	: x(m), y(n), z(m, n)
	, res(m, n)
	{
		index_t len = m * n;
		for (index_t i = 0; i < m; ++i) x[i] = double(std::rand()) / RAND_MAX;
		for (index_t i = 0; i < n; ++i) y[i] = double(std::rand()) / RAND_MAX;
		for (index_t i = 0; i < len; ++i) z[i] = double(std::rand()) / RAND_MAX;
	}

	index_t size() const { return z.nelems(); }
};


struct DirectBsxCalc : public BsxCalcTaskBase
{
	DirectBsxCalc(index_t m, index_t n) : BsxCalcTaskBase(m, n) { }

	void run()
	{
		const index_t m = z.nrows();
		const index_t n = z.ncolumns();

		for (index_t j = 0; j < n; ++j)
		{
			double yj = y[j];
			const double* cz = col_ptr(z, j);
			double *r = col_ptr(res, j);

#ifdef __INTEL_COMPILER
			#pragma simd
#endif
			for (index_t i = 0; i < m; ++i)
			{
				r[i] = (x[i] + yj) * std::exp(cz[i]);
			}
		}
	}
};


struct MatrixBsxCalc : public BsxCalcTaskBase
{
	MatrixBsxCalc(index_t m, index_t n) : BsxCalcTaskBase(m, n) { }

	void run()
	{
		const index_t m = z.nrows();
		const index_t n = z.ncolumns();

#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		res = (repeat_cols(x, n) + repeat_rows(y, m)) * exp(z);
	}
};




template<class Task>
void run(Task& tsk, const char *name, int ntimes)
{
	bench_stats bst = run_benchmark(tsk, 1, ntimes);
	std::printf("  %-16s:  %.2f M /sec\n", name, bst.MPS());
}


int main(int argc, char *argv[])
{
	std::printf("Element-wise evaluation benchmark\n");
	std::printf("======================================\n");

	DirectCalc direct_calc(1000, 1000);
	run(direct_calc, "direct-calc", 100);

	MatrixCalc matrix_calc(1000, 1000);
	run(matrix_calc, "matrix-calc", 100);

	PerColCalc percol_calc(1000, 1000);
	run(percol_calc, "percol-calc", 100);

	std::printf("broadcast evaluation benchmark\n");
	std::printf("======================================\n");

	DirectBsxCalc direct_bsx_calc(1000, 1000);
	run(direct_bsx_calc, "direct-bsx-calc", 200);

	MatrixBsxCalc matrix_bsx_calc(1000, 1000);
	run(matrix_bsx_calc, "matrix-bsx-calc", 200);

	std::printf("\n");
}



