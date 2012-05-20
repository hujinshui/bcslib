/**
 * @file bench_matrix_reduction.cpp
 *
 * Bench mark of matrix (partial) reduction
 *
 * @author Dahua Lin
 */


#include "bench_tools.h"
#include <bcslib/matrix.h>
#include <cstdio>
#include <cstdlib>

using namespace bcs;

struct ColwiseReducTaskBase
{
	dense_matrix<double> x;
	dense_matrix<double> y;

	dense_row<double> res;

	ColwiseReducTaskBase(index_t m, index_t n)
	: x(m, n), y(m, n), res(n)
	{
		index_t len = m * n;
		for (index_t i = 0; i < len; ++i) x[i] = double(std::rand()) / RAND_MAX;
		for (index_t i = 0; i < len; ++i) y[i] = double(std::rand()) / RAND_MAX;
	}

	index_t size() const { return x.nelems(); }
};


struct RowwiseReducTaskBase
{
	dense_matrix<double> x;
	dense_matrix<double> y;

	dense_col<double> res;

	RowwiseReducTaskBase(index_t m, index_t n)
	: x(m, n), y(m, n), res(m)
	{
		index_t len = m * n;
		for (index_t i = 0; i < len; ++i) x[i] = double(std::rand()) / RAND_MAX;
		for (index_t i = 0; i < len; ++i) y[i] = double(std::rand()) / RAND_MAX;
	}

	index_t size() const { return x.nelems(); }
};


struct ColwiseSumForLoop : public ColwiseReducTaskBase
{
	ColwiseSumForLoop(index_t m, index_t n) : ColwiseReducTaskBase(m, n) { }

	void run()
	{
		const index_t m = x.nrows();
		const index_t n = x.ncolumns();

		for (index_t j = 0; j < n; ++j)
		{
			double s = 0;
			const double *cx = col_ptr(x, j);

#ifdef __INTEL_COMPILER
			#pragma simd reduction(+:s)
#endif
			for (index_t i = 0; i < m; ++i)
			{
				s += cx[i];
			}

			res[j] = s;
		}
	}

};


struct ColwiseSumMatrix : public ColwiseReducTaskBase
{
	ColwiseSumMatrix(index_t m, index_t n) : ColwiseReducTaskBase(m, n) { }

	void run()
	{
#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		res = sum(colwise(x));
	}
};

struct ColwiseSumPerCol : public ColwiseReducTaskBase
{
	ColwiseSumPerCol(index_t m, index_t n) : ColwiseReducTaskBase(m, n) { }

	void run()
	{
		const index_t n = x.ncolumns();

#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		for (index_t j = 0; j < n; ++j)
		{
			res[j] = sum(x.column(j));
		}
	}
};


struct ColwiseDotForLoop : public ColwiseReducTaskBase
{
	ColwiseDotForLoop(index_t m, index_t n) : ColwiseReducTaskBase(m, n) { }

	void run()
	{
		const index_t m = x.nrows();
		const index_t n = x.ncolumns();

		for (index_t j = 0; j < n; ++j)
		{
			double s = 0;
			const double *cx = col_ptr(x, j);
			const double *cy = col_ptr(y, j);

#ifdef __INTEL_COMPILER
			#pragma simd reduction(+:s)
#endif
			for (index_t i = 0; i < m; ++i)
			{
				s += cx[i] * cy[i];
			}

			res[j] = s;
		}
	}

};


struct ColwiseDotMatrix : public ColwiseReducTaskBase
{
	ColwiseDotMatrix(index_t m, index_t n) : ColwiseReducTaskBase(m, n) { }

	void run()
	{
#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		res = dot(colwise(x), colwise(y));
	}
};

struct ColwiseDotPerCol : public ColwiseReducTaskBase
{
	ColwiseDotPerCol(index_t m, index_t n) : ColwiseReducTaskBase(m, n) { }

	void run()
	{
		const index_t n = x.ncolumns();

#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		for (index_t j = 0; j < n; ++j)
		{
			res[j] = dot(x.column(j), y.column(j));
		}
	}
};


struct ColwiseL2DiffForLoop : public ColwiseReducTaskBase
{
	ColwiseL2DiffForLoop(index_t m, index_t n) : ColwiseReducTaskBase(m, n) { }

	void run()
	{
		const index_t m = x.nrows();
		const index_t n = x.ncolumns();

		for (index_t j = 0; j < n; ++j)
		{
			double s = 0;
			const double *cx = col_ptr(x, j);
			const double *cy = col_ptr(y, j);

#ifdef __INTEL_COMPILER
			#pragma simd reduction(+:s)
#endif
			for (index_t i = 0; i < m; ++i)
			{
				s += math::sqr( cx[i] - cy[i] );
			}

			res[j] = math::sqrt(s);
		}
	}

};


struct ColwiseL2DiffMatrix : public ColwiseReducTaskBase
{
	ColwiseL2DiffMatrix(index_t m, index_t n) : ColwiseReducTaskBase(m, n) { }

	void run()
	{
#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		res = L2norm_diff(colwise(x), colwise(y));
	}
};

struct ColwiseL2DiffPerCol : public ColwiseReducTaskBase
{
	ColwiseL2DiffPerCol(index_t m, index_t n) : ColwiseReducTaskBase(m, n) { }

	void run()
	{
		const index_t n = x.ncolumns();

#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		for (index_t j = 0; j < n; ++j)
		{
			res[j] = L2norm_diff(x.column(j), y.column(j));
		}
	}
};



struct RowwiseSumForLoop : public RowwiseReducTaskBase
{
	RowwiseSumForLoop(index_t m, index_t n) : RowwiseReducTaskBase(m, n) { }

	void run()
	{
		const index_t m = x.nrows();
		const index_t n = x.ncolumns();

		fill(res, 0.0);

		for (index_t j = 0; j < n; ++j)
		{
			double s = 0;
			const double *cx = col_ptr(x, j);

#ifdef __INTEL_COMPILER
			#pragma simd
#endif
			for (index_t i = 0; i < m; ++i)
			{
				res[i] += cx[i];
			}
		}
	}

};


struct RowwiseSumMatrix : public RowwiseReducTaskBase
{
	RowwiseSumMatrix(index_t m, index_t n) : RowwiseReducTaskBase(m, n) { }

	void run()
	{
#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		res = sum(rowwise(x));
	}
};

struct RowwiseSumPerCol : public RowwiseReducTaskBase
{
	RowwiseSumPerCol(index_t m, index_t n) : RowwiseReducTaskBase(m, n) { }

	void run()
	{
		const index_t n = x.ncolumns();

		fill(res, 0.0);

#ifdef __INTEL_COMPILER
		#pragma ivdep
#endif
		for (index_t j = 0; j < n; ++j)
		{
			res += x.column(j);
		}
	}
};





template<class Task>
void run(Task& tsk, const char *name, int ntimes)
{
	bench_stats bst = run_benchmark(tsk, 1, ntimes);
	std::printf("  %-26s:  %.2f M /sec\n", name, bst.MPS());
}


int main(int argc, char *argv[])
{
	std::printf("Partial reduction benchmark\n");
	std::printf("======================================\n");

	// colwise sum

	std::printf("colwise sum:\n");

	ColwiseSumForLoop colwise_sum_forloop(1000, 1000);
	run(colwise_sum_forloop, "colwise-sum-forloop", 500);

	ColwiseSumMatrix colwise_sum_matrix(1000, 1000);
	run(colwise_sum_matrix, "colwise-sum-matrix", 500);

	ColwiseSumPerCol colwise_sum_percol(1000, 1000);
	run(colwise_sum_percol, "colwise-sum-percol", 500);

	// colwise dot

	std::printf("colwise dot:\n");

	ColwiseDotForLoop colwise_dot_forloop(1000, 1000);
	run(colwise_dot_forloop, "colwise-dot-forloop", 500);

	ColwiseDotMatrix colwise_dot_matrix(1000, 1000);
	run(colwise_dot_matrix, "colwise-dot-matrix", 500);

	ColwiseDotPerCol colwise_dot_percol(1000, 1000);
	run(colwise_dot_percol, "colwise-dot-percol", 500);

	// colwise L2diff

	std::printf("colwise L2diff:\n");

	ColwiseL2DiffForLoop colwise_L2diff_forloop(1000, 1000);
	run(colwise_L2diff_forloop, "colwise-L2diff-forloop", 500);

	ColwiseL2DiffMatrix colwise_L2diff_matrix(1000, 1000);
	run(colwise_L2diff_matrix, "colwise-L2diff-matrix", 500);

	ColwiseL2DiffPerCol colwise_L2diff_percol(1000, 1000);
	run(colwise_L2diff_percol, "colwise-L2diff-percol", 500);

	// rowwise sum

	std::printf("rowwise sum:\n");

	RowwiseSumForLoop rowwise_sum_forloop(1000, 1000);
	run(rowwise_sum_forloop, "rowwise-sum-forloop", 500);

	RowwiseSumMatrix rowwise_sum_matrix(1000, 1000);
	run(rowwise_sum_matrix, "rowwise-sum-matrix", 500);

	RowwiseSumPerCol rowwise_sum_percol(1000, 1000);
	run(rowwise_sum_percol, "rowwise-sum-percol", 500);

	std::printf("\n");
}






