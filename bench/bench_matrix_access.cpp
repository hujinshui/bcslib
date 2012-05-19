/**
 * @file bench_matrix_access.cpp
 *
 * The program to test matrix's element access performance
 * 
 * @author Dahua Lin
 */

#include "bench_tools.h"
#include <bcslib/matrix.h>
#include <cstdio>

using namespace bcs;


struct CopyTaskBase
{
	dense_matrix<double> src;
	dense_matrix<double> dst;

	CopyTaskBase(index_t m, index_t n)
	: src(m, n), dst(m, n)
	{
		index_t len = m * n;
		for (index_t i = 0; i < len; ++i) src[i] = double(i);
	}

	index_t size() const { return src.nelems(); }
};

struct MemCopyTask : public CopyTaskBase
{
	MemCopyTask(index_t m, index_t n) : CopyTaskBase(m, n) { }

	void run()
	{
		std::memcpy(dst.ptr_data(), src.ptr_data(), src.size() * sizeof(double));
	}
};

struct DirectCopyTask : public CopyTaskBase
{
	DirectCopyTask(index_t m, index_t n) : CopyTaskBase(m, n) { }

	void run()
	{
		bcs::copy(src, dst);
	}
};

struct PerColCopyTask : public CopyTaskBase
{
	PerColCopyTask(index_t m, index_t n) : CopyTaskBase(m, n) { }

	void run()
	{
		const index_t n = src.ncolumns();
		for (index_t j = 0; j < n; ++j)
		{
			dst.column(j) = src.column(j);
		}
	}
};


struct LinearAccessTask : public CopyTaskBase
{
	LinearAccessTask(index_t m, index_t n) : CopyTaskBase(m, n) { }

	void run()
	{
		const index_t len = src.nelems();
		for (index_t i = 0; i < len; ++i) dst[i] = src[i];
	}
};

struct SubscriptAccessTask : public CopyTaskBase
{
	SubscriptAccessTask(index_t m, index_t n) : CopyTaskBase(m, n) { }

	void run()
	{
		const index_t m = src.nrows();
		const index_t n = src.ncolumns();
		for (index_t j = 0; j < n; ++j)
		{
			for (index_t i = 0; i < m; ++i)
			{
				dst(i, j) = src(i, j);
			}
		}
	}
};



template<class Task>
void run(Task& tsk, const char *name, int ntimes)
{
	bench_stats bst = run_benchmark(tsk, 1, ntimes);
	std::printf("%16s:  %.2f M doubles/sec ==> %.2f MBytes/sec\n", name, bst.MPS(), bst.MPS() * 8);
}


int main(int argc, char *argv[])
{
	std::printf("Matrix access benchmark\n");
	std::printf("==============================\n");

	MemCopyTask mem_copy(1000, 1000);
	run(mem_copy, "mem-copy", 500);

	DirectCopyTask direct_copy(1000, 1000);
	run(direct_copy, "direct-copy", 500);

	DirectCopyTask percol_copy(1000, 1000);
	run(percol_copy, "per-column-copy", 500);

	LinearAccessTask linear_acc(1000, 1000);
	run(linear_acc, "linear-access", 500);

	LinearAccessTask subscript_acc(1000, 1000);
	run(subscript_acc, "subscript-access", 500);

	std::printf("\n");
}


