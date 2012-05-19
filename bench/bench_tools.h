/**
 * @file bench_tools.h
 *
 * Tools to support benchmarking
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_BENCH_TOOLS_H_
#define BCSLIB_BENCH_TOOLS_H_

#include <bcslib/core/basic_defs.h>
#include <bcslib/utils/timer.h>

namespace bcs
{

	struct bench_stats
	{
		index_t size;
		int ntimes;
		double elapsed_secs;

		double MPS() const
		{
			return double(size) * double(ntimes) * 1.0e-6 / elapsed_secs;
		}

		double GPS() const
		{
			return double(size) * double(ntimes) * 1.0e-9 / elapsed_secs;
		}
	};


	template<class Task>
	inline bench_stats run_benchmark(Task& tsk, int n_warmup, int n_main)
	{
		for (int i = 0; i < n_warmup; ++i) tsk.run();

		timer tm(true);
		for (int i = 0; i < n_main; ++i) tsk.run();
		double e = tm.elapsed_secs();

		bench_stats bs;
		bs.size = static_cast<index_t>(tsk.size());
		bs.ntimes = n_main;
		bs.elapsed_secs = e;

		return bs;
	}

}

#endif
