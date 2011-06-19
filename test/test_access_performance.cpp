/**
 * @file test_access_performance.cpp
 *
 * The program to test array's element access performance
 * 
 * @author Dahua Lin
 */

#include <cstdio>
#include <cstdlib>

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/test/performance_timer.h>

#include <algorithm>


using namespace bcs;
using namespace bcs::test;


void test_copy_syntax()
{
	double src[100];

	const_aview1d<double, step_ind> view(src, step_ind(10, 2));

	std::copy(view.begin(), view.end(), src);
}


double time_copy_memory(int nrepeats, const size_t nelems, const double *src, double *buf)
{
	copy_elements(src, buf, nelems);

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		bcs::copy_elements(src, buf, nelems);
	}

	return timer.elapsed_seconds();
}


double time_raw_for_loop(int nrepeats, const size_t nelems, const double *src, double *buf)
{
	copy_elements(src, buf, nelems);

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		for (size_t j = 0; j < nelems; ++j)
		{
			buf[j] = src[j];
		}
	}

	return timer.elapsed_seconds();
}


double time_dense1d_export(int nrepeats, const size_t nelems, const double *src, double *buf)
{
	const_aview1d<double> view(src, nelems);

	export_to(view, buf);

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		export_to(view, buf);
	}

	return timer.elapsed_seconds();

}


double time_dense1d_access(int nrepeats, const size_t nelems, const double *src, double *buf)
{
	const_aview1d<double> view(src, nelems);

	index_t n = (index_t)nelems;

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			buf[j] = view(j);
		}
	}

	return timer.elapsed_seconds();
}


double time_step1d_export(int nrepeats, const size_t nelems, const double *src, double *buf)
{
	const_aview1d<double, step_ind> view(src, step_ind(nelems, 2));

	export_to(view, buf);

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		export_to(view, buf);
	}

	return timer.elapsed_seconds();
}


double time_step1d_access(int nrepeats, const size_t nelems, const double *src, double *buf)
{
	const_aview1d<double, step_ind> view(src, step_ind(nelems, 2));

	index_t n = (index_t)nelems;

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			buf[j] = view(j);
		}
	}

	return timer.elapsed_seconds();
}


double time_dense2d_rm_export(int nrepeats, const size_t m, const size_t n, const double *src, double *buf)
{
	const_aview2d<double, row_major_t> view(src, (index_t)m, (index_t)n, id_ind(m), id_ind(n));

	export_to(view, buf);

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		export_to(view, buf);
	}

	return timer.elapsed_seconds();
}


double time_dense2d_cm_export(int nrepeats, const size_t m, const size_t n, const double *src, double *buf)
{
	const_aview2d<double, column_major_t> view(src, (index_t)m, (index_t)n, id_ind(m), id_ind(n));

	export_to(view, buf);

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		export_to(view, buf);
	}
	return timer.elapsed_seconds();

}


double time_dense2d_rm_access(int nrepeats, const size_t m, const size_t n, const double *src, double *buf)
{
	const_aview2d<double, row_major_t> view(src, (index_t)m, (index_t)n, id_ind(m), id_ind(n));

	index_t d0 = (index_t)m;
	index_t d1 = (index_t)n;

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		double *p = buf;
		for (index_t j0 = 0; j0 < d0; ++j0)
		{
			for (index_t j1 = 0; j1 < d1; ++j1)
			{
				*p++ = view(j0, j1);
			}
		}
	}

	return timer.elapsed_seconds();
}


double time_dense2d_cm_access(int nrepeats, const size_t m, const size_t n, const double *src, double *buf)
{
	const_aview2d<double, column_major_t> view(src, (index_t)m, (index_t)n, id_ind(m), id_ind(n));

	index_t d0 = (index_t)m;
	index_t d1 = (index_t)n;

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		double *p = buf;
		for (index_t j1 = 0; j1 < d1; ++j1)
		{
			for (index_t j0 = 0; j0 < d0; ++j0)
			{
				*p++ = view(j0, j1);
			}
		}
	}

	return timer.elapsed_seconds();
}


double time_step2d_rm_export(int nrepeats, const size_t m, const size_t n, const double *src, double *buf)
{
	const_aview2d<double, row_major_t, step_ind, step_ind> view(src, (index_t)(2*m), (index_t)(2*n), step_ind(m, 2), step_ind(n,2));

	export_to(view, buf);

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		export_to(view, buf);
	}

	return timer.elapsed_seconds();
}


double time_step2d_cm_export(int nrepeats, const size_t m, const size_t n, const double *src, double *buf)
{
	const_aview2d<double, column_major_t, step_ind, step_ind> view(src, (index_t)(2*m), (index_t)(2*n), step_ind(m,2), step_ind(n,2));

	export_to(view, buf);

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		export_to(view, buf);
	}

	return timer.elapsed_seconds();

}


double time_step2d_rm_access(int nrepeats, const size_t m, const size_t n, const double *src, double *buf)
{
	const_aview2d<double, row_major_t, step_ind, step_ind> view(src, (index_t)(2*m), (index_t)(2*n), step_ind(m, 2), step_ind(n,2));

	index_t d0 = (index_t)m;
	index_t d1 = (index_t)n;

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		double *p = buf;
		for (index_t j0 = 0; j0 < d0; ++j0)
		{
			for (index_t j1 = 0; j1 < d1; ++j1)
			{
				*p++ = view(j0, j1);
			}
		}
	}

	return timer.elapsed_seconds();
}


double time_step2d_cm_access(int nrepeats, const size_t m, const size_t n, const double *src, double *buf)
{
	const_aview2d<double, column_major_t, step_ind, step_ind> view(src, (index_t)(2*m), (index_t)(2*n), step_ind(m,2), step_ind(n,2));

	index_t d0 = (index_t)m;
	index_t d1 = (index_t)n;

	performance_timer timer(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		double *p = buf;
		for (index_t j1 = 0; j1 < d1; ++j1)
		{
			for (index_t j0 = 0; j0 < d0; ++j0)
			{
				*p++ = view(j0, j1);
			}
		}
	}

	return timer.elapsed_seconds();

}




int main(int argc, char *argv[])
{
	const size_t nrows = 1000;
	const size_t ncols = 1000;
	const size_t nelems = nrows * ncols;
	const size_t nelems_s = 5 * nelems;

	const int nrepeats = 100;

	// generate random source

	std::printf("preparing data source ...\n");

	double *src = new double[nelems_s];

	std::srand( (unsigned int)(std::time( 0 )) );
	for (size_t i = 0; i < nelems_s; ++i)
	{
		src[i] = (double)std::rand() / RAND_MAX;
	}

	double *buf = new double[nelems];

	// time copy_memory

	std::printf("testing copy_memory ...\n");
	double e_copymem = time_copy_memory(nrepeats, nelems, src, buf);

	std::printf("testing raw_for_loop ...\n");
	double e_rawloop = time_raw_for_loop(nrepeats, nelems, src, buf);

	std::printf("testing dense1d_export ...\n");
	double e_dense1d_export = time_dense1d_export(nrepeats, nelems, src, buf);

	std::printf("testing dense1d_access ...\n");
	double e_dense1d_access = time_dense1d_access(nrepeats, nelems, src, buf);

	std::printf("testing step1d_export ...\n");
	double e_step1d_export = time_step1d_export(nrepeats, nelems, src, buf);

	std::printf("testing step1d_access ...\n");
	double e_step1d_access = time_step1d_access(nrepeats, nelems, src, buf);

	std::printf("testing dense2d_rm_export ...\n");
	double e_dense2d_rm_export = time_dense2d_rm_export(nrepeats, nrows, ncols, src, buf);

	std::printf("testing dense2d_cm_export ...\n");
	double e_dense2d_cm_export = time_dense2d_cm_export(nrepeats, nrows, ncols, src, buf);

	std::printf("testing dense2d_rm_access ...\n");
	double e_dense2d_rm_access = time_dense2d_rm_access(nrepeats, nrows, ncols, src, buf);

	std::printf("testing dense2d_cm_access ...\n");
	double e_dense2d_cm_access = time_dense2d_cm_access(nrepeats, nrows, ncols, src, buf);

	std::printf("testing step2d_rm_export ...\n");
	double e_step2d_rm_export = time_step2d_rm_export(nrepeats, nrows, ncols, src, buf);

	std::printf("testing step2d_cm_export ...\n");
	double e_step2d_cm_export = time_step2d_cm_export(nrepeats, nrows, ncols, src, buf);

	std::printf("testing step2d_rm_access ...\n");
	double e_step2d_rm_access = time_step2d_rm_access(nrepeats, nrows, ncols, src, buf);

	std::printf("testing step2d_cm_access ...\n");
	double e_step2d_cm_access = time_step2d_cm_access(nrepeats, nrows, ncols, src, buf);

	// output

	std::printf("Elapsed time:\n");
	std::printf("=================\n");
	std::printf("\tcopy_memory:    %.4f s\n", e_copymem);
	std::printf("\traw_for_loop:   %.4f s\n", e_rawloop);
	std::printf("\n");
	std::printf("\tdense1d_export: %.4f s\n", e_dense1d_export);
	std::printf("\tdense1d_access: %.4f s\n", e_dense1d_access);
	std::printf("\tstep1d_export:  %.4f s\n", e_step1d_export);
	std::printf("\tstep1d_access:  %.4f s\n", e_step1d_access);
	std::printf("\n");
	std::printf("\tdense2d_rm_export: %.4f s\n", e_dense2d_rm_export);
	std::printf("\tdense2d_cm_export: %.4f s\n", e_dense2d_cm_export);
	std::printf("\tdense2d_rm_access: %.4f s\n", e_dense2d_rm_access);
	std::printf("\tdense2d_cm_access: %.4f s\n", e_dense2d_cm_access);
	std::printf("\tstep2d_rm_export: %.4f s\n", e_step2d_rm_export);
	std::printf("\tstep2d_cm_export: %.4f s\n", e_step2d_cm_export);
	std::printf("\tstep2d_rm_access: %.4f s\n", e_step2d_rm_access);
	std::printf("\tstep2d_cm_access: %.4f s\n", e_step2d_cm_access);
	std::printf("\n");

	delete[] src;
	delete[] buf;
}


