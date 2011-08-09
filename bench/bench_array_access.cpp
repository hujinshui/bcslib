/**
 * @file bench_array_access.cpp
 *
 * The program to test array's element access performance
 * 
 * @author Dahua Lin
 */

#include <cstdio>
#include <cstdlib>

#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>
#include <bcslib/base/timer.h>

#include <algorithm>
#include <ctime>


using namespace bcs;


double time_copy_memory(int nrepeats, const index_t nelems, const double *src, double *buf)
{
	copy_elements(src, buf, nelems);

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		bcs::copy_elements(src, buf, nelems);
	}

	return tm.elapsed( SECONDS );
}


double time_raw_for_loop(int nrepeats, const index_t nelems, const double *src, double *buf)
{
	copy_elements(src, buf, nelems);

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		for (index_t j = 0; j < nelems; ++j)
		{
			buf[j] = src[j];
		}
	}

	return tm.elapsed( SECONDS );
}


double time_dense1d_export(int nrepeats, const index_t nelems, const double *src, double *buf)
{
	caview1d<double> view(src, nelems);

	view.export_to(buf);

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		view.export_to(buf);
	}

	return tm.elapsed( SECONDS );

}


double time_dense1d_access(int nrepeats, const index_t nelems, const double *src, double *buf)
{
	caview1d<double> view(src, nelems);

	index_t n = (index_t)nelems;

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			buf[j] = view(j);
		}
	}

	return tm.elapsed( SECONDS );
}


double time_dense1d_access_via_base(int nrepeats, const index_t nelems, const double *src, double *buf)
{
	caview1d<double> view(src, nelems);
	caview1d_base<caview1d<double> >& viewb = view;

	index_t n = (index_t)nelems;

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			buf[j] = viewb(j);
		}
	}

	return tm.elapsed( SECONDS );
}



double time_step1d_export(int nrepeats, const index_t nelems, const double *src, double *buf)
{
	caview1d_ex<double, step_ind> view(src, step_ind(nelems, 2));

	view.export_to(buf);

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		view.export_to(buf);
	}

	return tm.elapsed( SECONDS );
}


double time_step1d_access(int nrepeats, const index_t nelems, const double *src, double *buf)
{
	caview1d_ex<double, step_ind> view(src, step_ind(nelems, 2));

	index_t n = (index_t)nelems;

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		for (index_t j = 0; j < n; ++j)
		{
			buf[j] = view(j);
		}
	}

	return tm.elapsed( SECONDS );
}


double time_dense2d_rm_export(int nrepeats, const index_t m, const index_t n, const double *src, double *buf)
{
	caview2d<double, row_major_t> view = make_caview2d_rm(src, m, n);

	view.export_to(buf);

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		view.export_to(buf);
	}

	return tm.elapsed( SECONDS );
}


double time_dense2d_cm_export(int nrepeats, const index_t m, const index_t n, const double *src, double *buf)
{
	caview2d<double, column_major_t> view = make_caview2d_cm(src, m, n);

	view.export_to(buf);

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		view.export_to(buf);
	}
	return tm.elapsed( SECONDS );

}


double time_dense2d_rm_access(int nrepeats, const index_t m, const index_t n, const double *src, double *buf)
{
	caview2d<double, row_major_t> view = make_caview2d_rm(src, m, n);

	index_t d0 = m;
	index_t d1 = n;

	timer tm(true);
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

	return tm.elapsed( SECONDS );
}


double time_dense2d_cm_access(int nrepeats, const index_t m, const index_t n, const double *src, double *buf)
{
	caview2d<double, column_major_t> view = make_caview2d_cm(src, m, n);

	index_t d0 = m;
	index_t d1 = n;

	timer tm(true);
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

	return tm.elapsed( SECONDS );
}


double time_step2d_rm_export(int nrepeats, const index_t m, const index_t n, const double *src, double *buf)
{
	caview2d_ex<double, row_major_t, step_ind, step_ind> view(
			src, 2*m, 2*n, step_ind(m, 2), step_ind(n,2));

	view.export_to(buf);

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		view.export_to(buf);
	}

	return tm.elapsed( SECONDS );
}


double time_step2d_cm_export(int nrepeats, const index_t m, const index_t n, const double *src, double *buf)
{
	caview2d_ex<double, column_major_t, step_ind, step_ind> view(
			src, 2*m, 2*n, step_ind(m, 2), step_ind(n,2));

	view.export_to(buf);

	timer tm(true);
	for (int i = 0; i < nrepeats; ++i)
	{
		view.export_to(buf);
	}

	return tm.elapsed( SECONDS );

}


double time_step2d_rm_access(int nrepeats, const index_t m, const index_t n, const double *src, double *buf)
{
	caview2d_ex<double, row_major_t, step_ind, step_ind> view(
			src, 2*m, 2*n, step_ind(m, 2), step_ind(n,2));

	index_t d0 = m;
	index_t d1 = n;

	timer tm(true);
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

	return tm.elapsed( SECONDS );
}


double time_step2d_cm_access(int nrepeats, const index_t m, const index_t n, const double *src, double *buf)
{
	caview2d_ex<double, column_major_t, step_ind, step_ind> view(
			src, 2*m, 2*n, step_ind(m, 2), step_ind(n,2));

	index_t d0 = m;
	index_t d1 = n;

	timer tm(true);
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

	return tm.elapsed( SECONDS );

}


int main(int argc, char *argv[])
{
	const index_t nrows = 1000;
	const index_t ncols = 1000;
	const index_t nelems = nrows * ncols;
	const index_t nelems_s = 5 * nelems;

	const int nrepeats = 100;

	// generate random source

	std::printf("preparing data source ...\n");

	double *src = new double[nelems_s];

	std::srand( (unsigned int)( std::time( 0 )) );
	for (index_t i = 0; i < nelems_s; ++i)
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

	std::printf("testing dense1d_access (via base) ...\n");
	double e_dense1d_access_b = time_dense1d_access_via_base(nrepeats, nelems, src, buf);

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
	std::printf("\tdense1d_access (via base): %.4f s\n", e_dense1d_access_b);
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


