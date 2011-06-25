/*
 * @file test_calc_performance.cpp
 *
 * Test the performance (and correctness) of calculation implementation
 *
 * @author Dahua Lin
 */


#include <bcslib/base/basic_mem.h>
#include <bcslib/veccomp/veccalc.h>
#include <bcslib/test/performance_timer.h>

#include <functional>
#include <algorithm>

#include <cstdlib>
#include <cstdio>


using namespace bcs;
using namespace bcs::test;


// Auxiliary functions

inline double calc_mops(size_t n, size_t nrepeat, double elapsed_secs)
{
	return (double)(n * nrepeat) / (elapsed_secs * 1e6);
}


template<typename T>
inline double maxdev(size_t n, const T *a, const T *b)
{
	double s = 0;
	for (size_t i = 0; i < n; ++i)
	{
		double v = (double)(std::abs(a[i] - b[i]));
		if (v > s) s = v;
	}
	return s;
}


void report_perf(const char *name, size_t n, size_t nrepeat, double etime)
{
	std::printf("\tspeed of %s = %.4f MOPS\n", name, calc_mops(n, nrepeat, etime));
}

void raise_largedev_warning(double dev)
{
	std::printf("\tWARNING: Large deviation: %.4g\n", dev);
}


template<typename T>
void random_fill(T *x, size_t n, double lb, double ub)
{
	std::srand( (unsigned int)(std::time(0)) );

	double u = (double)(std::rand()) / RAND_MAX;
	double s = ub - lb;
	for (size_t i = 0; i < n; ++i) x[i] = (T)(lb + u * s);
}



// Testing functions

template<typename T, class SFunc, class VFunc>
void timed_test(const char* title,
		SFunc sfunc, VFunc vfunc, const T *x1, const T *x2, T *y0, T *y, size_t n, size_t nrepeat, double dthres)
{
	std::printf("Testing %s\n", title);

	// warming and verification

	set_zeros_to_elements(y0, n);
	set_zeros_to_elements(y, n);

	for (size_t i = 0; i < n; ++i) y0[i] = sfunc(x1[i], x2[i]);
	vfunc(n, x1, x2, y);

	double dev = 0;
	if ((dev = maxdev(n, y, y0)) <= dthres)
	{
		performance_timer tm0(true);
		for (size_t k = 0; k < nrepeat; ++k)
		{
			for (size_t i = 0; i < n; ++i) y0[i] = sfunc(x1[i], x2[i]);
		}
		double et0 = tm0.elapsed_seconds();

		performance_timer tm1(true);
		for (size_t k = 0; k < nrepeat; ++k)
		{
			vfunc(n, x1, x2, y);
		}
		double et1 = tm1.elapsed_seconds();

		report_perf("for-loop", n, nrepeat, et0);
		report_perf("vec-func", n, nrepeat, et1);
	}
	else
	{
		raise_largedev_warning(dev);
	}
}


int main(int argc, char *argv[])
{
	// prepare storage and data

	std::printf("Preparing data ...");

	const size_t n = 10000000;

	scoped_buffer<double> x1d_buf(n);
	scoped_buffer<double> x2d_buf(n);
	scoped_buffer<double> y0d_buf(n);
	scoped_buffer<double> y1d_buf(n);

	scoped_buffer<float> x1f_buf(n);
	scoped_buffer<float> x2f_buf(n);
	scoped_buffer<float> y0f_buf(n);
	scoped_buffer<float> y1f_buf(n);

	random_fill(x1d_buf.pbase(), n, -1.0, 1.0);
	random_fill(x2d_buf.pbase(), n, 0.1, 1.0);

	random_fill(x1f_buf.pbase(), n, -1.0f, 1.0f);
	random_fill(x2f_buf.pbase(), n, 0.1f, 1.0f);

	const double *x1d = x1d_buf.pbase();
	const double *x2d = x2d_buf.pbase();
	const float  *x1f = x1f_buf.pbase();
	const float  *x2f = x2f_buf.pbase();

	double *y0d = y0d_buf.pbase();
	double *y1d = y1d_buf.pbase();
	float  *y0f = y0f_buf.pbase();
	float  *y1f = y1f_buf.pbase();

	std::printf("\n");

	// testing

	std::printf("Start testing ...\n");

	timed_test("add-vec-vec (64f)", std::plus<double>(), vec_vec_add_ftor<double>(), x1d, x2d, y0d, y1d, n, 10, 0);
	timed_test("add-vec-vec (32f)", std::plus<float>(),  vec_vec_add_ftor<float>(),  x1f, x2f, y0f, y1f, n, 10, 0);

	std::printf("\n");

	return 0;
}



