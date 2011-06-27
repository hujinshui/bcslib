/*
 * @file test_calc_performance.cpp
 *
 * Test the performance (and correctness) of calculation implementation
 *
 * @author Dahua Lin
 */


#include <bcslib/base/mathfun.h>
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


inline void report_perf(const char *name, size_t n, size_t nrepeat, double etime)
{
	std::printf("\tspeed of %s = %.4f MOPS\n", name, calc_mops(n, nrepeat, etime));
}

inline void raise_largedev_warning(double dev)
{
	std::printf("\tWARNING: Large deviation: %.4g\n", dev);
}

template<typename T>
inline void print_samples(const char *title, const T *x, size_t n)
{
	std::printf("%s: ", title);
	for (size_t i = 0; i < n; ++i)
	{
		std::printf("%.3f ", (double)x[i]);
	}
	std::printf("...\n");
}


template<typename T>
void random_fill(T *x, size_t n, double lb, double ub)
{
	double s = ub - lb;
	for (size_t i = 0; i < n; ++i)
	{
		double u = (double)(std::rand()) / RAND_MAX;
		x[i] = (T)(lb + u * s);
	}
}


// Tasks


template<typename T, class SFunc, class VFunc>
struct uniop_task
{
	SFunc sfunc;
	VFunc vfunc;
	size_t len;

	const T *src1;
	T *dst0;
	T *dst1;

	double dthres;

	uniop_task(SFunc sf, VFunc vf, size_t n, const T *x1, T *y0, T* y1, double dthr)
	: sfunc(sf), vfunc(vf)
	, len(n), src1(x1), dst0(y0), dst1(y1), dthres(dthr) { }

	void reset() { }

	void do_for_loop()
	{
		for (size_t i = 0; i < len; ++i)
		{
			dst0[i] = sfunc(src1[i]);
		}
	}

	void do_vec_func()
	{
		vfunc(len, src1, dst1);
	}
};


template<typename T, class SFunc, class VFunc>
uniop_task<T, SFunc, VFunc>
inline mk_task(SFunc sf, VFunc vf, size_t n, const T *x1, T *y0, T* y1, double dthr)
{
	return uniop_task<T, SFunc, VFunc>(sf, vf, n, x1, y0, y1, dthr);
}


template<typename T, class SFunc, class VFunc>
struct binop_task
{
	SFunc sfunc;
	VFunc vfunc;
	size_t len;

	const T *src1;
	const T *src2;
	T *dst0;
	T *dst1;

	double dthres;

	void reset() { }

	binop_task(SFunc sf, VFunc vf, size_t n, const T *x1, const T *x2, T *y0, T* y1, double dthr)
	: sfunc(sf), vfunc(vf)
	, len(n), src1(x1), src2(x2), dst0(y0), dst1(y1), dthres(dthr) { }

	void do_for_loop()
	{
		for (size_t i = 0; i < len; ++i)
		{
			dst0[i] = sfunc(src1[i], src2[i]);
		}
	}

	void do_vec_func()
	{
		vfunc(len, src1, src2, dst1);
	}

};

template<typename T, class SFunc, class VFunc>
binop_task<T, SFunc, VFunc>
inline mk_task(SFunc sf, VFunc vf, size_t n, const T *x1, const T *x2, T *y0, T* y1, double dthr)
{
	return binop_task<T, SFunc, VFunc>(sf, vf, n, x1, x2, y0, y1, dthr);
}


template<typename T, class SFunc, class VFunc>
struct uniop_ip_task
{
	SFunc sfunc;
	VFunc vfunc;
	size_t len;

	const T *src1;
	T *dst0;
	T *dst1;

	double dthres;

	uniop_ip_task(SFunc sf, VFunc vf, size_t n, const T *x1, T *y0, T* y1, double dthr)
	: sfunc(sf), vfunc(vf)
	, len(n), src1(x1), dst0(y0), dst1(y1), dthres(dthr) { }

	void reset()
	{
		copy_elements(src1, dst0, len);
		copy_elements(src1, dst1, len);
	}

	void do_for_loop()
	{
		for (size_t i = 0; i < len; ++i)
		{
			dst0[i] = sfunc(src1[i]);
		}
	}

	void do_vec_func()
	{
		vfunc(len, dst1);
	}
};


template<typename T, class SFunc, class VFunc>
uniop_ip_task<T, SFunc, VFunc>
inline mk_ip_task(SFunc sf, VFunc vf, size_t n, const T *x1, T *y0, T* y1, double dthr)
{
	return uniop_ip_task<T, SFunc, VFunc>(sf, vf, n, x1, y0, y1, dthr);
}


template<typename T, class SFunc, class VFunc>
struct binop_ip_task
{
	SFunc sfunc;
	VFunc vfunc;
	size_t len;

	const T *src1;
	const T *src2;
	T *dst0;
	T *dst1;

	double dthres;

	binop_ip_task(SFunc sf, VFunc vf, size_t n, const T *x1, const T *x2, T *y0, T* y1, double dthr)
	: sfunc(sf), vfunc(vf)
	, len(n), src1(x1), src2(x2), dst0(y0), dst1(y1), dthres(dthr) { }

	void reset()
	{
		copy_elements(src1, dst0, len);
		copy_elements(src1, dst1, len);
	}

	void do_for_loop()
	{
		for (size_t i = 0; i < len; ++i)
		{
			dst0[i] = sfunc(src1[i], src2[i]);
		}
	}

	void do_vec_func()
	{
		vfunc(len, dst1, src2);
	}
};


template<typename T, class SFunc, class VFunc>
binop_ip_task<T, SFunc, VFunc>
inline mk_ip_task(SFunc sf, VFunc vf, size_t n, const T *x1, const T *x2, T *y0, T* y1, double dthr)
{
	return binop_ip_task<T, SFunc, VFunc>(sf, vf, n, x1, x2, y0, y1, dthr);
}




// Testing functions

template<class Task>
void timed_test(const char* title, Task task, size_t nrepeat)
{
	std::printf("Testing %s\n", title);

	// warming and verification

	set_zeros_to_elements(task.dst0, task.len);
	set_zeros_to_elements(task.dst1, task.len);

	task.reset();
	task.do_for_loop();
	task.do_vec_func();

	double dev;
	if ((dev = maxdev(task.len, task.dst0, task.dst1)) <= task.dthres)
	{
		task.reset();

		performance_timer tm0(true);
		for (size_t k = 0; k < nrepeat; ++k)
		{
			task.do_for_loop();
		}
		double et0 = tm0.elapsed_seconds();

		performance_timer tm1(true);
		for (size_t k = 0; k < nrepeat; ++k)
		{
			task.do_vec_func();
		}
		double et1 = tm1.elapsed_seconds();

		report_perf("nrm-loop", task.len, nrepeat, et0);
		report_perf("vec-func", task.len, nrepeat, et1);
		std::printf("\tgain = %.3f\n", et0 / et1);
	}
	else
	{
		raise_largedev_warning(dev);
	}
}


// main test scripts


void test_comparison(size_t n, size_t nr,
		const double *x1d, const double *x2d, double *y0d, double *y1d,
		const float  *x1f, const float  *x2f, float  *y0f, float  *y1f)
{
	std::printf("Testing Comparison and Bounding:\n");
	std::printf("------------------------------------\n");

	// max_each

	timed_test("max-each (64f)", mk_task(max_fun<double>(), vec_vec_max_each_ftor<double>(), n, x1d, x2d, y0d, y1d, 0), nr);
	timed_test("max-each (32f)", mk_task(max_fun<float>(),  vec_vec_max_each_ftor<float>(),  n, x1f, x2f, y0f, y1f, 0), nr);

	// min_each

	timed_test("max-each (64f)", mk_task(min_fun<double>(), vec_vec_min_each_ftor<double>(), n, x1d, x2d, y0d, y1d, 0), nr);
	timed_test("max-each (32f)", mk_task(min_fun<float>(),  vec_vec_min_each_ftor<float>(),  n, x1f, x2f, y0f, y1f, 0), nr);

	// lbound

	timed_test("lbound (64f)", mk_task(lbound_fun<double>(0.2), vec_lbound_ftor<double>(0.2), n, x1d, y0d, y1d, 0), nr);
	timed_test("lbound (32f)", mk_task(lbound_fun<float>(0.2f), vec_lbound_ftor<float>(0.2f), n, x1f, y0f, y1f, 0), nr);

	timed_test("lbound-ip (64f)", mk_ip_task(lbound_fun<double>(0.2), vec_lbound_ftor<double>(0.2), n, x1d, y0d, y1d, 0), nr);
	timed_test("lbound-ip (32f)", mk_ip_task(lbound_fun<float>(0.2f), vec_lbound_ftor<float>(0.2f), n, x1f, y0f, y1f, 0), nr);

	// ubound

	timed_test("ubound (64f)", mk_task(ubound_fun<double>(-0.2), vec_ubound_ftor<double>(-0.2), n, x1d, y0d, y1d, 0), nr);
	timed_test("ubound (32f)", mk_task(ubound_fun<float>(-0.2f), vec_ubound_ftor<float>(-0.2f), n, x1f, y0f, y1f, 0), nr);

	timed_test("ubound-ip (64f)", mk_ip_task(ubound_fun<double>(-0.2), vec_ubound_ftor<double>(-0.2), n, x1d, y0d, y1d, 0), nr);
	timed_test("ubound-ip (32f)", mk_ip_task(ubound_fun<float>(-0.2f), vec_ubound_ftor<float>(-0.2f), n, x1f, y0f, y1f, 0), nr);

	// rgn_bound

	timed_test("rgn_bound (64f)", mk_task(rgn_bound_fun<double>(-0.5, 0.5),  vec_rgn_bound_ftor<double>(-0.5, 0.5),  n, x1d, y0d, y1d, 0), nr);
	timed_test("rgn_bound (32f)", mk_task(rgn_bound_fun<float>(-0.5f, 0.5f), vec_rgn_bound_ftor<float>(-0.5f, 0.5f), n, x1f, y0f, y1f, 0), nr);

	timed_test("rgn_bound-ip (64f)", mk_ip_task(rgn_bound_fun<double>(-0.5, 0.5),  vec_rgn_bound_ftor<double>(-0.5, 0.5),  n, x1d, y0d, y1d, 0), nr);
	timed_test("rgn_bound-ip (32f)", mk_ip_task(rgn_bound_fun<float>(-0.5f, 0.5f), vec_rgn_bound_ftor<float>(-0.5f, 0.5f), n, x1f, y0f, y1f, 0), nr);

	// abound

	timed_test("abound (64f)", mk_task(rgn_bound_fun<double>(-0.6, 0.6),  vec_abound_ftor<double>(0.6), n, x1d, y0d, y1d, 0), nr);
	timed_test("abound (32f)", mk_task(rgn_bound_fun<float>(-0.6f, 0.6f), vec_abound_ftor<float>(0.6f), n, x1f, y0f, y1f, 0), nr);

	timed_test("abound-ip (64f)", mk_ip_task(rgn_bound_fun<double>(-0.6, 0.6),  vec_abound_ftor<double>(0.6), n, x1d, y0d, y1d, 0), nr);
	timed_test("abound-ip (32f)", mk_ip_task(rgn_bound_fun<float>(-0.6f, 0.6f), vec_abound_ftor<float>(0.6f), n, x1f, y0f, y1f, 0), nr);

	std::printf("\n");
}


void test_arithmetic(size_t n, size_t nr,
		const double *x1d, const double *x2d, double *y0d, double *y1d,
		const float  *x1f, const float  *x2f, float  *y0f, float  *y1f)
{
	using namespace std;

	double dv0 = 3.14;
	float  fv0 = 3.14f;

	std::printf("Testing Arithmetic:\n");
	std::printf("----------------------\n");

	// add

	timed_test("add-vec-vec (64f)", mk_task(plus<double>(), vec_vec_add_ftor<double>(), n, x1d, x2d, y0d, y1d, 0), nr);
	timed_test("add-vec-vec (32f)", mk_task(plus<float>(),  vec_vec_add_ftor<float>(),  n, x1f, x2f, y0f, y1f, 0), nr);

	timed_test("add-vec-sca (64f)", mk_task(bind2nd(plus<double>(), dv0), vec_sca_add_ftor<double>(dv0), n, x1d, y0d, y1d, 0), nr);
	timed_test("add-vec-sca (32f)", mk_task(bind2nd(plus<float>(),  fv0), vec_sca_add_ftor<float>(fv0),  n, x1f, y0f, y1f, 0), nr);

	timed_test("add-vec-vec-ip (64f)", mk_ip_task(plus<double>(), vec_vec_add_ftor<double>(), n, x1d, x2d, y0d, y1d, 0), nr);
	timed_test("add-vec-vec-ip (32f)", mk_ip_task(plus<float>(),  vec_vec_add_ftor<float>(),  n, x1f, x2f, y0f, y1f, 0), nr);

	timed_test("add-vec-sca-ip (64f)", mk_ip_task(bind2nd(plus<double>(), dv0), vec_sca_add_ftor<double>(dv0), n, x1d, y0d, y1d, 0), nr);
	timed_test("add-vec-sca-ip (32f)", mk_ip_task(bind2nd(plus<float>(),  fv0), vec_sca_add_ftor<float>(fv0),  n, x1f, y0f, y1f, 0), nr);

	// sub

	timed_test("sub-vec-vec (64f)", mk_task(minus<double>(), vec_vec_sub_ftor<double>(), n, x1d, x2d, y0d, y1d, 0), nr);
	timed_test("sub-vec-vec (32f)", mk_task(minus<float>(),  vec_vec_sub_ftor<float>(),  n, x1f, x2f, y0f, y1f, 0), nr);

	timed_test("sub-vec-sca (64f)", mk_task(bind2nd(minus<double>(), dv0), vec_sca_sub_ftor<double>(dv0), n, x1d, y0d, y1d, 0), nr);
	timed_test("sub-vec-sca (32f)", mk_task(bind2nd(minus<float>(),  fv0), vec_sca_sub_ftor<float>(fv0),  n, x1f, y0f, y1f, 0), nr);

	timed_test("sub-sca-vec (64f)", mk_task(bind1st(minus<double>(), dv0), sca_vec_sub_ftor<double>(dv0), n, x1d, y0d, y1d, 0), nr);
	timed_test("sub-sca-vec (32f)", mk_task(bind1st(minus<float>(),  fv0), sca_vec_sub_ftor<float>(fv0),  n, x1f, y0f, y1f, 0), nr);

	timed_test("sub-vec-vec-ip (64f)", mk_ip_task(minus<double>(), vec_vec_sub_ftor<double>(), n, x1d, x2d, y0d, y1d, 0), nr);
	timed_test("sub-vec-vec-ip (32f)", mk_ip_task(minus<float>(),  vec_vec_sub_ftor<float>(),  n, x1f, x2f, y0f, y1f, 0), nr);

	timed_test("sub-vec-sca-ip (64f)", mk_ip_task(bind2nd(minus<double>(), dv0), vec_sca_sub_ftor<double>(dv0), n, x1d, y0d, y1d, 0), nr);
	timed_test("sub-vec-sca-ip (32f)", mk_ip_task(bind2nd(minus<float>(),  fv0), vec_sca_sub_ftor<float>(fv0),  n, x1f, y0f, y1f, 0), nr);

	timed_test("sub-sca-vec-ip (64f)", mk_ip_task(bind1st(minus<double>(), dv0), sca_vec_sub_ftor<double>(dv0), n, x1d, y0d, y1d, 0), nr);
	timed_test("sub-sca-vec-ip (32f)", mk_ip_task(bind1st(minus<float>(),  fv0), sca_vec_sub_ftor<float>(fv0),  n, x1f, y0f, y1f, 0), nr);

	// mul

	timed_test("mul-vec-vec (64f)", mk_task(multiplies<double>(), vec_vec_mul_ftor<double>(), n, x1d, x2d, y0d, y1d, 0), nr);
	timed_test("mul-vec-vec (32f)", mk_task(multiplies<float>(),  vec_vec_mul_ftor<float>(),  n, x1f, x2f, y0f, y1f, 0), nr);

	timed_test("mul-vec-sca (64f)", mk_task(bind2nd(multiplies<double>(), dv0), vec_sca_mul_ftor<double>(dv0), n, x1d, y0d, y1d, 0), nr);
	timed_test("mul-vec-sca (32f)", mk_task(bind2nd(multiplies<float>(),  fv0), vec_sca_mul_ftor<float>(fv0),  n, x1f, y0f, y1f, 0), nr);

	timed_test("mul-vec-vec-ip (64f)", mk_ip_task(multiplies<double>(), vec_vec_mul_ftor<double>(), n, x1d, x2d, y0d, y1d, 0), nr);
	timed_test("mul-vec-vec-ip (32f)", mk_ip_task(multiplies<float>(),  vec_vec_mul_ftor<float>(),  n, x1f, x2f, y0f, y1f, 0), nr);

	timed_test("mul-vec-sca-ip (64f)", mk_ip_task(bind2nd(multiplies<double>(), dv0), vec_sca_mul_ftor<double>(dv0), n, x1d, y0d, y1d, 0), nr);
	timed_test("mul-vec-sca-ip (32f)", mk_ip_task(bind2nd(multiplies<float>(),  fv0), vec_sca_mul_ftor<float>(fv0),  n, x1f, y0f, y1f, 0), nr);

	// div

	double div_td = 1e-14;
	float div_tf = 1e-6f;

	timed_test("div-vec-vec (64f)", mk_task(divides<double>(), vec_vec_div_ftor<double>(), n, x1d, x2d, y0d, y1d, div_td), nr);
	timed_test("div-vec-vec (32f)", mk_task(divides<float>(),  vec_vec_div_ftor<float>(),  n, x1f, x2f, y0f, y1f, div_tf), nr);

	timed_test("div-vec-sca (64f)", mk_task(bind2nd(divides<double>(), dv0), vec_sca_div_ftor<double>(dv0), n, x1d, y0d, y1d, div_td), nr);
	timed_test("div-vec-sca (32f)", mk_task(bind2nd(divides<float>(),  fv0), vec_sca_div_ftor<float>(fv0),  n, x1f, y0f, y1f, div_tf), nr);

	timed_test("div-sca-vec (64f)", mk_task(bind1st(divides<double>(), dv0), sca_vec_div_ftor<double>(dv0), n, x1d, y0d, y1d, div_td), nr);
	timed_test("div-sca-vec (32f)", mk_task(bind1st(divides<float>(),  fv0), sca_vec_div_ftor<float>(fv0),  n, x1f, y0f, y1f, div_tf), nr);

	timed_test("div-vec-vec-ip (64f)", mk_ip_task(divides<double>(), vec_vec_div_ftor<double>(), n, x1d, x2d, y0d, y1d, div_td), nr);
	timed_test("div-vec-vec-ip (32f)", mk_ip_task(divides<float>(),  vec_vec_div_ftor<float>(),  n, x1f, x2f, y0f, y1f, div_tf), nr);

	timed_test("div-vec-sca-ip (64f)", mk_ip_task(bind2nd(divides<double>(), dv0), vec_sca_div_ftor<double>(dv0), n, x1d, y0d, y1d, div_td), nr);
	timed_test("div-vec-sca-ip (32f)", mk_ip_task(bind2nd(divides<float>(),  fv0), vec_sca_div_ftor<float>(fv0),  n, x1f, y0f, y1f, div_tf), nr);

	timed_test("div-sca-vec-ip (64f)", mk_ip_task(bind1st(divides<double>(), dv0), sca_vec_div_ftor<double>(dv0), n, x1d, y0d, y1d, div_td), nr);
	timed_test("div-sca-vec-ip (32f)", mk_ip_task(bind1st(divides<float>(),  fv0), sca_vec_div_ftor<float>(fv0),  n, x1f, y0f, y1f, div_tf), nr);

	// negate

	timed_test("negate-vec (64f)", mk_task(negate<double>(), vec_neg_ftor<double>(), n, x1d, y0d, y1d, 0), nr);
	timed_test("negate-vec (32f)", mk_task(negate<float>(),  vec_neg_ftor<float>(),  n, x1f, y0f, y1f, 0), nr);

	timed_test("negate-vec-ip (64f)", mk_ip_task(negate<double>(), vec_neg_ftor<double>(), n, x1d, y0d, y1d, 0), nr);
	timed_test("negate-vec-ip (32f)", mk_ip_task(negate<float>(),  vec_neg_ftor<float>(),  n, x1f, y0f, y1f, 0), nr);

	// abs

	timed_test("abs-vec (64f)", mk_task(abs_fun<double>(), vec_abs_ftor<double>(), n, x1d, y0d, y1d, 0), nr);
	timed_test("abs-vec (32f)", mk_task(abs_fun<float>(),  vec_abs_ftor<float>(),  n, x1f, y0f, y1f, 0), nr);

	timed_test("abs-vec-ip (64f)", mk_ip_task(abs_fun<double>(), vec_abs_ftor<double>(), n, x1d, y0d, y1d, 0), nr);
	timed_test("abs-vec-ip (32f)", mk_ip_task(abs_fun<float>(),  vec_abs_ftor<float>(),  n, x1f, y0f, y1f, 0), nr);


	std::printf("\n");
}


void test_power_funs(size_t n, size_t nr,
		const double *x1d, const double *x2d, double *y0d, double *y1d,
		const float  *x1f, const float  *x2f, float  *y0f, float  *y1f)
{

	using std::bind2nd;

	std::printf("Testing Power Functions:\n");
	std::printf("---------------------------\n");

	double td = 1e-12;
	float tf = 1e-6f;

	// sqr

	timed_test("sqr-vec (64f)", mk_task(sqr_fun<double>(), vec_sqr_ftor<double>(), n, x1d, y0d, y1d, 0), nr);
	timed_test("sqr-vec (32f)", mk_task(sqr_fun<float>(),  vec_sqr_ftor<float>(),  n, x1f, y0f, y1f, 0), nr);

	// sqrt

	timed_test("sqrt-vec (64f)", mk_task(sqrt_fun<double>(), vec_sqrt_ftor<double>(), n, x2d, y0d, y1d, td), nr);
	timed_test("sqrt-vec (32f)", mk_task(sqrt_fun<float>(),  vec_sqrt_ftor<float>(),  n, x2f, y0f, y1f, tf), nr);

	// rcp

	timed_test("rcp-vec (64f)", mk_task(rcp_fun<double>(), vec_rcp_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("rcp-vec (32f)", mk_task(rcp_fun<float>(),  vec_rcp_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// rsqrt

	timed_test("rsqrt-vec (64f)", mk_task(rsqrt_fun<double>(), vec_rsqrt_ftor<double>(), n, x2d, y0d, y1d, td), nr);
	timed_test("rsqrt-vec (32f)", mk_task(rsqrt_fun<float>(),  vec_rsqrt_ftor<float>(),  n, x2f, y0f, y1f, tf), nr);

	// pow

	double ed = 2.4;
	float ef = 2.4f;

	timed_test("pow-vec-vec (64f)", mk_task(pow_fun<double>(), vec_pow_ftor<double>(), n, x2d, x1d, y0d, y1d, td), nr);
	timed_test("pow-vec-vec (32f)", mk_task(pow_fun<float>(),  vec_pow_ftor<float>(),  n, x2f, x1f, y0f, y1f, tf), nr);

	timed_test("pow-vec-sca (64f)", mk_task(bind2nd(pow_fun<double>(), ed), vec_sca_pow_ftor<double>(ed), n, x2d, y0d, y1d, td), nr);
	timed_test("pow-vec-sca (32f)", mk_task(bind2nd(pow_fun<float>(),  ef), vec_sca_pow_ftor<float>(ef),  n, x2f, y0f, y1f, tf), nr);

	// hypot

	timed_test("hypot-vec (64f)", mk_task(hypot_fun<double>(), vec_hypot_ftor<double>(), n, x2d, x1d, y0d, y1d, td), nr);
	timed_test("hypot-vec (32f)", mk_task(hypot_fun<float>(),  vec_hypot_ftor<float>(),  n, x2f, x1f, y0f, y1f, tf), nr);

	std::printf("\n");
}


void test_exp_and_log_funs(size_t n, size_t nr,
		const double *x1d, const double *x2d, double *y0d, double *y1d,
		const float  *x1f, const float  *x2f, float  *y0f, float  *y1f)
{
	std::printf("Testing Exp and Log Functions:\n");
	std::printf("----------------------------------\n");

	double td = 1e-14;
	float tf = 3e-7f;

	// exp

	timed_test("exp-vec (64f)", mk_task(exp_fun<double>(), vec_exp_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("exp-vec (32f)", mk_task(exp_fun<float>(),  vec_exp_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// log

	timed_test("log-vec (64f)", mk_task(log_fun<double>(), vec_log_ftor<double>(), n, x2d, y0d, y1d, td), nr);
	timed_test("log-vec (32f)", mk_task(log_fun<float>(),  vec_log_ftor<float>(),  n, x2f, y0f, y1f, tf), nr);

	// log10

	timed_test("log10-vec (64f)", mk_task(log10_fun<double>(), vec_log10_ftor<double>(), n, x2d, y0d, y1d, td), nr);
	timed_test("log10-vec (32f)", mk_task(log10_fun<float>(),  vec_log10_ftor<float>(),  n, x2f, y0f, y1f, tf), nr);

	std::printf("\n");
}


void test_rounding_funs(size_t n, size_t nr,
		const double *x1d, const double *x2d, double *y0d, double *y1d,
		const float  *x1f, const float  *x2f, float  *y0f, float  *y1f)
{
	std::printf("Testing Rounding Functions:\n");
	std::printf("----------------------------------\n");

	double td = 1e-14;
	float tf = 3e-7f;

	// floor

	timed_test("floor-vec (64f)", mk_task(floor_fun<double>(), vec_floor_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("floor-vec (32f)", mk_task(floor_fun<float>(),  vec_floor_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// ceil

	timed_test("ceil-vec (64f)", mk_task(ceil_fun<double>(), vec_ceil_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("ceil-vec (32f)", mk_task(ceil_fun<float>(),  vec_ceil_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// round

	timed_test("round-vec (64f)", mk_task(round_fun<double>(), vec_round_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("round-vec (32f)", mk_task(round_fun<float>(),  vec_round_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	std::printf("\n");
}


void test_trigonometric_funs(size_t n, size_t nr,
		const double *x1d, const double *x2d, double *y0d, double *y1d,
		const float  *x1f, const float  *x2f, float  *y0f, float  *y1f)
{

	std::printf("Testing Trigonometric Functions:\n");
	std::printf("--------------------------------------\n");

	double td = 1e-14;
	float tf = 3e-7f;

	// sin

	timed_test("sin-vec (64f)", mk_task(sin_fun<double>(), vec_sin_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("sin-vec (32f)", mk_task(sin_fun<float>(),  vec_sin_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// cos

	timed_test("cos-vec (64f)", mk_task(cos_fun<double>(), vec_cos_ftor<double>(), n, x2d, y0d, y1d, td), nr);
	timed_test("cos-vec (32f)", mk_task(cos_fun<float>(),  vec_cos_ftor<float>(),  n, x2f, y0f, y1f, tf), nr);

	// tan

	timed_test("tan-vec (64f)", mk_task(tan_fun<double>(), vec_tan_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("tan-vec (32f)", mk_task(tan_fun<float>(),  vec_tan_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// asin

	timed_test("asin-vec (64f)", mk_task(asin_fun<double>(), vec_asin_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("asin-vec (32f)", mk_task(asin_fun<float>(),  vec_asin_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// acos

	timed_test("acos-vec (64f)", mk_task(acos_fun<double>(), vec_acos_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("acos-vec (32f)", mk_task(acos_fun<float>(),  vec_acos_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// atan

	timed_test("atan-vec (64f)", mk_task(atan_fun<double>(), vec_atan_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("atan-vec (32f)", mk_task(atan_fun<float>(),  vec_atan_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// atan2

	timed_test("atan2-vec (64f)", mk_task(atan2_fun<double>(), vec_atan2_ftor<double>(), n, x2d, x1d, y0d, y1d, td), nr);
	timed_test("atan2-vec (32f)", mk_task(atan2_fun<float>(),  vec_atan2_ftor<float>(),  n, x2f, x1f, y0f, y1f, tf), nr);

	std::printf("\n");
}


void test_hyperbolic_funs(size_t n, size_t nr,
		const double *x1d, const double *x2d, double *y0d, double *y1d,
		const float  *x1f, const float  *x2f, float  *y0f, float  *y1f)
{

	std::printf("Testing Hyperbolic Functions:\n");
	std::printf("--------------------------------------\n");

	double td = 1e-14;
	float tf = 3e-7f;

	// sinh

	timed_test("sinh-vec (64f)", mk_task(sinh_fun<double>(), vec_sinh_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("sinh-vec (32f)", mk_task(sinh_fun<float>(),  vec_sinh_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	// cosh

	timed_test("cosh-vec (64f)", mk_task(cosh_fun<double>(), vec_cosh_ftor<double>(), n, x2d, y0d, y1d, td), nr);
	timed_test("cosh-vec (32f)", mk_task(cosh_fun<float>(),  vec_cosh_ftor<float>(),  n, x2f, y0f, y1f, tf), nr);

	// tanh

	timed_test("tanh-vec (64f)", mk_task(tanh_fun<double>(), vec_tanh_ftor<double>(), n, x1d, y0d, y1d, td), nr);
	timed_test("tanh-vec (32f)", mk_task(tanh_fun<float>(),  vec_tanh_ftor<float>(),  n, x1f, y0f, y1f, tf), nr);

	std::printf("\n");
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

	std::srand( 0 );

	random_fill(x1d_buf.pbase(), n, -1.0, 1.0);
	random_fill(x2d_buf.pbase(), n, 0.5, 1.0);

	random_fill(x1f_buf.pbase(), n, -1.0f, 1.0f);
	random_fill(x2f_buf.pbase(), n, 0.5f, 1.0f);

	const double *x1d = x1d_buf.pbase();
	const double *x2d = x2d_buf.pbase();
	const float  *x1f = x1f_buf.pbase();
	const float  *x2f = x2f_buf.pbase();

	double *y0d = y0d_buf.pbase();
	double *y1d = y1d_buf.pbase();
	float  *y0f = y0f_buf.pbase();
	float  *y1f = y1f_buf.pbase();

	std::printf("data samples:\n");
	size_t ns0 = 6;
	print_samples("x1d", x1d, ns0);
	print_samples("x2d", x2d, ns0);
	print_samples("x1f", x1f, ns0);
	print_samples("x2f", x2f, ns0);
	std::printf("\n");

	// testing

	std::printf("Start testing ...\n\n");

	test_comparison(n, 10, x1d, x2d, y0d, y1d, x1f, x2f, y0f, y1f);
	test_arithmetic(n, 10, x1d, x2d, y0d, y1d, x1f, x2f, y0f, y1f);
	test_rounding_funs(n, 10, x1d, x2d, y0d, y1d, x1f, x2f, y0f, y1f);
	test_power_funs(n, 3, x1d, x2d, y0d, y1d, x1f, x2f, y0f, y1f);
	test_exp_and_log_funs(n, 5, x1d, x2d, y0d, y1d, x1f, x2f, y0f, y1f);
	test_trigonometric_funs(n, 3, x1d, x2d, y0d, y1d, x1f, x2f, y0f, y1f);
	test_hyperbolic_funs(n, 3, x1d, x2d, y0d, y1d, x1f, x2f, y0f, y1f);

	std::printf("\n");

	return 0;
}



