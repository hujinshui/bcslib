/**
 * @file veccalc_functors.h
 *
 * Wrappers of vectorized calculation into functors
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_VECCALC_FUNCTORS_H
#define BCSLIB_VECCALC_FUNCTORS_H

#include <bcslib/veccomp/veccalc.h>

namespace bcs
{

	// addition

	template<typename T>
	struct vec_vec_add_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, const T *x1, const T *x2, T *y) const { vec_add(n, x1, x2, y); }
	};

	template<typename T>
	struct vec_sca_add_ftor
	{
		T s;
		vec_sca_add_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, const T *x1, T *y) const { vec_add(n, x1, s, y); }
	};

	template<typename T>
	struct vec_vec_add_ip_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, T *y, const T *x) const { vec_add_inplace(n, y, x); }
	};

	template<typename T>
	struct vec_sca_add_ip_ftor
	{
		T s;
		vec_sca_add_ip_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, T *y) const { vec_add_inplace(n, y, s); }
	};


	// subtraction

	template<typename T>
	struct vec_vec_sub_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, const T *x1, const T *x2, T *y) const { vec_sub(n, x1, x2, y); }
	};

	template<typename T>
	struct vec_sca_sub_ftor
	{
		T s;
		vec_sca_sub_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, const T *x1, T *y) const { vec_sub(n, x1, s, y); }
	};

	template<typename T>
	struct sca_vec_sub_ftor
	{
		T s;
		sca_vec_sub_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, const T *x2, T *y) const { vec_sub(n, s, x2, y); }
	};

	template<typename T>
	struct vec_vec_sub_ip_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, T *y, const T *x) const { vec_sub_inplace(n, y, x); }
	};

	template<typename T>
	struct vec_sca_sub_ip_ftor
	{
		T s;
		vec_sca_sub_ip_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, T *y) const { vec_sub_inplace(n, y, s); }
	};


	// multiplication

	template<typename T>
	struct vec_vec_mul_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, const T *x1, const T *x2, T *y) const { vec_mul(n, x1, x2, y); }
	};

	template<typename T>
	struct vec_sca_mul_ftor
	{
		T s;
		vec_sca_mul_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, const T *x1, T *y) const { vec_mul(n, x1, s, y); }
	};

	template<typename T>
	struct vec_vec_mul_ip_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, T *y, const T *x) const { vec_mul_inplace(n, y, x); }
	};

	template<typename T>
	struct vec_sca_mul_ip_ftor
	{
		T s;
		vec_sca_mul_ip_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, T *y) const { vec_mul_inplace(n, y, s); }
	};


	// division

	template<typename T>
	struct vec_vec_div_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, const T *x1, const T *x2, T *y) const { vec_div(n, x1, x2, y); }
	};

	template<typename T>
	struct vec_sca_div_ftor
	{
		T s;
		vec_sca_div_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, const T *x1, T *y) const { vec_div(n, x1, s, y); }
	};

	template<typename T>
	struct sca_vec_div_ftor
	{
		T s;
		sca_vec_div_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, const T *x2, T *y) const { vec_div(n, s, x2, y); }
	};

	template<typename T>
	struct vec_vec_div_ip_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, T *y, const T *x) const { vec_div_inplace(n, y, x); }
	};

	template<typename T>
	struct vec_sca_div_ip_ftor
	{
		T s;
		vec_sca_div_ip_ftor(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, T *y) const { vec_div_inplace(n, y, s); }
	};


	// negation

	template<typename T>
	struct vec_neg_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, const T *x, T *y) const { vec_negate(n, x, y); }


	};


	template<typename T>
	struct vec_neg_ip_ftor
	{
		typedef T result_value_type;
		void operator() (size_t n, T *y) const { vec_negate(n, y); }
	};


	// absolute value

	template<typename T>
	struct vec_abs_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const { return vec_abs(n, x, y); }
		void operator() (size_t n, T *y) const { return vec_abs(n, y, y); }
	};


	// power and root functions

	// sqr

	template<typename T>
	struct vec_sqr_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const { return vec_sqr(n, x, y); }
		void operator() (size_t n, T *y) const { return vec_sqr(n, y, y); }
	};

	// sqrt

	template<typename T>
	struct vec_sqrt_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const { return vec_sqrt(n, x, y); }
		void operator() (size_t n, T *y) const { return vec_sqrt(n, y, y); }
	};


	// rcp

	template<typename T>
	struct vec_rcp_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const { return vec_rcp(n, x, y); }
		void operator() (size_t n, T *y) const { return vec_rcp(n, y, y); }
	};


	// rsqrt

	template<typename T>
	struct vec_rsqrt_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const { return vec_rsqrt(n, x, y); }
		void operator() (size_t n, T *y) const { return vec_rsqrt(n, y, y); }
	};


	// pow (vec-vec)

	template<typename T>
	struct vec_pow_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, const T *e, T *y) const { return vec_pow(n, x, e, y); }
		void operator() (size_t n, T *y, const T* e) const { return vec_pow(n, y, e, y); }
	};


	// pow (vec-scalar)

	template<typename T>
	struct vec_sca_pow_ftor
	{
		typedef T result_value_type;

		T e;
		vec_sca_pow_ftor(const T& e_) : e(e_) { }

		void operator() (size_t n, const T *x, T *y) const { return vec_pow(n, x, e, y); }
		void operator() (size_t n, T *y) const { return vec_pow(n, y, e, y); }
	};



	// exponential and logarithm functions

	template<typename T>
	struct vec_exp_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const { return vec_exp(n, x, y); }
		void operator() (size_t n, T *y) const { return vec_exp(n, y, y); }
	};

	template<typename T>
	struct vec_log_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const { return vec_log(n, x, y); }
		void operator() (size_t n, T *y) const { return vec_log(n, y, y); }
	};

	template<typename T>
	struct vec_log10_ftor
	{
		typedef T result_value_type;

		void operator() (size_t n, const T *x, T *y) const { return vec_log10(n, x, y); }
		void operator() (size_t n, T *y) const { return vec_log10(n, y, y); }
	};



}

#endif 
