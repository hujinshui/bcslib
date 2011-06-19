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

	template<typename T>
	struct vec_vec_add
	{
		typedef T result_value_type;
		void operator() (size_t n, const T *x1, const T *x2, T *y) const { vec_add(n, x1, x2, y); }
	};

	template<typename T>
	struct vec_sca_add
	{
		T s;
		vec_sca_add(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, const T *x1, T *y) const { vec_add(n, x1, s, y); }
	};

	template<typename T>
	struct vec_vec_add_ip
	{
		typedef T result_value_type;
		void operator() (size_t n, T *y, const T *x) const { vec_add_inplace(n, y, x); }
	};

	template<typename T>
	struct vec_sca_add_ip
	{
		T s;
		vec_sca_add_ip(const T& s_) : s(s_) { }

		typedef T result_value_type;
		void operator() (size_t n, T *y, const T &x) const { vec_add_inplace(n, y, x); }
	};

}

#endif 
