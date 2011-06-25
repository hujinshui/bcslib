/**
 * @file veccalc.h
 *
 * Vectorized calculation
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_VECCALC_H
#define BCSLIB_VECCALC_H

#include <bcslib/base/basic_defs.h>

#include <cmath>
#include <algorithm>

namespace bcs
{
	/********************************************
	 *
	 *  Order comparison
	 *
	 *******************************************/

	// eq

	template<typename T>
	inline void vec_eq(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] == x2[i]);
	}

	template<typename T>
	inline void vec_eq(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] == x2);
	}

	// ne

	template<typename T>
	inline void vec_ne(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] != x2[i]);
	}

	template<typename T>
	inline void vec_ne(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] != x2);
	}

	// gt

	template<typename T>
	inline void vec_gt(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] > x2[i]);
	}

	template<typename T>
	inline void vec_gt(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] > x2);
	}

	// ge

	template<typename T>
	inline void vec_ge(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] >= x2[i]);
	}

	template<typename T>
	inline void vec_ge(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] >= x2);
	}

	// lt

	template<typename T>
	inline void vec_lt(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] < x2[i]);
	}

	template<typename T>
	inline void vec_lt(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] < x2);
	}

	// le

	template<typename T>
	inline void vec_le(size_t n, const T *x1, const T *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] <= x2[i]);
	}

	template<typename T>
	inline void vec_le(size_t n, const T *x1, const T& x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = (x1[i] <= x2);
	}

	// max_each and min_each

	template<typename T>
	inline void vec_max_each(size_t n, const T* x1, const T* x2, T *y)
	{
		using std::max;
		for (size_t i = 0; i < n; ++i) y[i] = max(x1[i], x2[i]);
	}

	template<typename T>
	inline void vec_min_each(size_t n, const T* x1, const T* x2, T *y)
	{
		using std::min;
		for (size_t i = 0; i < n; ++i) y[i] = min(x1[i], x2[i]);
	}


	/********************************************
	 *
	 *  Bounding (Thresholding)
	 *
	 *******************************************/

	template<typename T>
	inline void vec_lbound(size_t n, const T* x, const T& lb, T *y)
	{
		using std::max;
		for (size_t i = 0; i < n; ++i) y[i] = max(x[i], lb);
	}

	template<typename T>
	inline void vec_lbound_inplace(size_t n, T* y, const T& lb)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (y[i] < lb) y[i] = lb;
		}
	}

	template<typename T>
	inline void vec_ubound(size_t n, const T* x, const T& ub, T *y)
	{
		using std::min;
		for (size_t i = 0; i < n; ++i) y[i] = min(x[i], ub);
	}

	template<typename T>
	inline void vec_ubound_inplace(size_t n, T* y, const T& ub)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (y[i] > ub) y[i] = ub;
		}
	}

	template<typename T>
	inline void vec_rgn_bound(size_t n, const T* x, const T& lb, const T& ub, T *y)
	{
		for (size_t i = 0; i < n; ++i)
		{
			y[i] = rgn_bound(x[i], lb, ub);
		}
	}

	template<typename T>
	inline void vec_rgn_bound_inplace(size_t n, T* y, const T& lb, const T& ub)
	{
		for (size_t i = 0; i < n; ++i)
		{
			T v = y[i];
			if (v < lb) y[i] = lb;
			else if (v > ub) y[i] = ub;
		}
	}

	template<typename T>
	inline void vec_abound(size_t n, const T* x, const T& ab, T *y)
	{
		vec_rgn_bound(n, x, -ab, ab, y);
	}

	template<typename T>
	inline void vec_abound_inplace(size_t n, T* y, const T& ab)
	{
		vec_rgn_bound_inplace(n, y, -ab, ab);
	}



	/********************************************
	 *
	 *  Arithmetic Calculation
	 *
	 *******************************************/

	// add

	template<typename T>
	inline void vec_add(size_t n, const T* x1, const T *x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] + x2[i];
	}

	template<typename T>
	inline void vec_add(size_t n, const T* x1, const T& x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] + x2;
	}

	template<typename T>
	inline void vec_add_inplace(size_t n, T *y, const T *x)
	{
		for (size_t i = 0; i < n; ++i) y[i] += x[i];
	}

	template<typename T>
	inline void vec_add_inplace(size_t n, T *y, const T& x)
	{
		for (size_t i = 0; i < n; ++i) y[i] += x;
	}


	// sub

	template<typename T>
	inline void vec_sub(size_t n, const T* x1, const T *x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] - x2[i];
	}

	template<typename T>
	inline void vec_sub(size_t n, const T* x1, const T& x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] - x2;
	}

	template<typename T>
	inline void vec_sub(size_t n, const T& x1, const T* x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1 - x2[i];
	}

	template<typename T>
	inline void vec_sub_inplace(size_t n, T *y, const T *x)
	{
		for (size_t i = 0; i < n; ++i) y[i] -= x[i];
	}

	template<typename T>
	inline void vec_sub_inplace(size_t n, T *y, const T& x)
	{
		for (size_t i = 0; i < n; ++i) y[i] -= x;
	}

	template<typename T>
	inline void vec_sub_inplace(size_t n, const T& x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x - y[i];
	}


	// mul

	template<typename T>
	inline void vec_mul(size_t n, const T* x1, const T *x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] * x2[i];
	}

	template<typename T>
	inline void vec_mul(size_t n, const T* x1, const T& x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] * x2;
	}

	template<typename T>
	inline void vec_mul_inplace(size_t n, T *y, const T *x)
	{
		for (size_t i = 0; i < n; ++i) y[i] *= x[i];
	}

	template<typename T>
	inline void vec_mul_inplace(size_t n, T *y, const T& x)
	{
		for (size_t i = 0; i < n; ++i) y[i] *= x;
	}

	// div

	template<typename T>
	inline void vec_div(size_t n, const T* x1, const T *x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] / x2[i];
	}

	template<typename T>
	inline void vec_div(size_t n, const T* x1, const T& x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] / x2;
	}

	template<typename T>
	inline void vec_div(size_t n, const T& x1, const T* x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1 / x2[i];
	}

	template<typename T>
	inline void vec_div_inplace(size_t n, T *y, const T *x)
	{
		for (size_t i = 0; i < n; ++i) y[i] /= x[i];
	}

	template<typename T>
	inline void vec_div_inplace(size_t n, T *y, const T& x)
	{
		for (size_t i = 0; i < n; ++i) y[i] /= x;
	}

	template<typename T>
	inline void vec_div_inplace(size_t n, const T& x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x / y[i];
	}

	// negate

	template<typename T>
	inline void vec_negate(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = -x[i];
	}

	template<typename T>
	inline void vec_negate(size_t n, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = -y[i];
	}


	// absolute values

	template<typename T>
	inline void vec_abs(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::abs(x[i]);
	}


	/********************************************
	 *
	 *  Elementary Function Evaluation
	 *
	 *******************************************/

	// power and root functions

	template<typename T>
	inline void vec_sqr(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = sqr(x[i]);
	}

	template<typename T>
	inline void vec_sqrt(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::sqrt(x[i]);
	}

	template<typename T>
	inline void vec_rcp(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = T(1) / x[i];
	}

	template<typename T>
	inline void vec_rsqrt(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = T(1) / std::sqrt(x[i]);
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, const T* e, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::pow(x[i], e[i]);
	}

	template<typename T>
	inline void vec_pow(size_t n, const T *x, const T& e, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::pow(x[i], e);
	}


	// exponential and logarithm functions

	template<typename T>
	inline void vec_exp(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::exp(x[i]);
	}

	template<typename T>
	inline void vec_log(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::log(x[i]);
	}

	template<typename T>
	inline void vec_log10(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::log10(x[i]);
	}


	// rounding functions

	template<typename T>
	inline void vec_ceil(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::ceil(x[i]);
	}

	template<typename T>
	inline void vec_floor(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::floor(x[i]);
	}


	// trigonometric functions

	template<typename T>
	inline void vec_sin(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::sin(x[i]);
	}

	template<typename T>
	inline void vec_cos(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::cos(x[i]);
	}

	template<typename T>
	inline void vec_tan(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::tan(x[i]);
	}

	template<typename T>
	inline void vec_asin(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::asin(x[i]);
	}

	template<typename T>
	inline void vec_acos(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::acos(x[i]);
	}

	template<typename T>
	inline void vec_atan(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::atan(x[i]);
	}

	template<typename T>
	inline void vec_atan2(size_t n, const T *x1, const T* x2, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::atan2(x1[i], x2[i]);
	}


	// hyperbolic functions

	template<typename T>
	inline void vec_sinh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::sinh(x[i]);
	}

	template<typename T>
	inline void vec_cosh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::cosh(x[i]);
	}

	template<typename T>
	inline void vec_tanh(size_t n, const T *x, T *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = std::tanh(x[i]);
	}
}



/********************************************
 *
 *  Vendor-specific implementations
 *
 *******************************************/

#ifdef BCS_ENABLE_INTEL_IPPS
#include <bcslib/veccomp/intel_calc.h>
#endif


namespace bcs
{
	/********************************************
	 *
	 *  Order comparison
	 *
	 *******************************************/

	// equal to

	template<typename T>
	struct vec_vec_eq_ftor
	{
		bool operator() (const T& v1, const T& v2) const
		{
			return v1 == v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_eq(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_eq_ftor
	{
		T r;
		vec_sca_eq_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return v1 == r;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_eq(n, x1, r, y);
		}
	};


	// not equal

	template<typename T>
	struct vec_vec_ne_ftor
	{
		bool operator() (const T& v1, const T& v2) const
		{
			return v1 != v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_ne(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_ne_ftor
	{
		T r;
		vec_sca_ne_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return v1 != r;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_ne(n, x1, r, y);
		}
	};

	// greater than

	template<typename T>
	struct vec_vec_gt_ftor
	{
		bool operator() (const T& v1, const T& v2) const
		{
			return v1 > v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_gt(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_gt_ftor
	{
		T r;
		vec_sca_gt_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return v1 > r;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_gt(n, x1, r, y);
		}
	};


	template<typename T>
	struct sca_vec_gt_ftor
	{
		T r;
		sca_vec_gt_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return r > v1;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_lt(n, x1, r, y);
		}
	};


	// greater than or equal to

	template<typename T>
	struct vec_vec_ge_ftor
	{
		bool operator() (const T& v1, const T& v2) const
		{
			return v1 >= v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_ge(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_ge_ftor
	{
		T r;
		vec_sca_ge_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return v1 >= r;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_ge(n, x1, r, y);
		}
	};


	template<typename T>
	struct sca_vec_ge_ftor
	{
		T r;
		sca_vec_ge_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return r >= v1;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_le(n, x1, r, y);
		}
	};


	// less than

	template<typename T>
	struct vec_vec_lt_ftor
	{
		bool operator() (const T& v1, const T& v2) const
		{
			return v1 < v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_lt(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_lt_ftor
	{
		T r;
		vec_sca_lt_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return v1 < r;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_lt(n, x1, r, y);
		}
	};


	template<typename T>
	struct sca_vec_lt_ftor
	{
		T r;
		sca_vec_lt_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return r < v1;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_gt(n, x1, r, y);
		}
	};


	// less than or equal to

	template<typename T>
	struct vec_vec_le_ftor
	{
		bool operator() (const T& v1, const T& v2) const
		{
			return v1 <= v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, bool *y) const
		{
			vec_le(n, x1, x2, y);
		}
	};

	template<typename T>
	struct vec_sca_le_ftor
	{
		T r;
		vec_sca_le_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return v1 <= r;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_le(n, x1, r, y);
		}
	};


	template<typename T>
	struct sca_vec_le_ftor
	{
		T r;
		sca_vec_le_ftor(const T& r_)
		: r(r_) { }

		bool operator() (const T& v1) const
		{
			return r <= v1;
		}

		void operator() (size_t n, const T *x1, bool *y) const
		{
			vec_ge(n, x1, r, y);
		}
	};


	// max_each

	template<typename T>
	struct vec_vec_max_each_ftor
	{
		T operator() (const T& v1, const T& v2) const
		{
			using std::max;
			return max(v1, v2);
		}

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_max_each(n, x1, x2, y);
		}
	};


	// min_each

	template<typename T>
	struct vec_vec_min_each_ftor
	{
		T operator() (const T& v1, const T& v2) const
		{
			using std::min;
			return min(v1, v2);
		}

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_min_each(n, x1, x2, y);
		}
	};


	/********************************************
	 *
	 *  Bounding
	 *
	 *******************************************/

	template<typename T>
	struct vec_lbound_ftor
	{
		T b;
		vec_lbound_ftor(const T& b_) : b(b_) { }

		T operator() (const T& v) const
		{
			using std::max;
			return max(v, b);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_lbound(n, x, b, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_lbound_inplace(n, y, b);
		}
	};


	template<typename T>
	struct vec_ubound_ftor
	{
		T b;
		vec_ubound_ftor(const T& b_) : b(b_) { }

		T operator() (const T& v) const
		{
			using std::min;
			return min(v, b);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_ubound(n, x, b, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_ubound_inplace(n, y, b);
		}
	};


	template<typename T>
	struct vec_rgn_bound_ftor
	{
		T lb, ub;
		vec_rgn_bound_ftor(const T& lb_, const T& ub_) : lb(lb_), ub(ub_) { }

		T operator() (const T& v) const
		{
			return rgn_bound(v, lb, ub);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_rgn_bound(n, x, lb, ub, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_rgn_bound_inplace(n, y, lb, ub);
		}
	};


	template<typename T>
	struct vec_abound_ftor
	{
		T ab;
		vec_abound_ftor(const T& ab_) : ab(ab_) { }

		T operator() (const T& v) const
		{
			return rgn_bound(v, -ab, ab);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_abound(n, x, ab, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_abound_inplace(n, y, ab);
		}
	};


	/********************************************
	 *
	 *  Arithmetic Calculation
	 *
	 *******************************************/

	// addition

	template<typename T>
	struct vec_vec_add_ftor
	{
		T operator() (const T& v1, const T& v2) const
		{
			return v1 + v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_add(n, x1, x2, y);
		}

		void operator() (size_t n, T*y, const T*x1) const
		{
			vec_add_inplace(n, y, x1);
		}
	};

	template<typename T>
	struct vec_sca_add_ftor
	{
		T s;
		vec_sca_add_ftor(const T& s_)
		: s(s_) { }

		T operator() (const T& v1) const
		{
			return v1 + s;
		}

		void operator() (size_t n, const T *x1, T *y) const
		{
			vec_add(n, x1, s, y);
		}

		void operator() (size_t n, T* y) const
		{
			return vec_add_inplace(n, y, s);
		}
	};


	// subtraction

	template<typename T>
	struct vec_vec_sub_ftor
	{
		T operator() (const T& v1, const T& v2) const
		{
			return v1 - v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_sub(n, x1, x2, y);
		}

		void operator() (size_t n, T *y, const T *x1) const
		{
			vec_sub_inplace(n, y, x1);
		}
	};

	template<typename T>
	struct vec_sca_sub_ftor
	{
		T s;
		vec_sca_sub_ftor(const T& s_)
		: s(s_) { }

		T operator() (const T& v1) const
		{
			return v1 - s;
		}

		void operator() (size_t n, const T *x1, T *y) const
		{
			vec_sub(n, x1, s, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_sub_inplace(n, y, s);
		}
	};

	template<typename T>
	struct sca_vec_sub_ftor
	{
		T s;
		sca_vec_sub_ftor(const T& s_)
		: s(s_) { }

		T operator() (const T& v2) const
		{
			return s - v2;
		}

		void operator() (size_t n, const T *x2, T *y) const
		{
			vec_sub(n, s, x2, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_sub_inplace(n, s, y);
		}
	};


	// multiplication

	template<typename T>
	struct vec_vec_mul_ftor
	{
		T operator() (const T& v1, const T& v2) const
		{
			return v1 * v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_mul(n, x1, x2, y);
		}

		void operator() (size_t n, T*y, const T*x1) const
		{
			vec_mul_inplace(n, y, x1);
		}
	};

	template<typename T>
	struct vec_sca_mul_ftor
	{
		T s;
		vec_sca_mul_ftor(const T& s_)
		: s(s_) { }

		T operator() (const T& v1) const
		{
			return v1 * s;
		}

		void operator() (size_t n, const T *x1, T *y) const
		{
			vec_mul(n, x1, s, y);
		}

		void operator() (size_t n, T* y) const
		{
			return vec_mul_inplace(n, y, s);
		}
	};


	// division

	template<typename T>
	struct vec_vec_div_ftor
	{
		T operator() (const T& v1, const T& v2) const
		{
			return v1 / v2;
		}

		void operator() (size_t n, const T *x1, const T *x2, T *y) const
		{
			vec_div(n, x1, x2, y);
		}

		void operator() (size_t n, T *y, const T *x1) const
		{
			vec_div_inplace(n, y, x1);
		}
	};

	template<typename T>
	struct vec_sca_div_ftor
	{
		T s;
		vec_sca_div_ftor(const T& s_)
		: s(s_) { }

		T operator() (const T& v1) const
		{
			return v1 / s;
		}

		void operator() (size_t n, const T *x1, T *y) const
		{
			vec_div(n, x1, s, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_div_inplace(n, y, s);
		}
	};

	template<typename T>
	struct sca_vec_div_ftor
	{
		T s;
		sca_vec_div_ftor(const T& s_)
		: s(s_) { }

		T operator() (const T& v2) const
		{
			return s / v2;
		}

		void operator() (size_t n, const T *x2, T *y) const
		{
			vec_div(n, s, x2, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_div_inplace(n, s, y);
		}
	};



	// negation

	template<typename T>
	struct vec_neg_ftor
	{
		T operator() (const T& v) const
		{
			return -v;
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			vec_negate(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			vec_negate(n, y);
		}
	};


	// absolute value

	template<typename T>
	struct vec_abs_ftor
	{
		T operator() (const T& v) const
		{
			return std::abs(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_abs(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_abs(n, y, y);
		}
	};


	/********************************************
	 *
	 *  Elementary Function Evaluation
	 *
	 *******************************************/

	// power and root functions

	// sqr

	template<typename T>
	struct vec_sqr_ftor
	{
		T operator() (const T& v) const
		{
			return sqr(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_sqr(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_sqr(n, y, y);
		}
	};


	// sqrt

	template<typename T>
	struct vec_sqrt_ftor
	{
		T operator() (const T& v) const
		{
			return std::sqrt(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_sqrt(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_sqrt(n, y, y);
		}
	};


	// rcp

	template<typename T>
	struct vec_rcp_ftor
	{
		T operator() (const T& v) const
		{
			return T(1) / v;
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_rcp(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_rcp(n, y, y);
		}
	};


	// rsqrt

	template<typename T>
	struct vec_rsqrt_ftor
	{
		T operator() (const T& v) const
		{
			return T(1) / std::sqrt(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_rsqrt(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_rsqrt(n, y, y);
		}
	};


	// pow (vec-vec)

	template<typename T>
	struct vec_pow_ftor
	{
		T operator() (const T& v, const T& e) const
		{
			return std::pow(v, e);
		}

		void operator() (size_t n, const T *x, const T *e, T *y) const
		{
			return vec_pow(n, x, e, y);
		}

		void operator() (size_t n, T *y, const T* e) const
		{
			return vec_pow(n, y, e, y);
		}
	};


	// pow (vec-scalar)

	template<typename T>
	struct vec_sca_pow_ftor
	{
		T e;
		vec_sca_pow_ftor(const T& e_)
		: e(e_) { }

		T operator() (const T& v) const
		{
			return std::pow(v, e);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_pow(n, x, e, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_pow(n, y, e, y);
		}
	};



	// exponential and logarithm functions

	// exp

	template<typename T>
	struct vec_exp_ftor
	{
		T operator() (const T& v) const
		{
			return std::exp(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_exp(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_exp(n, y, y);
		}
	};

	// log

	template<typename T>
	struct vec_log_ftor
	{
		T operator() (const T& v) const
		{
			return std::log(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_log(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_log(n, y, y);
		}
	};

	// log10

	template<typename T>
	struct vec_log10_ftor
	{
		T operator() (const T& v) const
		{
			return std::log10(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_log10(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_log10(n, y, y);
		}
	};


	// rounding functions

	// floor

	template<typename T>
	struct vec_floor_ftor
	{
		T operator() (const T& v) const
		{
			return std::floor(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_floor(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_floor(n, y, y);
		}
	};

	// ceil

	template<typename T>
	struct vec_ceil_ftor
	{
		T operator() (const T& v) const
		{
			return std::ceil(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_ceil(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_ceil(n, y, y);
		}
	};


	// trigonometric functions

	// sin

	template<typename T>
	struct vec_sin_ftor
	{
		T operator() (const T& v) const
		{
			return std::sin(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_sin(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_sin(n, y, y);
		}
	};

	// cos

	template<typename T>
	struct vec_cos_ftor
	{
		T operator() (const T& v) const
		{
			return std::cos(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_cos(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_cos(n, y, y);
		}
	};

	// tan

	template<typename T>
	struct vec_tan_ftor
	{
		T operator() (const T& v) const
		{
			return std::tan(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_tan(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_tan(n, y, y);
		}
	};

	// asin

	template<typename T>
	struct vec_asin_ftor
	{
		T operator() (const T& v) const
		{
			return std::asin(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_asin(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_asin(n, y, y);
		}
	};

	// acos

	template<typename T>
	struct vec_acos_ftor
	{
		T operator() (const T& v) const
		{
			return std::acos(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_acos(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_acos(n, y, y);
		}
	};

	// atan

	template<typename T>
	struct vec_atan_ftor
	{
		T operator() (const T& v) const
		{
			return std::atan(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_atan(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_atan(n, y, y);
		}
	};

	// atan2

	template<typename T>
	struct vec_atan2_ftor
	{
		T operator() (const T& v1, const T& v2) const
		{
			return std::atan2(v1, v2);
		}

		void operator() (size_t n, const T *x1, const T* x2, T *y) const
		{
			return vec_atan2(n, x1, x2, y);
		}
	};


	// hyperbolic functions

	// sinh

	template<typename T>
	struct vec_sinh_ftor
	{
		T operator() (const T& v) const
		{
			return std::sinh(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_sinh(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_sinh(n, y, y);
		}
	};

	// cosh

	template<typename T>
	struct vec_cosh_ftor
	{
		T operator() (const T& v) const
		{
			return std::cosh(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_cosh(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_cosh(n, y, y);
		}
	};

	// tanh

	template<typename T>
	struct vec_tanh_ftor
	{
		T operator() (const T& v) const
		{
			return std::tanh(v);
		}

		void operator() (size_t n, const T *x, T *y) const
		{
			return vec_tanh(n, x, y);
		}

		void operator() (size_t n, T *y) const
		{
			return vec_tanh(n, y, y);
		}
	};
}


#endif 
