/**
 * @file vecstat.h
 *
 * Statistics over vectors
 *
 * @author Dahua Lin
 */

#ifndef BCSLIB_VECSTAT_H
#define BCSLIB_VECSTAT_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/mathfun.h>
#include <bcslib/base/arg_check.h>

#include <algorithm>

namespace bcs
{

	// sum & mean

	template<typename T>
	inline T vec_sum(size_t n, const T *x)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			s += x[i];
		}
		return s;
	}

	template<typename T>
	inline T vec_dot_prod(size_t n, const T *x, const T *y)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			s += x[i] * y[i];
		}
		return s;
	}

	template<typename T>
	inline T vec_sum_log(size_t n, const T *x)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			s += std::log(x[i]);
		}
		return s;
	}

	template<typename T>
	inline T vec_sum_xlogy(size_t n, const T *x, const T *y)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			if (x[i] != 0)
			{
				s += x[i] * std::log(y[i]);
 			}
		}
		return s;
	}


	template<typename T>
	inline T vec_mean(size_t n, const T *x)
	{
		check_arg(n > 0, "n must be positive for vec_mean.");
		return vec_sum(n, x) / (T)n;
	}


	// min & max

	template<typename T>
	inline T vec_min(size_t n, const T *x)
	{
		check_arg(n > 0, "n must be positive for vec_min.");

		T r(x[0]);
		for (size_t i = 1; i < n; ++i)
		{
			if (x[i] < r) r = x[i];
		}
		return r;
	}

	template<typename T>
	inline T vec_max(size_t n, const T *x)
	{
		check_arg(n > 0, "n must be positive for vec_max.");

		T r(x[0]);
		for (size_t i = 1; i < n; ++i)
		{
			if (x[i] > r) r = x[i];
		}
		return r;
	}

	template<typename T>
	inline std::pair<T, T> vec_minmax(size_t n, const T *x)
	{
		check_arg(n > 0, "n must be positive for vec_minmax.");

		T vmin(x[0]);
		T vmax(x[0]);

		for (size_t i = 1; i < n; ++i)
		{
			if (x[i] > vmax) vmax = x[i];
			else if (x[i] < vmin) vmin = x[i];
		}
		return std::make_pair(vmin, vmax);
	}

	template<typename T>
	inline index_t vec_min_index(size_t n, const T *x)
	{
		check_arg(n > 0, "n must be positive for vec_min_index.");

		size_t p = 0;
		T v(x[0]);

		for (size_t i = 1; i < n; ++i)
		{
			if (x[i] < v)
			{
				p = i;
				v = x[i];
			}
		}

		return (index_t)p;
	}


	template<typename T>
	inline index_t vec_max_index(size_t n, const T *x)
	{
		check_arg(n > 0, "n must be positive for vec_max_index.");

		size_t p = 0;
		T v(x[0]);

		for (size_t i = 1; i < n; ++i)
		{
			if (x[i] > v)
			{
				p = i;
				v = x[i];
			}
		}

		return (index_t)p;
	}


	// median

	template<typename T>
	inline T vec_median_inplace(size_t n, T *x)
	{
		check_arg(n > 0, "n must be positive for vec_index_min.");

		if (n == 1)
		{
			return *x;
		}
		else if (n == 2)
		{
			T x0 = x[0];
			T x1 = x[1];
			return x0 + (x1 - x0) / 2;
		}
		else if (n % 2 == 0) // even
		{
	        T *pm = x + (n/2);
	        std::nth_element(x, pm, x+n);

	        T v1 = *pm;
	        T v0 = *(std::max_element(x, pm));

	        return v0 + (v1 - v0) / 2;
		}
		else  // odd
		{
			T *pm = x + (n/2);
			std::nth_element(x, pm, x+n);
			return *pm;
		}
	}


	// norms

	template<typename T>
	inline T vec_norm_L0(size_t n, const T *x)
	{
		size_t c = 0;
		for (size_t i = 0; i < n; ++i)
		{
			if (x[i] != 0) ++c;
		}
		return (T)c;
	}

	template<typename T>
	inline T vec_diff_norm_L0(size_t n, const T *x, const T *y)
	{
		size_t c = 0;
		for (size_t i = 0; i < n; ++i)
		{
			if (x[i] != y[i]) ++c;
		}
		return (T)c;
	}


	template<typename T>
	inline T vec_norm_L1(size_t n, const T *x)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			s += std::abs(x[i]);
		}
		return s;
	}

	template<typename T>
	inline T vec_diff_norm_L1(size_t n, const T *x, const T *y)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			s += std::abs(x[i] - y[i]);
		}
		return s;
	}


	template<typename T>
	inline T vec_sqrsum(size_t n, const T *x)
	{
		return vec_dot_prod(n, x, x);
	}

	template<typename T>
	inline T vec_diff_sqrsum(size_t n, const T *x, const T *y)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			s += sqr(x[i] - y[i]);
		}
		return s;
	}


	template<typename T>
	inline T vec_norm_L2(size_t n, const T *x)
	{
		return std::sqrt(vec_sqrsum(n, x));
	}

	template<typename T>
	inline T vec_diff_norm_L2(size_t n, const T *x, const T *y)
	{
		return std::sqrt(vec_diff_sqrsum(n, x, y));
	}


	template<typename T>
	inline T vec_norm_Linf(size_t n, const T *x)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			T a = std::abs(x[i]);
			if (a > s) s = a;
		}

		return s;
	}

	template<typename T>
	inline T vec_diff_norm_Linf(size_t n, const T *x, const T* y)
	{
		T s(0);
		for (size_t i = 0; i < n; ++i)
		{
			T a = std::abs(x[i] - y[i]);
			if (a > s) s = a;
		}

		return s;
	}
}



namespace bcs
{

	// sum & mean

	template<typename T>
	struct vec_sum_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_sum(n, x);
		}
	};


	template<typename T>
	struct vec_dot_prod_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x, const T *y) const
		{
			return vec_dot_prod(n, x, y);
		}
	};

	template<typename T>
	struct vec_sum_log_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_sum_log(n, x);
		}
	};


	template<typename T>
	struct vec_sum_xlogy_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x, const T *y) const
		{
			return vec_sum_xlogy(n, x, y);
		}
	};


	template<typename T>
	struct vec_mean_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_mean(n, x);
		}
	};


	// min & max

	template<typename T>
	struct vec_min_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_min(n, x);
		}
	};

	template<typename T>
	struct vec_max_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_max(n, x);
		}
	};

	template<typename T>
	struct vec_minmax_ftor
	{
		typedef std::pair<T, T> result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_minmax(n, x);
		}
	};

	template<typename T>
	struct vec_min_index_ftor
	{
		typedef index_t result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_min_index(n, x);
		}
	};

	template<typename T>
	struct vec_max_index_ftor
	{
		typedef index_t result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_max_index(n, x);
		}
	};


	// norms

	template<typename T>
	struct vec_norm_L0_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_norm_L0(n, x);
		}
	};

	template<typename T>
	struct vec_diff_norm_L0_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x, const T *y) const
		{
			return vec_diff_norm_L0(n, x, y);
		}
	};


	template<typename T>
	struct vec_norm_L1_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_norm_L1(n, x);
		}
	};

	template<typename T>
	struct vec_diff_norm_L1_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x, const T *y) const
		{
			return vec_diff_norm_L1(n, x, y);
		}
	};


	template<typename T>
	struct vec_sqrsum_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_sqrsum(n, x);
		}
	};

	template<typename T>
	struct vec_diff_sqrsum_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x, const T *y) const
		{
			return vec_diff_sqrsum(n, x, y);
		}
	};


	template<typename T>
	struct vec_norm_L2_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_norm_L2(n, x);
		}
	};

	template<typename T>
	struct vec_diff_norm_L2_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x, const T *y) const
		{
			return vec_diff_norm_L2(n, x, y);
		}
	};


	template<typename T>
	struct vec_norm_Linf_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x) const
		{
			return vec_norm_Linf(n, x);
		}
	};

	template<typename T>
	struct vec_diff_norm_Linf_ftor
	{
		typedef T result_type;

		result_type operator() (size_t n, const T *x, const T *y) const
		{
			return vec_diff_norm_Linf(n, x, y);
		}
	};

}


#endif

