/**
 * @file vecstat_functors.h
 *
 * The wrapper of vecstat computation routines into functors
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_VECSTAT_FUNCTORS_H_
#define BCSLIB_VECSTAT_FUNCTORS_H_

#include <bcslib/veccomp/vecstat.h>

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
