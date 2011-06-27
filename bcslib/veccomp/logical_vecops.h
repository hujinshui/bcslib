/**
 * @file logical_vecops.h
 *
 * Logical vector operation functions and corresponding functors
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_LOGICAL_VECOPS_H_
#define BCSLIB_LOGICAL_VECOPS_H_

namespace bcs
{
	// vectorized functions

	inline void vec_not(size_t n, const bool *x, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = !x[i];
	}

	inline void vec_and(size_t n, const bool *x1, const bool *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] & x2[i];
	}

	inline void vec_or(size_t n, const bool *x1, const bool *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] | x2[i];
	}

	inline void vec_xor(size_t n, const bool *x1, const bool *x2, bool *y)
	{
		for (size_t i = 0; i < n; ++i) y[i] = x1[i] ^ x2[i];
	}


	inline bool vec_all(size_t n, const bool *x)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (!x[i]) return false;
		}
		return true;
	}

	inline bool vec_any(size_t n, const bool *x)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (x[i]) return true;
		}
		return false;
	}

	template<typename TCount>
	inline TCount vec_count_true(size_t n, const bool *x, TCount c0)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (x[i]) ++c0;
		}
		return c0;
	}


	template<typename TCount>
	inline TCount vec_count_false(size_t n, const bool *x, TCount c0)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (!x[i]) ++c0;
		}
		return c0;
	}
}


namespace bcs
{
	// functors

	struct vec_not_ftor : public std::unary_function<bool, bool>
	{
		bool operator() (bool v) const
		{
			return !v;
		}

		void operator() (size_t n, const bool *x, bool *y) const
		{
			vec_not(n, x, y);
		}
	};

	struct vec_and_ftor : public std::binary_function<bool, bool, bool>
	{
		bool operator() (bool v1, bool v2) const
		{
			return v1 & v2;
		}

		void operator() (size_t n, const bool *x1, const bool *x2, bool *y) const
		{
			vec_and(n, x1, x2, y);
		}
	};

	struct vec_or_ftor : public std::binary_function<bool, bool, bool>
	{
		bool operator() (bool v1, bool v2) const
		{
			return v1 | v2;
		}

		void operator() (size_t n, const bool *x1, const bool *x2, bool *y) const
		{
			vec_or(n, x1, x2, y);
		}
	};

	struct vec_xor_ftor : public std::binary_function<bool, bool, bool>
	{
		bool operator() (bool v1, bool v2) const
		{
			return v1 ^ v2;
		}

		void operator() (size_t n, const bool *x1, const bool *x2, bool *y) const
		{
			vec_xor(n, x1, x2, y);
		}
	};

	struct vec_all_ftor
	{
		typedef bool result_type;

		bool operator() (size_t n, const bool *x) const
		{
			return vec_all(n, x);
		}
	};

	struct vec_any_ftor
	{
		typedef bool result_type;

		bool operator() (size_t n, const bool *x) const
		{
			return vec_any(n, x);
		}
	};

	template<typename TCount>
	struct vec_true_counter
	{
		typedef TCount result_type;
		TCount c0;

		vec_true_counter() : c0(0) { }
		vec_true_counter(TCount c) : c0(c) { }

		TCount operator() (size_t n, const bool *x) const
		{
			return vec_count_true(n, x, c0);
		}
	};


	template<typename TCount>
	struct vec_false_counter
	{
		typedef TCount result_type;
		TCount c0;

		vec_false_counter() : c0(0) { }
		vec_false_counter(TCount c) : c0(c) { }

		TCount operator() (size_t n, const bool *x) const
		{
			return vec_count_false(n, x, c0);
		}
	};

}

#endif 




