/**
 * @file basic_funcs.h
 *
 * Definitions of some basic functions
 *
 * @author Dahua Lin
 */

#ifndef BCSLIB_BASIC_FUNCS_H
#define BCSLIB_BASIC_FUNCS_H

#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>

#include <utility>
#include <functional>
#include <cstdlib>
#include <cmath>

namespace bcs
{
	// inplace functors

	template<typename T>
	struct inplace_plus
	{
		void operator() (T& y, const T& x) const
		{
			y += x;
		}
	};

	template<typename T>
	struct inplace_minus
	{
		void operator() (T& y, const T& x) const
		{
			y -= x;
		}
	};


	template<typename T>
	struct inplace_multiplies
	{
		void operator() (T& y, const T& x) const
		{
			y *= x;
		}
	};


	template<typename T>
	struct inplace_divides
	{
		void operator() (T& y, const T& x) const
		{
			y /= x;
		}
	};


	// elementary functions as functors

	template<typename T>
	inline T sqr(const T& x)
	{
		return x * x;
	}


	template<typename T>
	struct abs_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::abs(x);
		}
	};



	template<typename T>
	struct sqrt_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::sqrt(x);
		}
	};

	template<typename T>
	struct sqr_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return sqr(x);
		}
	};

	template<typename T>
	struct pow_fun : public std::binary_function<T, T, T>
	{
		T operator() (const T& x, const T& e) const
		{
			return std::pow(x, e);
		}
	};

	template<typename T>
	struct pow_n_fun : public std::unary_function<T, T>
	{
		int n;
		pow_n_fun(int exponent) : n(exponent) { }

		T operator() (const T& x)
		{
			return std::pow(x, n);
		}
	};


	template<typename T>
	struct exp_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::exp(x);
		}
	};

	template<typename T>
	struct log_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::log(x);
		}
	};

	template<typename T>
	struct log10_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::log10(x);
		}
	};

	template<typename T>
	struct ceil_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::ceil(x);
		}
	};


	template<typename T>
	struct floor_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::floor(x);
		}
	};


	template<typename T>
	struct sin_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::sin(x);
		}
	};

	template<typename T>
	struct cos_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::cos(x);
		}
	};

	template<typename T>
	struct tan_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::tan(x);
		}
	};

	template<typename T>
	struct asin_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::asin(x);
		}
	};

	template<typename T>
	struct acos_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::acos(x);
		}
	};

	template<typename T>
	struct atan_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::atan(x);
		}
	};

	template<typename T>
	struct atan2_fun : public std::binary_function<T, T, T>
	{
		T operator() (const T& y, const T& x) const
		{
			return std::atan2(y, x);
		}
	};

	template<typename T>
	struct sinh_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::sinh(x);
		}
	};

	template<typename T>
	struct cosh_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::cosh(x);
		}
	};

	template<typename T>
	struct tanh_fun : public std::unary_function<T, T>
	{
		T operator() (const T& x) const
		{
			return std::tanh(x);
		}
	};


	// accumulators

	class empty_accumulation : public base_exception
	{
	public:
		empty_accumulation(const char *msg)
		: base_exception(msg)
		{
		}
	};


	template<typename T>
	struct sum_accumulator
	{
		typedef T value_type;

		value_type value;

		sum_accumulator(const T& v0) : value(v0)
		{
		}

		void take_in(const T& v)
		{
			value += v;
		}

		value_type get_value() const
		{
			return value;
		}

		static T empty_value()
		{
			return T(0);
		}
	};


	template<typename T>
	struct prod_accumulator
	{
		typedef T value_type;

		value_type value;

		prod_accumulator(const T& v0) : value(v0)
		{
		}

		void take_in(const T& v)
		{
			value *= v;
		}

		value_type get_value() const
		{
			return value;
		}

		static T empty_value()
		{
			return T(1);
		}
	};


	template<typename T>
	struct min_accumulator
	{
		typedef T value_type;

		value_type value;

		min_accumulator(const T& v0) : value(v0)
		{
		}

		void take_in(const T& v)
		{
			if (v < value) value = v;
		}

		value_type get_value() const
		{
			return value;
		}

		static T empty_value()
		{
			throw empty_accumulation("Cannot take minimum over empty collection.");
		}
	};


	template<typename T>
	struct max_accumulator
	{
		typedef T value_type;

		value_type value;

		max_accumulator(const T& v0) : value(v0)
		{
		}

		void take_in(const T& v)
		{
			if (v > value) value = v;
		}

		value_type get_value() const
		{
			return value;
		}

		static T empty_value()
		{
			throw empty_accumulation("Cannot take maximum over empty collection.");
		}
	};


	template<typename T>
	struct minmax_accumulator
	{
		typedef std::pair<T, T> value_type;

		value_type min_value;
		value_type max_value;

		minmax_accumulator(const T& v) : min_value(v), max_value(v)
		{

		}

		minmax_accumulator(const T& vmin0, const T& vmax0) : min_value(vmin0), max_value(vmax0)
		{
		}

		minmax_accumulator(const std::pair<T, T>& minmax0) : min_value(minmax0.first), max_value(minmax0.second)
		{
		}

		void take_in(const T& v)
		{
			if (v < min_value)
			{
				min_value = v;
			}
			else if (v > max_value)
			{
				max_value = v;
			}
		}

		value_type get_value() const
		{
			return std::make_pair(min_value, max_value);
		}

		static T empty_value()
		{
			throw empty_accumulation("Cannot take minimum and maximum over empty collection.");
		}
	};




	// min

	template<typename T>
	inline const T& min(const T& a, const T& b)
	{
		return a < b ? a : b;
	}

	template<typename T>
	inline T& min(T& a, T& b)
	{
		return a < b ? a : b;
	}

	template<typename T>
	inline const T& min(const T& a, const T& b, const T& c)
	{
		return min(min(a, b), c);
	}

	template<typename T>
	inline T& min(T& a, T& b, T& c)
	{
		return min(min(a, b), c);
	}

	template<typename T>
	inline const T& min(const T& a, const T& b, const T& c, const T& d)
	{
		return min(min(a, b), min(c, d));
	}

	template<typename T>
	inline T& min(T& a, T& b, T& c, T& d)
	{
		return min(min(a, b), min(c, d));
	}

	// max

	template<typename T>
	inline const T& max(const T& a, const T& b)
	{
		return a > b ? a : b;
	}

	template<typename T>
	inline T& max(T& a, T& b)
	{
		return a > b ? a : b;
	}

	template<typename T>
	inline const T& max(const T& a, const T& b, const T& c)
	{
		return max(max(a, b), c);
	}

	template<typename T>
	inline T& max(T& a, T& b, T& c)
	{
		return max(max(a, b), c);
	}

	template<typename T>
	inline const T& max(const T& a, const T& b, const T& c, const T& d)
	{
		return max(max(a, b), max(c, d));
	}

	template<typename T>
	inline T& max(T& a, T& b, T& c, T& d)
	{
		return max(max(a, b), max(c, d));
	}

	// min_max

	template<typename T>
	inline std::pair<const T&, const T&> min_max(const T& a, const T& b)
	{
		typedef std::pair<const T&, const T&> pair_t;
		return a < b ? pair_t(a, b) : pair_t(b, a);
	}

	template<typename T>
	inline std::pair<T&, T&> min_max(T& a, T& b)
	{
		typedef std::pair<T&, T&> pair_t;
		return a < b ? pair_t(a, b) : pair_t(b, a);
	}

	template<typename T>
	inline std::pair<const T&, const T&> min_max(const T& a, const T& b, const T& c)
	{
		typedef std::pair<const T&, const T&> pair_t;
		pair_t ab = min_max(a, b);
		return pair_t(min(ab.first, c), max(ab.second, c));
	}

	template<typename T>
	inline std::pair<T&, T&> min_max(T& a, T& b, T& c)
	{
		typedef std::pair<T&, T&> pair_t;
		pair_t ab = min_max(a, b);
		return pair_t(min(ab.first, c), max(ab.second, c));
	}

	// ssort

	template<typename T>
	inline void ssort(T& a, T& b)
	{
		if (a > b)
		{
			T c(a);
			a = b;
			b = c;
		}
	}

	template<typename T>
	inline void ssort(T& a, T& b, T& c)
	{
		ssort(a, b);
		ssort(b, c);
		ssort(a, b);
	}


	// copy_n

	template<typename InputIter, typename OutputIter>
	void copy_n(InputIter src, size_t n, OutputIter dst)
	{
		for (size_t i = 0; i < n; ++i)
		{
			*dst = *src;
			++ src;
			++ dst;
		}
	}

	template<typename Iter>
	size_t count(Iter first, Iter last)
	{
		size_t c = 0;
		while(first != last)
		{
			++ c;
			++ first;
		}
		return c;
	}


}

#endif
