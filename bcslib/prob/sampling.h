/**
 * @file sampling.h
 *
 * The base of probabilistic sampling
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_SAMPLING_H
#define BCSLIB_SAMPLING_H


#include <bcslib/base/config.h>
#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_funcs.h>

#include <limits>
#include <cmath>

#if (__GNUC_MINOR__ >= 4)
#if (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_STD_DIR)
#include <random>
#elif (BCSLIB_TR1_INCLUDE_DIR == BCS_TR1_INCLUDE_TR1_DIR)
#include <tr1/random>
#endif
#define BCS_TR1_RANDOM_FROM_NAMESPACE BCS_TR1_FROM_NAMESPACE
#else
#include <boost/random.hpp>
#define BCS_TR1_RANDOM_FROM_NAMESPACE boost
#endif



namespace bcs
{


	typedef BCS_TR1_RANDOM_FROM_NAMESPACE::mt19937 default_tr1_rand_engine;


	class real_stream_for_diagnosis
	{
	public:
		real_stream_for_diagnosis(size_t n, const double *src)
		: m_n(n), m_src(src), m_i(0)
		{
		}

		void reset()
		{
			m_i = 0;
		}

		void seed(unsigned long s)
		{
			m_i = static_cast<size_t>(s % static_cast<unsigned long>(m_n));
		}

		double randf64()
		{
			double v = m_src[m_i];
			m_i = (m_i + 1) % m_n;
			return v;
		}

		float randf32()
		{
			return static_cast<float>(randf64());
		}

		void randf64_vec(size_t len, double *buf)
		{
			for (size_t i = 0; i < len; ++i)
			{
				buf[i] = randf64();
			}
		}

		void randf32_vec(size_t len, float *buf)
		{
			for (size_t i = 0; i < len; ++i)
			{
				buf[i] = randf32();
			}
		}

	private:
		size_t m_n;
		const double *m_src;
		size_t m_i;

	}; // end class real_stream_for_testing


	template<typename TEngine32=default_tr1_rand_engine>  // TEngine32 must generates 32-bit random bits
	class randstream
	{
	public:
		typedef typename TEngine32::result_type engine_result_type;

	public:
		explicit randstream(const TEngine32& eng = TEngine32())
		: m_eng(eng)
		{
		}

		void seed(unsigned long s)
		{
			m_eng.seed((uint32_t)(s));
		}

		uint32_t randu32()
		{
			return m_eng();
		}

		void randu32_vec(size_t len, uint32_t *buf)
		{
			for (size_t i = 0; i < len; ++i)
			{
				buf[i] = randu32();
			}
		}

		int32_t randi32()
		{
			return (int32_t)randu32();
		}

		void randi32_vec(size_t len, int32_t *buf)
		{
			uint32_t *ubuf = reinterpret_cast<uint32_t*>(buf);
			for (size_t i = 0; i < len; ++i)
			{
				ubuf[i] = randu32();
			}
		}

		float randf32()
		{
			return float(m_eng()) / std::numeric_limits<uint32_t>::max();
		}

		void randf32_vec(size_t len, float *buf)
		{
			for (size_t i = 0; i < len; ++i)
			{
				buf[i] = randf32();
			}
		}

		double randf64()
		{
			return double(m_eng()) / std::numeric_limits<uint32_t>::max();
		}

		void randf64_vec(size_t len, double *buf)
		{
			for (size_t i = 0; i < len; ++i)
			{
				buf[i] = randf64();
			}
		}


	private:
		TEngine32 m_eng;

	}; // end class randstream


	/******************************************************
	 *
	 * Convenient functors
	 *
	 *****************************************************/

	/**
	 * a functor for generating random uint32_t value up to n
	 *
	 * useful for working with functions like std::random_shuffle
	 */
	template<class RStream>
	class randu32_functor
	{
	public:
		explicit randu32_functor(RStream& rstream) : m_rstream(rstream)
		{
		}

		uint32_t operator() (uint32_t n)
		{
			return m_rstream.randu32() % n;
		}

	private:
		RStream& m_rstream;
	};


	template<class RStream>
	inline randu32_functor<RStream> randu32_fun(RStream& rstream)
	{
		return randu32_functor<RStream>(rstream);
	}


	/**
	 * a functor for generating random double real values
	 */
	template<class RStream>
	class rand_real_functor
	{
	public:
		explicit rand_real_functor(RStream& rstream) : m_rstream(rstream)
		{
		}

		double operator()()
		{
			return m_rstream.randf64();
		}

	private:
		RStream& m_rstream;
	};


	template<class RStream>
	inline rand_real_functor<RStream> rand_real_fun(RStream& rstream)
	{
		return rand_real_functor<RStream>(rstream);
	}


	/******************************************************
	 *
	 * distribution-specific RNGs
	 *
	 *****************************************************/

	namespace _detail
	{

		template<class RStream, typename TReal> struct rand_real_helper;

		template<class RStream>
		struct rand_real_helper<RStream, float>
		{
			static float gen(RStream& rstream)
			{
				return rstream.randf32();
			}

			static void gen_vec(RStream& rstream, size_t len, float *buf)
			{
				rstream.randf32_vec(len, buf);
			}
		};


		template<class RStream>
		struct rand_real_helper<RStream, double>
		{
			static double gen(RStream& rstream)
			{
				return rstream.randf64();
			}

			static void gen_vec(RStream& rstream, size_t len, double *buf)
			{
				rstream.randf64_vec(len, buf);
			}
		};

	}


	template<class RStream>
	inline bool get_bernoulli(RStream& rs, double p)
	{
		return rs.randf64() < p;
	}


	template<typename TInt, class RStream=randstream<> >
	class int_rng
	{
	public:
		typedef TInt value_type;
		typedef RStream rstream_type;

	public:
		static value_type get_uniform(rstream_type& rs, value_type n)  // ~ [0, n-1]
		{
			return static_cast<value_type>(rs.randu32() % static_cast<uint32_t>(n));
		}

		static void get_uniform(rstream_type& rs, value_type n, size_t len, value_type *buf)
		{
			for (size_t i = 0; i < len; ++i)
			{
				buf[i] = get_uniform(rs, n);
			}
		}

	}; // end class int_rng


	template<typename TReal, class RStream=randstream<> >
	class real_rng
	{
	public:
		typedef TReal value_type;
		typedef RStream rstream_type;

	private:
		typedef _detail::rand_real_helper<rstream_type, value_type> _helper;

	public:
		static value_type get_uniform(rstream_type& rs) // ~ U[0, 1)
		{
			return _helper::gen(rs);
		}

		static void get_uniform(rstream_type& rs, size_t len, value_type *buf)
		{
			_helper::gen_vec(rs, len, buf);
		}

		static value_type get_normal(rstream_type& rs) // ~ N(0, 1)
		{
			value_type u = _helper::gen(rs);
			value_type v = _helper::gen(rs);
			value_type x;
			normal_transform(u, v, x);
			return x;
		}

		static void get_normal(rstream_type& rs, size_t len, value_type *buf)
		{
			_helper::gen_vec(rs, len, buf);
			size_t m = len >> 1;
			for (size_t k = 0; k < m; ++k)
			{
				normal_transform(buf[0], buf[1], buf[0], buf[1]);
				buf += 2;
			}

			if (len & size_t(1))
			{
				normal_transform(*buf, _helper::gen(rs), *buf);
			}
		}

		static value_type get_exponential(rstream_type& rs)  // ~ Exp(1)
		{
			value_type u = _helper::gen(rs);
			return -std::log(u);
		}

		static void get_exponential(rstream_type& rs, size_t len, value_type *buf)
		{
			_helper::gen_vec(rs, len, buf);
			for (size_t i = 0; i < len; ++i)
			{
				exp_transform(buf[i]);
			}
		}

	private:
		static void normal_transform(value_type u, value_type v, value_type& x0)
		{
			const value_type two_pi = value_type(2 * 3.14159265358979323846);

			x0 = std::sqrt((-2) * std::log(u)) * std::cos( two_pi * v );
		}

		static void normal_transform(value_type u, value_type v, value_type& x0, value_type& x1)
		{
			const value_type two_pi = value_type(2 * 3.14159265358979323846);

			value_type r = std::sqrt((-2) * std::log(u));
			value_type t = two_pi * v;
			x0 = r * std::cos(t);
			x1 = r * std::sin(t);
		}

		static void exp_transform(value_type& u)
		{
			u = -std::log(u);
		}

	}; // end class real_rng



}

#endif 
