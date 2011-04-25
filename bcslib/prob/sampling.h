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

#if ((BCSLIB_COMPILER == BCSLIB_GCC) && (__GNUC_MINOR__ <= 2))
#include <boost/tr1/random.hpp>  	// g++ 4.2 has several bugs in <random>
#else
#include <tr1/random>
#endif


namespace bcs
{

	typedef std::tr1::mt19937 default_tr1_rgen_engine;


	template<typename T, class TEngine=default_tr1_rgen_engine>
	class uniform_rgen
	{
	public:
		typedef TEngine engine_type;
		typedef std::tr1::uniform_real<T> distribution_type;
		typedef T result_type;

		explicit uniform_rgen(engine_type& eng)
		: m_gen(eng, distribution_type())
		{
		}

		uniform_rgen(engine_type& eng, result_type ub)
		: m_gen(eng, distribution_type(0, ub))
		{
		}

		uniform_rgen(engine_type& eng, result_type lb, result_type ub)
		: m_gen(eng, distribution_type(lb, ub))
		{
		}

		result_type min() const
		{
			return m_gen.distribution().min();
		}

		result_type max() const
		{
			return m_gen.distribution().max();
		}

		result_type mean() const
		{
			return (min() + max()) / 2;
		}

		result_type span() const
		{
			return max() - min();
		}

		result_type variance() const
		{
			result_type s = span();
			return  (s / 12) * s;
		}

		result_type operator()()
		{
			return m_gen();
		}

	private:
		std::tr1::variate_generator<engine_type&, distribution_type> m_gen;

	}; // end class uniform_rgen


	template<typename T, class TEngine=default_tr1_rgen_engine>
	class duniform_rgen
	{
	public:
		typedef TEngine engine_type;
		typedef std::tr1::uniform_int<T> distribution_type;
		typedef T result_type;

		explicit duniform_rgen(engine_type& eng)
		: m_gen(eng, distribution_type())
		{
		}

		duniform_rgen(engine_type& eng, result_type ub)
		: m_gen(eng, distribution_type(0, ub))
		{
		}

		duniform_rgen(engine_type& eng, result_type lb, result_type ub)
		: m_gen(eng, distribution_type(lb, ub))
		{
		}

		result_type min() const
		{
			return m_gen.distribution().min();
		}

		result_type max() const
		{
			return m_gen.distribution().max();
		}

		double mean() const
		{
			return (min() + max()) * 0.5;
		}

		result_type span() const
		{
			return max() - min();
		}

		result_type num() const
		{
			return span() + 1;
		}

		double variance() const
		{
			double n = num();
			return  (n*n - 1) / 12;
		}

		result_type operator()()
		{
			return m_gen();
		}

	private:
		std::tr1::variate_generator<engine_type&, distribution_type> m_gen;

	}; // end class duniform_rgen




	template<typename T, class TEngine=default_tr1_rgen_engine>
	class normal_rgen
	{
	public:
		typedef TEngine engine_type;
		typedef std::tr1::normal_distribution<T> distribution_type;
		typedef T result_type;

		explicit normal_rgen(engine_type& eng)
		: m_gen(eng, distribution_type())
		{
		}

		normal_rgen(engine_type& eng, result_type sig)
		: m_gen(eng, distribution_type(0, sig))
		{
		}

		normal_rgen(engine_type& eng, result_type mu, result_type sig)
		: m_gen(eng, distribution_type(mu, sig))
		{
		}

		result_type mean() const
		{
			return m_gen.distribution().mean();
		}

		result_type sigma() const
		{
			return m_gen.distribution().sigma();
		}

		result_type variance() const
		{
			return sqr(m_gen.distribution().sigma());
		}

		result_type operator()()
		{
			return m_gen();
		}

	private:
		std::tr1::variate_generator<engine_type&, distribution_type> m_gen;

	}; // end class normal_rgen


	template<class TEngine=default_tr1_rgen_engine>
	class bernoulli_rgen
	{
	public:
		typedef TEngine engine_type;
		typedef std::tr1::bernoulli_distribution distribution_type;
		typedef bool result_type;

		explicit bernoulli_rgen(engine_type& eng)
		: m_gen(eng, distribution_type())
		{
		}

		bernoulli_rgen(engine_type& eng, double p)
		: m_gen(eng, distribution_type(p))
		{
		}

		double p() const
		{
			return m_gen.distribution().p();
		}

		double mean() const
		{
			return p();
		}

		double variance() const
		{
			return p() * (1 - p());
		}

		result_type operator()()
		{
			return m_gen();
		}

	private:
		std::tr1::variate_generator<engine_type&, distribution_type> m_gen;

	}; // end class bernoulli_rgen


	/**
	 * The random number generator for geometric distribution
	 *
	 * Note: we use the definition by pmf(k) = (1-p)^{k-1} p
	 */
	template<typename T, class TEngine=default_tr1_rgen_engine>
	class geometric_rgen
	{
	public:
		typedef TEngine engine_type;
		typedef std::tr1::geometric_distribution<T, double> distribution_type;
		typedef T result_type;

		explicit geometric_rgen(engine_type& eng)
		: m_gen(eng, distribution_type())
		{
		}

		geometric_rgen(engine_type& eng, double p)
		: m_gen(eng, distribution_type(1-p))
		{
		}

		double p() const
		{
			return 1 - m_gen.distribution().p();
		}

		double mean() const
		{
			return 1.0 / p();
		}

		double variance() const
		{
			return (1 - p()) / sqr(p());
		}

		result_type operator()()
		{
			return m_gen();
		}

	private:
		std::tr1::variate_generator<engine_type&, distribution_type> m_gen;

	}; // end class geometric_rgen


	template<typename T, class TEngine=default_tr1_rgen_engine>
	class poisson_rgen
	{
	public:
		typedef TEngine engine_type;
		typedef std::tr1::poisson_distribution<T, double> distribution_type;
		typedef T result_type;

		explicit poisson_rgen(engine_type& eng)
		: m_gen(eng, distribution_type())
		{
		}

		poisson_rgen(engine_type& eng, double mu)
		: m_gen(eng, distribution_type(mu))
		{
		}

		double mean() const
		{
			return m_gen.distribution().mean();
		}

		double variance() const
		{
			return mean();
		}

		result_type operator()()
		{
			return m_gen();
		}

	private:
		std::tr1::variate_generator<engine_type&, distribution_type> m_gen;

	}; // end class poisson_rgen


	template<typename T, class TEngine=default_tr1_rgen_engine>
	class binomial_rgen
	{
	public:
		typedef TEngine engine_type;
		typedef std::tr1::binomial_distribution<T, double> distribution_type;
		typedef T result_type;

		binomial_rgen(engine_type& eng, result_type n, double p = 0.5)
		: m_gen(eng, distribution_type(n, p))
		{
		}

		double p() const
		{
			return m_gen.distribution().p();
		}

		result_type n() const
		{
			return m_gen.distribution().t();
		}

		double mean() const
		{
			return n() * p();
		}

		double variance() const
		{
			return n() * p() * (1 - p());
		}

		result_type operator()()
		{
			return m_gen();
		}

	private:
		std::tr1::variate_generator<engine_type&, distribution_type> m_gen;

	}; // end class binomial_rgen


	template<typename T, class TEngine=default_tr1_rgen_engine>
	class exponential_rgen
	{
	public:
		typedef TEngine engine_type;
		typedef std::tr1::exponential_distribution<T> distribution_type;
		typedef T result_type;

		explicit exponential_rgen(engine_type& eng)
		: m_gen(eng, distribution_type())
		{
		}

		exponential_rgen(engine_type& eng, result_type lambda)
		: m_gen(eng, distribution_type(lambda))
		{
		}

		result_type lambda() const
		{
			return m_gen.distribution().lambda();
		}

		result_type mean() const
		{
			return T(1) / lambda();
		}

		result_type variance() const
		{
			return sqr(mean());
		}

		result_type operator()()
		{
			return m_gen();
		}

	private:
		std::tr1::variate_generator<engine_type&, distribution_type> m_gen;

	}; // end class exponential_rgen



	template<typename T, class TEngine=default_tr1_rgen_engine>
	class gamma_rgen
	{
	public:
		typedef TEngine engine_type;
		typedef std::tr1::gamma_distribution<T> distribution_type;
		typedef T result_type;

		explicit gamma_rgen(engine_type& eng)
		: m_gen(eng, distribution_type())
		{
		}

		gamma_rgen(engine_type& eng, result_type alpha)
		: m_gen(eng, distribution_type(alpha))
		{
		}

		result_type alpha() const
		{
			return m_gen.distribution().alpha();
		}

		result_type mean() const
		{
			return alpha();
		}

		result_type variance() const
		{
			return alpha();
		}

		result_type operator()()
		{
			return m_gen();
		}

	private:
		std::tr1::variate_generator<engine_type&, distribution_type> m_gen;

	}; // end class gamma_rgen

}

#endif 
