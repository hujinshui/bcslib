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

#include <limits>
#include <tr1/random>


namespace bcs
{

	typedef std::tr1::mt19937 default_tr1_rgen_engine;


	namespace _detail
	{
		template<typename T, bool IsInt> struct _tr1_unif_mapper_a;

		template<typename T>
		struct _tr1_unif_mapper_a<T, true>
		{
			typedef std::tr1::uniform_int<T> dtype;
			static dtype default_distr()
			{
				return dtype(T(0), T(9));
			}
		};

		template<typename T>
		struct _tr1_unif_mapper_a<T, false>
		{
			typedef std::tr1::uniform_real<T> dtype;
			static dtype default_distr()
			{
				return dtype();
			}
		};

		template<typename T>
		struct _tr1_unif_mapper
		{
			typedef _tr1_unif_mapper_a<T, std::numeric_limits<T>::is_integer> _delegate;
			typedef typename _delegate::dtype dtype;
			static dtype default_distr()
			{
				return _delegate::default_distr();
			}
		};
	}


	/**
	 * Wrap a reference to engine as an engine type
	 */
	template<class TEngine>
	class tr1_engine_ref_wrapper
	{
	public:
		typedef TEngine engine_type;
		typedef typename engine_type::result_type result_type;

	public:
		tr1_engine_ref_wrapper(engine_type& eng) : m_eng(eng)
		{
		}

		result_type operator() ()
		{
			return m_eng();
		}

		result_type min() const
		{
			return m_eng.min();
		}

		result_type max() const
		{
			return m_eng.max();
		}

		engine_type& engine()
		{
			return m_eng;
		}

	private:
		engine_type& m_eng;
	};


	template<class TDistr, class TEngine=default_tr1_rgen_engine>
	class tr1_rand_base
	{
	public:
		typedef TEngine engine_type;
		typedef TDistr distribution_type;
		typedef typename TDistr::result_type result_type;

	public:
		tr1_rand_base(engine_type& eng, const distribution_type& distr) : m_gen(eng, distr)
		{
		}

		result_type operator()()
		{
			return m_gen();
		}

	protected:
		std::tr1::variate_generator<tr1_engine_ref_wrapper<engine_type>, distribution_type> m_gen;

	}; // end class tr1_rand


	template<typename T, class TEngine=default_tr1_rgen_engine>
	class tr1_randu : public tr1_rand_base<typename _detail::_tr1_unif_mapper<T>::dtype, TEngine>
	{
	public:
		typedef tr1_rand_base<typename _detail::_tr1_unif_mapper<T>::dtype, TEngine> base_type;
		typedef typename base_type::engine_type engine_type;
		typedef typename base_type::distribution_type distribution_type;
		typedef typename base_type::result_type result_type;

		explicit tr1_randu(engine_type& eng) : base_type(eng, _detail::_tr1_unif_mapper<T>::default_distr())
		{
		}

		tr1_randu(engine_type& eng, result_type vmin, result_type vmax) : base_type(eng, distribution_type(vmin, vmax))
		{
		}

		result_type min() const
		{
			return this->m_gen.distribution().min();
		}

		result_type max() const
		{
			return this->m_gen.distribution().max();
		}
	};


	template<typename T, class TEngine=default_tr1_rgen_engine>
	class tr1_randn : public tr1_rand_base<std::tr1::normal_distribution<T>, TEngine>
	{
	public:
		typedef tr1_rand_base<std::tr1::normal_distribution<T>, TEngine> base_type;
		typedef typename base_type::engine_type engine_type;
		typedef typename base_type::distribution_type distribution_type;
		typedef typename base_type::result_type result_type;

		explicit tr1_randn(engine_type& eng) : base_type(eng, distribution_type())
		{
		}
	};

}

#endif 
