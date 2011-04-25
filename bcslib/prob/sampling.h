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


#include <tr1/random>


namespace bcs
{

	typedef std::tr1::mt19937 default_tr1_rgen_engine;

	template<typename T=double, typename TEngine=default_tr1_rgen_engine>
	class tr1_rand
	{
	public:
		typedef T value_type;
		typedef TEngine engine_type;

	public:
		tr1_rand(engine_type eng = engine_type()) : m_eng(eng), m_distr(T(0), T(1))
		{

		}

		value_type operator() ()
		{
			return m_distr(m_eng);
		}

	private:
		engine_type& m_eng;
		std::tr1::uniform_real<value_type> m_distr;

	}; // end class tr1_random_generator


}

#endif 
