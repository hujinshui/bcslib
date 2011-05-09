/**
 * @file discrete_distr.h
 *
 * The classes for representing discrete distribution(s)
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_UNIFORM_DISTR_H
#define BCSLIB_UNIFORM_DISTR_H

#include <bcslib/prob/pdistribution.h>
#include <bcslib/prob/sampling.h>
#include <cmath>
#include <algorithm>
#include <functional>

namespace bcs
{

	template<typename T>
	inline T direct_sample_from_discrete_pmf(T K, const double *p, double u)
	{
		T k = 0;
		double F = *p;
		T Km1 = K - 1;
		while (u > F && k < Km1)
		{
			F += p[++k];
		}
		return u <= F ? k : k+1;
	}


	template<typename T>
	inline void discrete_cdf(T K, const double *p, double *F)
	{
		double s = 0;
		for (T k = 0; k < K; ++k)
		{
			F[k] = (s += p[k]);
		}
	}


	template<typename T>
	class discrete_sampler
	{
	public:
		typedef scalar_sample_t sample_category;
		typedef T value_type;

		enum methods
		{
			DSAMP_AUTO,
			DSAMP_DIRECT_METHOD,
			DSAMP_SORT_METHOD
		};

	public:
		discrete_sampler(value_type K, const double *p, methods method = DSAMP_AUTO)
		: m_K(K), m_method(method)
		{
			_init_internal((size_t)K, p);
		}

		value_type K() const
		{
			return m_K;
		}

		double average_search_length() const
		{
			return m_avg_searchlen;
		}

		methods method() const
		{
			return m_method;
		}

	public:

		template<class RStream>
		T operator() (RStream& rs)
		{
			return get_one(rs.randf64());
		}

		template<class RStream>
		void operator() (RStream& rs, T *v)
		{
			*v = get_one(rs.randf64());
		}

		template<class RStream>
		void operator() (RStream& rs, size_t len, T *buf)
		{
			bool use_sort = (m_method == DSAMP_SORT_METHOD) ||
					(m_method == DSAMP_AUTO && m_avg_searchlen > 5 * std::log(double(len)) + 3 );

			block<double> ublk(len);
			rs.randf64_vec(len, ublk.pbase());

			if (use_sort)
			{
				tbuffer wsbuf(len * sizeof(indexed_entry<double>));
				get_multi_by_sort(len, ublk.pbase(), buf, wsbuf);
			}
			else
			{
				get_multi(len, ublk.pbase(), buf);
			}
		}

	public:
		T get_one(double u)
		{
			int k = 0;
			while (u > m_F[k] && k < m_K) ++k;
			return k < m_K ? m_I[(int)k] : m_K;
		}

		void get_multi(size_t len, const double* u, T *dst)
		{
			for (size_t i = 0; i < len; ++i)
			{
				dst[i] = get_one(u[i]);
			}
		}

		void get_multi_by_sort(size_t len, const double *u, T *dst, tbuffer& wsbuf);

	private:
		void _init_internal(size_t K, const double *p);

	private:
		value_type m_K;
		methods m_method;

		const_memory_proxy<T> m_I;
		const_memory_proxy<double> m_F;
		double m_avg_searchlen;

	}; // end class discrete_sampler


	template<typename T>
	void discrete_sampler<T>::get_multi_by_sort(size_t len, const double *u, T *dst, tbuffer& wsbuf)
	{
		// prepare work space

		indexed_entry<double> *ws = (indexed_entry<double>*)(wsbuf.request(len * sizeof(indexed_entry<double>)));

		// sort u with indices

		copy_elements_attach_indices(u, ws, len);
		std::sort(ws, ws + len);

		// do assignment

		size_t i = 0;
		T k = T(0);
		double F_k = m_F[0];
		T I_k = m_I[0];

		while (i < len && k < m_K)
		{
			if (ws[i].value <= F_k)
			{
				dst[ws[i].index] = I_k;
				++ i;
			}
			else
			{
				++ k;
				F_k = m_F[(int)k];
				I_k = m_I[(int)k];
			}
		}

		// remaining one

		while (i < len)
		{
			dst[ws[i++].index] = m_K;
		}
	}

	template<typename T>
	void discrete_sampler<T>::_init_internal(size_t K, const double *p)
	{
		// sort p in descending order with indices

		block<indexed_entry<double> > bb(K);
		copy_elements_attach_indices(p, bb.pbase(), K);
		std::sort(bb.pbase(), bb.pend(), std::greater<indexed_entry<double> >());

		// generate m_I, m_F, and calculate avg_searchlen

		block<T> *pbI = new block<T>(K);
		block<double> *pbF = new block<double>(K);

		T *I = pbI->pbase();
		double *F = pbF->pbase();

		double lastF = 0;
		m_avg_searchlen = 0;
		for (size_t k = 0; k < K; ++k)
		{
			I[k] = bb[k].index;
			double p = bb[k].value;
			F[k] = (lastF += p);
			m_avg_searchlen += p * (k+1);
		}
		m_avg_searchlen += (1 - lastF) * K;

		m_I.set_block(pbI);
		m_F.set_block(pbF);
	}



	/**
	 * The class to represent a generic discrete distribution
	 *
	 * over [0, K-1]
	 */
	template<typename T>
	class discrete_distr
	{
	public:

		typedef discrete_distribution_t distribution_category;
		typedef scalar_sample_t sample_category;
		static const bool is_sampleable = true;

		typedef T value_type;
		typedef discrete_sampler<value_type> sampler_type;

	public:
		discrete_distr(T K, const double *p) : m_K(K), m_p(p)
		{
		}

		size_t dim() const
		{
			return 1;
		}

		value_type K() const
		{
			return m_K;
		}

		const double *pbase() const
		{
			return m_p;
		}

		double p(const value_type& x) const
		{
			return m_p[x];
		}

		template<typename RStream>
		value_type direct_sample(RStream& rstream)
		{
			return direct_sample_from_discrete_pmf(m_K, m_p, rstream.randf64());
		}

		sampler_type get_sampler()
		{
			return sampler_type(m_K, m_p);
		}

		sampler_type get_sampler(typename sampler_type::methods method)
		{
			return sampler_type(m_K, m_p, method);
		}


	private:
		value_type m_K;
		const double *m_p;

	}; // end class discrete_distr


	/**
	 * Make a uniform discrete distribution on a buffer and return
	 */
	template<typename T>
	inline discrete_distr<T> make_uniform_discrete_distr(T m, double *p)
	{
		double pv = 1.0 / double(m);
		for (T i = 0; i < m; ++i)
		{
			p[i] = pv;
		}
		return discrete_distr<T>(m, p);
	}



	/**
	 * Compute a normalized exponent
	 *
	 * Note: this operation can be done inplace, meaning that ps == pd
	 * is allowed.
	 *
	 */
	template<typename TValue>
	void normalize_exp(size_t n, const TValue *ps, TValue *pd)
	{
		TValue mv = ps[0];
		for (size_t i = 1; i < n; ++i)
		{
			if (ps[i] > mv) mv = ps[i];
		}

		TValue s = 0;
		for (size_t i = 0; i < n; ++i)
		{
			pd[i] = std::exp(ps[i] - mv);
			s += pd[i];
		}

		for (size_t i = 0; i < n; ++i)
		{
			pd[i] /= s;
		}
	}


	template<typename T>
	inline discrete_distr<T> make_discrete_distr_by_nrmexp(T m, const double *x, double *p)
	{
		normalize_exp((size_t)m, x, p);
		return discrete_distr<T>(m, p);
	}


}


#endif 
