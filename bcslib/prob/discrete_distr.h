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
					(m_method == DSAMP_AUTO && m_K > 3 * m_avg_searchlen);

			block<double> ublk(len);
			rs.randf64_vec(len, ublk.pbase());

			if (use_sort)
			{
				block<indexed_entry<double> > wsblk;
				get_multi_by_sort(len, ublk.pbase(), buf, wsblk.pbase());
			}
			else
			{
				get_multi(len, ublk.pbase(), buf);
			}
		}

		value_type K() const
		{
			return m_K;
		}

	public:
		T get_one(double u)
		{
			int k = 0;
			while (u > m_F[k] && k < m_K) ++k;
			return m_I[(int)k];
		}

		void get_multi(size_t len, const double* u, T *dst)
		{
			for (size_t i = 0; i < len; ++i)
			{
				dst[i] = get_one(u[i]);
			}
		}

		void get_multi_by_sort(size_t len, const double *u, T *dst, indexed_entry<double>* ws)
		{
			// sort u with indices

			copy_elements_attach_indices(u, ws, len);
			std::sort(ws, ws + len);

			// do assignment

			size_t i = 0;
			T k = T(0);
			double F_k = m_F[0];
			double I_k = m_I[0];

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
				dst[ws[i].index] = m_K;
			}
		}

	private:
		void _init_internal(size_t K, const double *p)
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
				m_avg_searchlen += (k+1) * p;
			}
			m_avg_searchlen += (1 - F) * K;

			m_I.reset(pbI);
			m_F.reset(pbF);
		}


	private:
		value_type m_K;
		methods m_method;

		const_memory_proxy<T> m_I;
		const_memory_proxy<double> m_F;
		double m_avg_searchlen;

	}; // end class discrete_sampler



	/**
	 * The class to represent a generic discrete distribution
	 *
	 * over [0, m-1]
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


	private:
		value_type m_K;
		const double *m_p;

	}; // end class discrete_distr


	template<typename T>
	class discrete_distr_set : public pdistribution_set_t<discrete_distr<T> >
	{
	public:
		typedef discrete_distr<T> distribution_type;
		typedef typename distribution_type::value_type value_type;
		typedef typename distribution_type::sample_type sample_type;

	public:
		discrete_distr_set(T m, size_t n, const double *p)
		: m_m(m), m_n(n), m_p(p)
		{
		}

		size_t dim() const
		{
			return 1;
		}

		size_t domain_size() const
		{
			return (size_t)m_m;
		}

		size_t ndistrs() const
		{
			return m_n;
		}

		value_type m() const
		{
			return m_m;
		}

		distribution_type distribution(size_t i) const
		{
			return distribution_type(m_m, m_p + m_m * i);
		}

	private:
		value_type m_m;
		size_t m_n;
		const double *m_p;

	}; // end class discrete_distr_set


	/**
	 * Make a uniform discrete distribution on a buffer and return
	 */
	template<typename T>
	inline discrete_distr<T> make_uniform_discrete_distr(T m, double *p)
	{
		double pv = 1.0 / ds;
		for (T i = 0; i < m; ++i)
		{
			p[i] = pv;
		}
		return discrete_distr<T>(m, p);
	}


	/**
	 * Make a set of uniform distributions on a buffer and return
	 */
	template<typename T>
	discrete_distr_set<T> make_uniform_discrete_distr_set(T m, size_t n, double *p)
	{
		double pv = 1.0 / ds;

		size_t N = (size_t)m * n;
		for (size_t i = 0; i < N; ++i)
		{
			p[i] = pv;
		}

		return discrete_distr_set<T>(m, n, p);
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
		TValue mv = ps[i];
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
	inline discrete_distr<T> make_discrete_distr_nrmexp(T m, const double *x, double *p)
	{
		normalize_exp((size_t)m, x, p);
		return discrete_distr<T>(m, p);
	}

	template<typename T>
	discrete_distr<T> make_discrete_distr_set_nrmexp(size_t m, size_t n, const double *x, double *p)
	{
		for (size_t i = 0; i < n; ++i)
		{
			normalize_exp((size_t)m, x + i * m, p + i * m);
		}
		return discrete_distr_set<T>(m, n, p);
	}

}


#endif 
