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
#include <cmath>

namespace bcs
{

	/**
	 * The class to represent a generic discrete distribution
	 */
	class discrete_distr : public discrete_distribution_t
	{
	public:
		discrete_distr(size_t ds, const double *p) : m_ds(ds), m_p(p)
		{
		}

		size_t dim() const
		{
			return 0;
		}

		size_t domain_size() const
		{
			return m_ds;
		}

		const double *pbase() const
		{
			return m_p;
		}

		double pdf(int x) const
		{
			return m_p[x];
		}

		double logpdf(int x) const
		{
			return std::log(pdf(x));
		}

	private:
		size_t m_ds;
		const double *m_p;


	}; // end class discrete_distr


	/**
	 * Make a uniform discrete distribution on a buffer and return
	 */
	inline discrete_distr make_uniform_discrete_distr(size_t ds, double *p)
	{
		double pv = 1.0 / ds;
		for (size_t i = 0; i < ds; ++i)
		{
			p[i] = pv;
		}
		return discrete_distr(ds, p);
	}


	class discrete_distr_set : public pdistribution_set_t<discrete_distr>
	{
	public:
		typedef discrete_distr distribution_type;

	public:
		discrete_distr_set(size_t ds, size_t n, const double *p)
		: m_ds(ds), m_n(n), m_p(p)
		{
		}

		size_t dim() const
		{
			return 0;
		}

		size_t domain_size() const
		{
			return m_ds;
		}

		size_t ndistrs() const
		{
			return m_n;
		}

		discrete_distr distribution(size_t i) const
		{
			return discrete_distr(m_ds, m_p + m_ds * i);
		}

	private:
		size_t m_ds;
		size_t m_n;
		const double *m_p;

	}; // end class discrete_distr_set


	/**
	 * Make a set of uniform distributions on a buffer and return
	 */
	inline discrete_distr_set make_uniform_discrete_distr_set(size_t ds, size_t n, double *p)
	{
		double pv = 1.0 / ds;

		size_t N = ds * n;
		for (size_t i = 0; i < N; ++i)
		{
			p[i] = pv;
		}

		return discrete_distr_set(ds, n, p);
	}



	/**
	 * Compute a normalized exponent
	 *
	 * Note: this operation can be done inplace, meaning that ps == pd
	 * is allowed.
	 *
	 */
	inline void normalize_exp(size_t n, const double *ps, double *pd)
	{
		double mv = ps[i];
		for (size_t i = 1; i < n; ++i)
		{
			if (ps[i] > mv) mv = ps[i];
		}

		double s = 0;
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


	inline discrete_distr make_discrete_distr_nrmexp(size_t ds, const double *x, double *p)
	{
		normalize_exp(ds, x, p);
		return discrete_distr(ds, p);
	}

	inline discrete_distr make_discrete_distr_set_nrmexp(size_t ds, size_t n, const double *x, double *p)
	{
		for (size_t i = 0; i < n; ++i)
		{
			normalize_exp(ds, x + i * ds, p + i * ds);
		}
		return discrete_distr_set(ds, n, p);
	}

}


#endif 
