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
#include <bcslib/array/dynamic_sparse_vector.h>

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

		byte *ws_bytes = (byte*)(
				wsbuf.request_buffer(len * sizeof(indexed_entry<double>)));
		indexed_entry<double>* ws = reinterpret_cast<indexed_entry<double>*>(ws_bytes);

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

		// finalize

		wsbuf.return_buffer(ws_bytes);
	}

	template<typename T>
	void discrete_sampler<T>::_init_internal(size_t K, const double *p)
	{
		// sort p in descending order with indices

		block<indexed_entry<double, T> > bb(K);
		copy_elements_attach_indices(p, bb.pbase(), K);
		std::sort(bb.pbase(), bb.pend(), std::greater<indexed_entry<double, T> >());

		// generate m_I, m_F, and calculate avg_searchlen

		block<T> *pbI = new block<T>(K);
		block<double> *pbF = new block<double>(K);

		T *I = pbI->pbase();
		double *F = pbF->pbase();

		double lastF = 0;
		m_avg_searchlen = 0;

		int Ki = (int)K;
		for (int k = 0; k < Ki; ++k)
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





	class variable_discrete_distr
	{
	public:
		typedef discrete_distribution_t distribution_category;
		typedef scalar_sample_t sample_category;

		typedef dynamic_ordered_spvec<double> internal_vector_t;
		typedef internal_vector_t::size_type size_type;
		typedef internal_vector_t::index_type value_type;

	public:

		variable_discrete_distr(size_type K, double w, tbuffer& buf, double thres = 0)
		: m_vec(K, thres), m_total_weight(0.0), m_adjust_counter(0)
		{
			byte *wspace = (byte*)buf.request_buffer( (sizeof(double) + sizeof(value_type)) * (size_t)K );

			value_type *inds = (value_type*)wspace;
			double *weights = (double*)(wspace + sizeof(value_type) * (size_t)K);

			for (size_type k = 0; k < K; ++k)
			{
				inds[k] = (value_type)k;
				weights[k] = w;
			}

			_init(K, inds, weights);

			buf.return_buffer(wspace);
		}

		variable_discrete_distr(size_type K, const double *weights, tbuffer& buf, double thres = 0)
		: m_vec(K, thres), m_total_weight(0.0), m_adjust_counter(0)
		{
			byte *wspace = (byte*)buf.request_buffer(sizeof(value_type) * (size_t)K);

			value_type *inds = (value_type*)wspace;

			for (size_type k = 0; k < K; ++k)
			{
				inds[k] = (value_type)k;
			}

			_init(K, inds, weights);

			buf.return_buffer(wspace);
		}

		template<typename IndexIter, typename WeightIter>
		variable_discrete_distr(size_type K, size_type len,
				IndexIter inds, WeightIter weights, double thres = 0)
		: m_vec(K, thres), m_total_weight(0.0), m_adjust_counter(0)
		{
			_init(len, inds, weights);
		}

		size_t dim() const
		{
			return 1;
		}

		value_type K() const
		{
			return m_vec.dim0();
		}

		size_type nactives() const
		{
			return m_vec.nactives();
		}

		value_type active_value(size_type i) const
		{
			return m_vec.active_index(i);
		}

		double active_weight(size_type i) const
		{
			return m_vec.active_value(i);
		}

		double total_weight() const
		{
			return m_total_weight;
		}

		double weight(const value_type& k) const
		{
			size_type i = m_vec.position_of_index(k);
			if (i < nactives())
			{
				return m_vec.active_value(i);
			}
			else
			{
				return 0.0;
			}
		}

		double p(const value_type& k) const
		{
			size_type i = m_vec.position_of_index(k);
			if (i < nactives())
			{
				return m_vec.active_value(i) / total_weight();
			}
			else
			{
				return 0.0;
			}
		}

		double average_search_length() const
		{
			double s = 0;
			size_type na = nactives();
			for (size_type i = 0; i < na; ++i)
			{
				s += m_vec.active_value(i) * double(i+1);
			}
			return s / total_weight();
		}

		template<typename RStream>
		value_type direct_sample(RStream& rstream)
		{
			double v = rstream.randf64() * total_weight();
			size_type i = direct_sample_from_discrete_pmf<size_type>(
					nactives(), m_vec.active_values(), v);

			if (i >= nactives())
			{
				i = nactives() - 1;
			}

			return m_vec.active_index(i);
		}

		bool set_weight(value_type k, double w)
		{
			size_type i = m_vec.position_of_index(k);
			if (i < m_vec.nactives())
			{
				double w0 = m_vec.active_value(i);

				if (w == w0) return false;

				m_vec.update_value_at_position(i, w);
				m_total_weight += (w - w0);
			}
			else
			{
				if (!m_vec.can_be_active_value(w)) return false;

				m_vec.add_new_pair(k, w);
				m_total_weight += w;
			}

			++ m_adjust_counter;
			if (m_adjust_counter == m_maximum_adjust_times)
			{
				recalc_total_weight();
			}

			return true;
		}


	private:

		template<typename IndexIter, typename WeightIter>
		void _init(size_type len, IndexIter inds, WeightIter weights)
		{
			m_vec.initialize(len, inds, weights);
			recalc_total_weight();
		}

		void recalc_total_weight()
		{
			size_t na = m_vec.nactives();
			double tw = 0;
			for (size_t i = 0; i < na; ++i)
			{
				tw += m_vec.active_value(i);
			}

			m_total_weight = tw;
			m_adjust_counter = 0;
		}


	private:
		internal_vector_t m_vec;

		double m_total_weight;
		uint32_t m_adjust_counter;

		static const uint32_t m_maximum_adjust_times = 100;

	}; // end class variable_discrete_distr




}


#endif 
