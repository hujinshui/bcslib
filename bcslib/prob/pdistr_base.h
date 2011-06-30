/*
 * @file pdistr_base.h
 *
 * The base of probability distributions
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_PDISTR_BASE_H
#define BCSLIB_PDISTR_BASE_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/arg_check.h>
#include <bcslib/base/mathfun.h>

#include <cmath>
#include <type_traits>
#include <limits>

namespace bcs
{

	/**
	 * A prob. distribution class must satisfy the following requirement
	 * ------------------------------------------------------------
	 *
	 * Let P be the distribution class name, and p be an instance of P.
	 * Then:
	 *
	 * 1. P has the following typedefs:
	 *
	 * 		- value_type;	// the value type of each component, int32_t, double, float, etc
	 * 		- param_type;	// the type of the parameter (nil_type is no parameter is needed)
	 *
	 * 2. P has the following constants:
	 *
	 * 		- sample_dim:	// the dimension of the sample space (1 for scalar-valued space)
	 * 						// the type of sample_dim is size_t
	 *
	 * 3. bcs::is_pdistr<P>::value should be set to true. (By default, it is false)
	 *
	 * 4. A pdistr_traits<P> class is defined, which has the following:
	 *
	 * 		- value_type;
	 * 		- param_type;
	 * 		- sample_dim;
	 * 		- is_sampleable; 	// whether one can sample from the distribution
	 * 		- is_pdf_evaluable;	// whether one can evaluate pdf values at given samples
	 *
	 * 	  A generic template struct pdistr_traits is defined in this file, which simply
	 * 	  takes value_type, param_type, and sample_dim from the class P, and set
	 * 	  the boolean constants to false.
	 *
	 * 	  For each class, a specialized version of pdistr_traits can be provided to
	 * 	  override this default behavior for that class.
	 *
	 * 5. P is default-constructible, copy-constructible, and assignable
	 *
	 * 6. There is a constructor that takes a parameter of type param_type as argument
	 *
	 * 7. Each instance p of P can be used as follows:
	 *
	 * 		- p.dim();		// returns sample_dim
	 * 		- p.param();	// returns const reference to the parameter
	 *
	 * 	  If is_sampleable is true:
	 *
	 * 		- p.draw(rstream, dst);		// draws a sample and writes it to dst
	 * 									// - rstream: a random number stream
	 * 								 	// - dst:	destination address, of type value_type*
	 * 								 	// no return
	 *
	 * 		- p.draw(rstream, dst, n);	// draws n samples, no return
	 *
	 * 	  If is_pdf_evaluable is true:
	 *
	 * 	  	- p.pdf(sample);		// evaluates the pdf value at the input sample, and return (of type double)
	 * 	  							// sample is of type value_type*
	 *
	 * 	  	- p.pdfs(n, samples, out); 		// evaluates the pdf value at the input samples,
	 * 	  									// the evaluated values are written to out (of type double*)
	 * 	  									// no return
	 *
	 *		- p.logpdf(sample);		// evaluate logarithm of pdf value
	 *
	 *		- p.logpdfs(n, samples, out);	// evaluate the logarithm of pdf values
	 *
	 * 	8. Note: P can have its specific requirement on what interfaces should be supported
	 * 			 by rstream.
	 *
	 */

	template<class P>
	struct is_pdistr
	{
		static const bool value = false;
	};


	template<class P>
	struct pdistr_traits
	{
		typedef typename P::value_type value_type;
		typedef typename P::param_type param_type;

		static const size_t sample_dim = P::sample_dim;

		static const bool is_sampleable = false;
		static const bool is_pdf_evaluable = false;
	};


	template<class P>
	struct is_discrete_pdistr
	{
		static const bool value = is_pdistr<P>::value &&
				std::is_integral<typename pdistr_traits<P>::value_type>::value;
	};

	template<class P>
	struct is_continous_pdistr
	{
		static const bool value = is_pdistr<P>::value &&
				std::is_floating_point<typename pdistr_traits<P>::value_type>::value;
	};




	/******************************************************
	 *
	 *  Most widely used probability distributions
	 *
	 *  - bernoulli_distr: 		bool
	 *  - uniform_int_distr: 	int32_t
	 *  - uniform_distr:  		double
	 *  - normal_distr:			double
	 *  - exponential_distr:	double
	 *
	 ******************************************************/


	/**
	 *  Bernoulli distribution
	 */
	class bernoulli_distr
	{
	public:
		typedef bool value_type;
		typedef double param_type;
		static const size_t sample_dim = 1;

	public:

		explicit bernoulli_distr()
		: m_p(0.5), m_q(0.5)
		{
		}

		explicit bernoulli_distr(const double& p)
		: m_p(p), m_q(1.0 - m_p)
		{
		}

		size_t dim() const
		{
			return sample_dim;
		}

		const param_type& param() const
		{
			return m_p;
		}

		double p() const
		{
			return m_p;
		}

		double q() const
		{
			return m_q;
		}

		double mean() const
		{
			return m_p;
		}

		double var() const
		{
			return m_p * m_q;
		}

	public:

		template<class Rstream>
		void draw(Rstream& rstream, value_type *sample) const
		{
			return rstream.uniform_f64() < p();
		}

		template<class Rstream>
		void draw(Rstream& rstream, value_type *samples, size_t n) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				draw(rstream, samples+i);
			}
		}

		double pdf(const value_type *sample) const
		{
			return *sample ? m_p : m_q;
		}

		void pdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = pdf(samples + i);
			}
		}

		double logpdf(const value_type *sample) const
		{
			return std::log(pdf(sample));
		}

		void logpdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = logpdf(samples + i);
			}
		}

	private:
		double m_p;
		double m_q;

	}; // end class bernoulli_distr


	template<>
	struct is_pdistr<bernoulli_distr>
	{
		static const bool value = true;
	};

	template<>
	struct pdistr_traits<bernoulli_distr>
	{
		typedef bernoulli_distr::value_type value_type;
		typedef bernoulli_distr::param_type param_type;
		static const size_t sample_dim = 1;

		static const bool is_sampleable = true;
		static const bool is_pdf_evaluable = true;
	};



	/**
	 *  Uniform discrete distribution over integers in [a, b]
	 */
	class uniform_int_distr
	{
	public:
		typedef int32_t value_type;

		class param_type
		{
			explicit param_type(value_type a = 0, value_type b = std::numeric_limits<value_type>::max())
			: m_a(a), m_b(b)
			{
			}

			value_type a() const
			{
				return m_a;
			}

			value_type b() const
			{
				return m_b;
			}

			value_type span() const
			{
				return m_b - m_a + 1;
			}

			bool in_range(value_type v) const
			{
				return v >= m_a && v <= m_b;
			}

		private:
			value_type m_a;
			value_type m_b;
		};

		static const size_t sample_dim = 1;

	public:

		explicit uniform_int_distr()
		: m_param(), m_p(1.0 / m_param.span())
		{
		}

		explicit uniform_int_distr(const param_type& param)
		: m_param(param), m_p(1.0 / m_param.span())
		{
		}

		size_t dim() const
		{
			return sample_dim;
		}

		const param_type& param() const
		{
			return m_param;
		}

		value_type a() const
		{
			return m_param.a();
		}

		value_type b() const
		{
			return m_param.b();
		}

		value_type span() const
		{
			return m_param.span();
		}

		double mean() const
		{
			return ( double(a()) + double(b()) ) / 2;
		}

		double var() const
		{
			double s = (double)span();
			return (sqr(s) - 1.0) / 12;
		}

	public:

		template<class Rstream>
		void draw(Rstream& rstream, value_type *sample) const
		{
			*sample = rstream.uniform_i32(a(), b());
		}

		template<class Rstream>
		void draw(Rstream& rstream, value_type *samples, size_t n) const
		{
			rstream.uniform_i32_m(n, samples, a(), b());
		}

		double pdf(const value_type *sample) const
		{
			return m_param.in_range(*sample) ? m_p : 0.0;
		}

		void pdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = pdf(samples + i);
			}
		}

		double logpdf(const value_type *sample) const
		{
			return m_param.in_range(*sample) ? std::log(m_p) : - std::numeric_limits<double>::infinity();
		}

		void logpdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = logpdf(samples + i);
			}
		}

	private:
		param_type m_param;
		double m_p;

	}; // end class uniform_int_distr

	template<>
	struct is_pdistr<uniform_int_distr>
	{
		static const bool value = true;
	};

	template<>
	struct pdistr_traits<uniform_int_distr>
	{
		typedef uniform_int_distr::value_type value_type;
		typedef uniform_int_distr::param_type param_type;
		static const size_t sample_dim = 1;

		static const bool is_sampleable = true;
		static const bool is_pdf_evaluable = true;
	};


	/**
	 *  Uniform discrete distribution over real values in [a, b]
	 */
	class uniform_distr
	{
	public:
		typedef double value_type;

		class param_type
		{
			explicit param_type(value_type a = 0.0, value_type b = 1.0)
			: m_a(a), m_b(b), m_span(b - a + 1)
			{
			}

			value_type a() const
			{
				return m_a;
			}

			value_type b() const
			{
				return m_b;
			}

			value_type span() const
			{
				return m_b - m_a;
			}

			bool in_range(value_type v) const
			{
				return v >= m_a && v <= m_b;
			}

		private:
			value_type m_a;
			value_type m_b;
		};

		static const size_t sample_dim = 1;

	public:

		explicit uniform_distr()
		: m_param(), m_p(1.0 / m_param.span())
		{
		}

		explicit uniform_distr(const param_type& param)
		: m_param(param), m_p(1.0 / m_param.span())
		{
		}

		size_t dim() const
		{
			return sample_dim;
		}

		const param_type& param() const
		{
			return m_param;
		}

		value_type a() const
		{
			return m_param.a();
		}

		value_type b() const
		{
			return m_param.b();
		}

		value_type span() const
		{
			return m_param.span();
		}

		double mean() const
		{
			return (a() + b()) / 2;
		}

		double var() const
		{
			return sqr(span()) / 12;
		}

	public:

		template<class Rstream>
		void draw(Rstream& rstream, value_type *sample) const
		{
			*sample = rstream.uniform_f64(a(), b());
		}

		template<class Rstream>
		void draw(Rstream& rstream, value_type *samples, size_t n) const
		{
			rstream.uniform_f64_m(n, samples, a(), b());
		}

		double pdf(const value_type *sample) const
		{
			return m_param.in_range(*sample) ? m_p : 0.0;
		}

		void pdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = pdf(samples + i);
			}
		}

		double logpdf(const value_type *sample) const
		{
			return m_param.in_range(*sample) ? std::log(m_p) : - std::numeric_limits<double>::infinity();
		}

		void logpdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = logpdf(samples + i);
			}
		}

	private:
		param_type m_param;
		double m_p;

	}; // end class uniform_distr

	template<>
	struct is_pdistr<uniform_distr>
	{
		static const bool value = true;
	};

	template<>
	struct pdistr_traits<uniform_distr>
	{
		typedef uniform_distr::value_type value_type;
		typedef uniform_distr::param_type param_type;
		static const size_t sample_dim = 1;

		static const bool is_sampleable = true;
		static const bool is_pdf_evaluable = true;
	};



	/**
	 *  (Univariate) normal distribution
	 */
	class normal_distr
	{
	public:
		typedef double value_type;

		class param_type
		{
		public:
			explicit param_type(value_type mu = 0.0, value_type sigma = 1.0)
			: m_mu(mu), m_sigma(sigma)
			{
			}

			value_type mu() const
			{
				return m_mu;
			}

			value_type sigma() const
			{
				return m_sigma;
			}

			value_type sigma2() const
			{
				return sqr(m_sigma);
			}

		private:
			value_type m_mu;
			value_type m_sigma;
		};

		static const size_t sample_dim = 1;

	public:

		explicit normal_distr()
		: m_param()
		{
			_init();
		}

		explicit normal_distr(const param_type& param)
		: m_param(param)
		{
			_init();
		}

		size_t dim() const
		{
			return sample_dim;
		}

		const param_type& param() const
		{
			return m_param;
		}

		value_type mu() const
		{
			return m_param.mu();
		}

		value_type sigma() const
		{
			return m_param.sigma();
		}

		double mean() const
		{
			return mu();
		}

		double var() const
		{
			return m_param.sigma2();
		}

	public:

		template<class Rstream>
		void draw(Rstream& rstream, value_type *sample) const
		{
			*sample = rstream.normal_f64(mu(), sigma());
		}

		template<class Rstream>
		void draw(Rstream& rstream, value_type *samples, size_t n) const
		{
			rstream.normal_f64_m(n, samples, mu(), sigma());
		}

		double pdf(const value_type *sample) const
		{
			return std::exp(logpdf(sample));
		}

		void pdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = pdf(samples + i);
			}
		}

		double logpdf(const value_type *sample) const
		{
			value_type x = *sample;
			return (-0.5) * (sqr(x - mu()) * m_lambda + m_c);
		}

		void logpdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = logpdf(samples + i);
			}
		}

	private:

		void _init()
		{
			const double two_pi = 6.283185307179586476925287;
			double sig2 = m_param.sigma2();
			m_lambda = 1.0 / sig2;
			m_c = std::log(two_pi * sig2);
		}

		param_type m_param;
		double m_lambda; 	// = 1 / sigma^2
		double m_c;			// log(2 * pi * sigma^2)

	}; // end class normal_distr


	template<>
	struct is_pdistr<normal_distr>
	{
		static const bool value = true;
	};

	template<>
	struct pdistr_traits<normal_distr>
	{
		typedef normal_distr::value_type value_type;
		typedef normal_distr::param_type param_type;
		static const size_t sample_dim = 1;

		static const bool is_sampleable = true;
		static const bool is_pdf_evaluable = true;
	};


	/**
	 *  exponential distribution: parameterized by mu (mean)
	 */
	class exponential_distr
	{
	public:
		typedef double value_type;
		typedef double param_type;
		static const size_t sample_dim = 1;

	public:

		explicit exponential_distr()
		: m_mu(1.0), m_lambda(1.0)
		{
		}

		explicit exponential_distr(const double& mu)
		: m_mu(mu), m_lambda(1.0 / mu)
		{
		}

		size_t dim() const
		{
			return sample_dim;
		}

		const param_type& param() const
		{
			return m_mu;
		}

		value_type mu() const
		{
			return m_mu;
		}

		value_type lambda() const
		{
			return m_lambda;
		}

		double mean() const
		{
			return m_mu;
		}

		double var() const
		{
			return sqr(m_mu);
		}

	public:

		template<class Rstream>
		void draw(Rstream& rstream, value_type *sample) const
		{
			*sample = rstream.exponential_f64(mu());
		}

		template<class Rstream>
		void draw(Rstream& rstream, value_type *samples, size_t n) const
		{
			rstream.exponential_f64_m(n, samples, mu());
		}

		double pdf(const value_type *sample) const
		{
			value_type x = *sample;
			return m_lambda * std::exp(-m_lambda * x);
		}

		void pdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = pdf(samples + i);
			}
		}

		double logpdf(const value_type *sample) const
		{
			value_type x = *sample;
			return std::log(m_lambda) - m_lambda * x;
		}

		void logpdfs(size_t n, const value_type *samples, double *res) const
		{
			for (size_t i = 0; i < n; ++i)
			{
				res[i] = logpdf(samples + i);
			}
		}

	private:
		double m_mu;
		double m_lambda;

	}; // end class exponential_distr


	template<>
	struct is_pdistr<exponential_distr>
	{
		static const bool value = true;
	};

	template<>
	struct pdistr_traits<exponential_distr>
	{
		typedef exponential_distr::value_type value_type;
		typedef exponential_distr::param_type param_type;
		static const size_t sample_dim = 1;

		static const bool is_sampleable = true;
		static const bool is_pdf_evaluable = true;
	};

}

#endif
