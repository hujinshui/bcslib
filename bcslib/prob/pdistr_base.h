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
#include <bcslib/array/array_calc.h>

#include <type_traits>
#include <cmath>
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
	 * 		- is_conj_prior;	// whether it can produce a posterior of the same class
	 * 							// given observed samples
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
	 * 	  If is_conj_prior is true:
	 *
	 *		- p.conj_update(sample);		// returns the posterior distribution given a sample (of weight 1)
	 *		- p.conj_update(sample, w);		// returns the posterior distribution given a weighted sample
	 * 	  	- p.conj_update(scoll);			// returns the posterior distribution given a sample collection
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
		static const bool is_conj_prior = false;
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



	// a light-weight wrapper of a (weighted) collection of samples
	template<typename T>
	class sample_collection
	{
	public:
		typedef T value_type;

		typedef caview1d<value_type> cview1d_type;
		typedef caview2d<value_type, column_major_t> cview2d_type;

	public:
		sample_collection(const cview1d_type& samples)
		: m_dim(1), m_n(samples.nelems()), m_samples(samples.pbase()), m_weights(BCS_NULL), m_shared_weight(1)
		{
		}

		sample_collection(const cview1d_type& samples, double w)
		: m_dim(1), m_n(samples.nelems()), m_samples(samples.pbase()), m_weights(BCS_NULL), m_shared_weight(w)
		{
		}

		sample_collection(const cview1d_type& samples, const cview1d_type& weights)
		: m_dim(1), m_n(samples.nelems()), m_samples(samples.pbase()), m_weights(weights.pbase()), m_shared_weight(0)
		{
			check_arg(m_n == weights.nelems(), "sample_collection: inconsistent sizes of samples and weights.");
		}

		sample_collection(const cview2d_type& samples)
		: m_dim(samples.nrows()), m_n(samples.ncolumns()), m_samples(samples.pbase()), m_weights(BCS_NULL), m_shared_weight(1)
		{
			check_arg(is_dense_view(samples), "sample_collection: the input samples must be a dense view.");
		}

		sample_collection(const cview2d_type& samples, double w)
		: m_dim(samples.nrows()), m_n(samples.ncolumns()), m_samples(samples.pbase()), m_weights(BCS_NULL), m_shared_weight(w)
		{
			check_arg(is_dense_view(samples), "sample_collection: the input samples must be a dense view.");
		}

		sample_collection(const cview2d_type& samples, const cview1d_type& weights)
		: m_dim(samples.nrows()), m_n(samples.ncolumns()), m_samples(samples.pbase()), m_weights(weights.pbase()), m_shared_weight(0)
		{
			check_arg(m_n == weights.nelems(), "sample_collection: inconsistent sizes of samples and weights.");
			check_arg(is_dense_view(samples), "sample_collection: the input samples must be a dense view.");
		}

	public:
		size_t dim() const
		{
			return m_dim;
		}

		size_t size() const
		{
			return m_n;
		}

		size_t nvalues() const
		{
			return m_n * m_dim;
		}

		const value_type *psamples() const
		{
			return m_samples;
		}

		const value_type *psample(index_t i) const
		{
			return m_samples + i * (index_t)m_dim;
		}

		bool has_shared_weight() const
		{
			return m_weights == BCS_NULL;
		}

		double shared_weight() const
		{
			return m_shared_weight;
		}

		const value_type *pweights() const
		{
			return m_weights;
		}

	public:
		cview1d_type samples_view1d() const
		{
			return get_aview1d(psamples(), nvalues());
		}

		cview2d_type samples_view2d() const
		{
			return get_aview2d_cm(psamples(), dim(), size());
		}

		caview1d<double> weights_view() const
		{
			return get_aview1d(pweights(), size());
		}

	private:
		size_t m_dim;
		size_t m_n;
		const value_type *m_samples;  	// do not have ownership
		const double *m_weights; 		// do not have ownership
		double m_shared_weight;

	}; // end class weighted_sample_set



	/******************************************************
	 *
	 *  Most widely used probability distributions
	 *
	 *  - bernoulli_ddistr: 	bool
	 *  - uniform_ddistr: 		int32_t
	 *  - uniform_distr:  		double
	 *  - normal_distr:			double
	 *  - exponential_distr:	double
	 *
	 ******************************************************/

	/**
	 *  Uniform discrete distribution over integers in [a, b]
	 */
	class uniform_ddistr
	{
	public:
		typedef int32_t value_type;

		class param_type
		{
			explicit param_type(value_type a = 0, value_type b = std::numeric_limits<value_type>::max())
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

		private:
			value_type m_a;
			value_type m_b;
		};

		static const size_t sample_dim = 1;

	public:

		explicit uniform_ddistr()
		: m_param()
		{
		}

		explicit uniform_ddistr(const param_type& param)
		: m_param(param)
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
			return m_param.
		}

	public:

		template<class Rstream>
		void draw(Rstream& rstream, value_type *sample)
		{
			rstream.randi32(sample);
		}

		template<class Rstream>
		void draw(Rstream& rstream, value_type *samples, size_t n)
		{
			rstream.randi32(samples, n);
		}


	private:
		param_type m_param;
		value_type m_span;

	}; // end class uniform_ddistr



}

#endif
