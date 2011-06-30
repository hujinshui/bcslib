/**
 * @file sample_collection.h
 *
 * The class to represent a collection of (weighted) samples
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_SAMPLE_COLLECTION_H_
#define BCSLIB_SAMPLE_COLLECTION_H_

#include <bcslib/array/array2d.h>

namespace bcs
{
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

}

#endif 
