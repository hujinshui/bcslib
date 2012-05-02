/**
 * @file matrix_concepts.h
 *
 * The basic concepts for matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_CONCEPTS_H_
#define BCSLIB_MATRIX_CONCEPTS_H_

#include <bcslib/matrix/matrix_ctf.h>
#include <bcslib/utils/arg_check.h>


#define BCS_MAT_TRAITS_DEFS_FOR_BASE(D, T) \
	typedef T value_type; \
	typedef typename mat_access<D>::const_pointer const_pointer; \
	typedef typename mat_access<D>::const_reference const_reference; \
	typedef typename mat_access<D>::pointer pointer; \
	typedef typename mat_access<D>::reference reference; \
	typedef size_t size_type; \
	typedef index_t difference_type; \
	typedef index_t index_type;

#define BCS_MAT_TRAITS_CDEFS(T) \
	typedef T value_type; \
	typedef const value_type* const_pointer; \
	typedef const value_type& const_reference; \
	typedef size_t size_type; \
	typedef index_t difference_type; \
	typedef index_t index_type;

#define BCS_MAT_TRAITS_DEFS(T) \
	typedef T value_type; \
	typedef const value_type* const_pointer; \
	typedef const value_type& const_reference; \
	typedef value_type* pointer; \
	typedef value_type& reference; \
	typedef size_t size_type; \
	typedef index_t difference_type; \
	typedef index_t index_type;


namespace bcs
{

	/********************************************
	 *
	 *  Concepts
	 *
	 *  Each concept is associated with a
	 *  class as a static polymorphism base
	 *
	 ********************************************/

	template<class Derived, typename T>
	class IMatrixXpr
	{
	public:
		BCS_MAT_TRAITS_DEFS_FOR_BASE(Derived, T)
		BCS_CRTP_REF

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

	}; // end class IMatrixBase


	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	inline bool is_subscripts_in_range(const IMatrixXpr<Mat, T>& X, index_t i, index_t j)
	{
		return i >= 0 && i < X.nrows() && j >= 0 && j < X.ncolumns();
	}

	template<class Mat, typename T>
	BCS_ENSURE_INLINE
	inline void check_subscripts_in_range(const IMatrixXpr<Mat, T>& X, index_t i, index_t j)
	{
#ifndef BCSLIB_NO_DEBUG
		check_range(is_subscripts_in_range(X, i, j),
				"Attempted to access element with subscripts out of valid range.");
#endif
	}



	/**
	 * The interfaces for matrix views
	 */
	template<class Derived, typename T>
	class IMatrixView : public IMatrixXpr<Derived, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS_FOR_BASE(Derived, T)
		BCS_CRTP_REF

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE value_type elem(index_type i, index_type j) const
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE value_type operator() (index_type i, index_type j) const
		{
			check_subscripts_in_range(*this, i, j);
			return elem(i, j);
		}

	}; // end class IDenseMatrixView


	/**
	 * The interfaces for matrix blocks
	 */
	template<class Derived, typename T>
	class IDenseMatrix : public IMatrixView<Derived, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS_FOR_BASE(Derived, T)
		BCS_CRTP_REF

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE const_pointer ptr_data() const
		{
			return derived().ptr_data();
		}

		BCS_ENSURE_INLINE pointer ptr_data()
		{
			return derived().ptr_data();
		}

		BCS_ENSURE_INLINE index_t lead_dim() const
		{
			return derived().lead_dim();
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			check_subscripts_in_range(derived(), i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			check_subscripts_in_range(derived(), i, j);
			return elem(i, j);
		}


	public:
		BCS_ENSURE_INLINE
		typename subviews<Derived>::const_row_type
		row(index_type i) const
		{
			return subviews<Derived>::get_row(derived(), i);
		}

		BCS_ENSURE_INLINE
		typename subviews<Derived>::row_type
		row(index_type i)
		{
			return subviews<Derived>::get_row(derived(), i);
		}

		BCS_ENSURE_INLINE
		typename subviews<Derived>::const_column_type
		column(index_type j) const
		{
			return subviews<Derived>::get_column(derived(), j);
		}

		BCS_ENSURE_INLINE
		typename subviews<Derived>::column_type
		column(index_type j)
		{
			return subviews<Derived>::get_column(derived(), j);
		}

		BCS_ENSURE_INLINE
		typename subviews<Derived>::const_multirow_type
		rows(const range& rgn) const
		{
			return subviews<Derived>::get_multirow(derived(),
					rgn.begin_index(), rgn.num());
		}

		BCS_ENSURE_INLINE
		typename subviews<Derived>::multirow_type
		rows(const range& rgn)
		{
			return subviews<Derived>::get_multirow(derived(),
					rgn.begin_index(), rgn.num());
		}

		BCS_ENSURE_INLINE
		typename subviews<Derived>::const_multicolumn_type
		columns(const range& rgn) const
		{
			return subviews<Derived>::get_multicolumn(derived(),
					rgn.begin_index(), rgn.num());
		}

		BCS_ENSURE_INLINE
		typename subviews<Derived>::multicolumn_type
		columns(const range& rgn)
		{
			return subviews<Derived>::get_multicolumn(derived(),
					rgn.begin_index(), rgn.num());
		}


	}; // end class IDenseMatrixBlock


}

#endif /* MATRIX_CONCEPTS_H_ */
