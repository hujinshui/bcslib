/**
 * @file matrix_base.h
 *
 * Basic definitions for vectors and matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_BASE_H_
#define BCSLIB_MATRIX_BASE_H_

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_math.h>
#include <bcslib/base/arg_check.h>
#include <bcslib/base/mem_op.h>

#define MAT_TRAITS_DEFS(T) \
	typedef T value_type; \
	typedef T* pointer; \
	typedef const T* const_pointer; \
	typedef T& reference; \
	typedef const T& const_reference; \
	typedef size_t size_type; \
	typedef index_t difference_type; \
	typedef index_t index_type;



namespace bcs
{
	// forward declarations

	template<class Derived> struct matrix_traits;

	template<class Derived, typename T> class IMatrixBase;
	template<class Derived, typename T> class IDenseMatrixView;
	template<class Derived, typename T> class IDenseMatrixBlock;
	template<class Derived, typename T> class IDenseMatrix;


	/********************************************
	 *
	 *  CRTP Bases for matrices
	 *
	 ********************************************/

	/**
	 * The interfaces shared by everything that can evaluate into a matrix
	 */
	template<class Derived, typename T>
	class IMatrixBase
	{
	public:
		MAT_TRAITS_DEFS(T)
		BCS_CRTP_REF

		static const index_type RowDimension = matrix_traits<Derived>::RowDimension;
		static const index_type ColDimension = matrix_traits<Derived>::ColDimension;

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return (size_type)nelems();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return derived().is_empty();
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			derived().eval_to(dst);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to_block(IDenseMatrixBlock<DstDerived, T>& dst) const
		{
			derived().eval_to(dst);
		}

	}; // end class caview2d_base



	namespace detail
	{
		template<class Mat>
		BCS_ENSURE_INLINE
		inline void check_matrix_indices(const Mat& mat, index_t i, index_t j)
		{
#ifndef BCSLIB_NO_DEBUG
			return i >= 0 && i < mat.nrows() && j >= 0 && j < mat.ncolumns();
#endif
		}
	}


	/**
	 * The interfaces for matrix views
	 */
	template<class Derived, typename T>
	class IDenseMatrixView : public IMatrixBase<Derived, T>
	{
	public:
		MAT_TRAITS_DEFS(T)
		BCS_CRTP_REF

		typedef IMatrixBase<Derived, T> base_t;

		using base_t::RowDimension;
		using base_t::ColDimension;

		using base_t::nelems;
		using base_t::size;
		using base_t::nrows;
		using base_t::ncolumns;
		using base_t::is_empty;
		using base_t::eval_to;
		using base_t::eval_to_block;

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

	};


	/**
	 * The interfaces for matrix blocks
	 */
	template<class Derived, typename T>
	class IDenseMatrixBlock : public IDenseMatrixView<Derived, T>
	{
	public:
		MAT_TRAITS_DEFS(T)
		BCS_CRTP_REF

		typedef IDenseMatrixView<Derived, T> base_t;

		using base_t::RowDimension;
		using base_t::ColDimension;

		using base_t::nelems;
		using base_t::size;
		using base_t::nrows;
		using base_t::ncolumns;
		using base_t::is_empty;
		using base_t::eval_to;
		using base_t::eval_to_block;

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return derived().ptr_base();
		}

		BCS_ENSURE_INLINE pointer ptr_base()
		{
			return derived().ptr_base();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
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
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

	};



	/**
	 * The interfaces for matrices with continuous layout
	 */
	template<class Derived, typename T>
	class IDenseMatrix : public IDenseMatrixBlock<Derived, T>
	{
	public:
		MAT_TRAITS_DEFS(T)
		BCS_CRTP_REF

		typedef IDenseMatrixBlock<Derived, T> base_t;

		using base_t::RowDimension;
		using base_t::ColDimension;

		using base_t::nelems;
		using base_t::size;
		using base_t::nrows;
		using base_t::ncolumns;
		using base_t::is_empty;
		using base_t::eval_to;
		using base_t::eval_to_block;

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return derived().ptr_base();
		}

		BCS_ENSURE_INLINE pointer ptr_base()
		{
			return derived().ptr_base();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
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
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

	};


}


#endif
