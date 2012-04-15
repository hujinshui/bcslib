/**
 * @file ref_matrix.h
 *
 * The reference matrix/vector/block classes
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REF_MATRIX_H_
#define BCSLIB_REF_MATRIX_H_

#include <bcslib/matrix/matrix_base.h>
#include "bits/dense_matrix_facet.h"

namespace bcs
{


	/********************************************
	 *
	 *  Internal implementation
	 *
	 ********************************************/

	namespace detail
	{

		template<typename T, int RowDim, int ColDim>
		struct RefMatrixInternal
		{
			T* data;
			index_t n_rows;
			index_t n_cols;

			BCS_ENSURE_INLINE
			RefMatrixInternal()
			: data(BCS_NULL), n_rows(RowDim), n_cols(ColDim)
			{
			}

			BCS_ENSURE_INLINE
			RefMatrixInternal(T *refptr, index_t m, index_t n)
			: data(refptr), n_rows(m), n_cols(n)
			{
				check_input_dims(m, n);
			}

			BCS_ENSURE_INLINE
			void reset(T *refptr, index_t m, index_t n)
			{
				check_input_dims(m, n);

				data = refptr;
				n_rows = m;
				n_cols = n;
			}

			BCS_ENSURE_INLINE
			void move_forward(index_t step) { data += step; }

			BCS_ENSURE_INLINE
			void move_backward(index_t step) { data -= step; }

			BCS_ENSURE_INLINE
			index_t nelems() const { return n_rows * n_cols; }

			BCS_ENSURE_INLINE
			index_t nrows() const { return n_rows; }

			BCS_ENSURE_INLINE
			index_t ncols() const { return n_cols; }

			BCS_ENSURE_INLINE
			const T *ptr() const { return data; }

			BCS_ENSURE_INLINE
			T *ptr() { return data; }

			BCS_ENSURE_INLINE
			const T& at(index_t i) const { return data[i]; }

			BCS_ENSURE_INLINE
			T& at(index_t i) { return data[i]; }

		private:
			BCS_ENSURE_INLINE
			void check_input_dims(index_t m, index_t n)
			{
				if (RowDim >= 1) check_arg( m == RowDim );
				if (ColDim >= 1) check_arg( n == ColDim );
			}
		};


		template<typename T, int RowDim, int ColDim>
		struct CRefMatrixInternal
		{
			const T* data;
			index_t n_rows;
			index_t n_cols;

			BCS_ENSURE_INLINE
			CRefMatrixInternal()
			: data(BCS_NULL), n_rows(RowDim), n_cols(ColDim)
			{
			}

			BCS_ENSURE_INLINE
			CRefMatrixInternal(const T *refptr, index_t m, index_t n)
			: data(refptr), n_rows(m), n_cols(n)
			{
				check_input_dims(m, n);
			}

			BCS_ENSURE_INLINE
			void reset(const T *refptr, index_t m, index_t n)
			{
				check_input_dims(m, n);

				data = refptr;
				n_rows = m;
				n_cols = n;
			}

			BCS_ENSURE_INLINE
			void move_forward(index_t step) { data += step; }

			BCS_ENSURE_INLINE
			void move_backward(index_t step) { data -= step; }

			BCS_ENSURE_INLINE
			index_t nelems() const { return n_rows * n_cols; }

			BCS_ENSURE_INLINE
			index_t nrows() const { return n_rows; }

			BCS_ENSURE_INLINE
			index_t ncols() const { return n_cols; }

			BCS_ENSURE_INLINE
			const T *ptr() const { return data; }

			BCS_ENSURE_INLINE
			const T *ptr() { return data; }

			BCS_ENSURE_INLINE
			const T& at(index_t i) const { return data[i]; }

			BCS_ENSURE_INLINE
			const T& at(index_t i) { return data[i]; }

		private:
			BCS_ENSURE_INLINE
			void check_input_dims(index_t m, index_t n)
			{
				if (RowDim >= 1) check_arg( m == RowDim );
				if (ColDim >= 1) check_arg( n == ColDim );
			}
		};

	}





	/********************************************
	 *
	 *  RefMatrix
	 *
	 ********************************************/

	template<typename T, int RowDim, int ColDim>
	struct matrix_traits<RefMatrix<T, RowDim, ColDim> >
	{
		MAT_TRAITS_DEFS(T)

		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		static const bool IsReadOnly = false;

		typedef const RefMatrix<T, RowDim, ColDim>& eval_return_type;
	};


	template<typename T, int RowDim, int ColDim>
	class RefMatrix
	: public detail::DenseMatrixFacet<T,
	  	  detail::RefMatrixInternal<T, RowDim, ColDim>,
	  	  RefMatrix<T, RowDim, ColDim> >
	{

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(RowDim == DynamicDim || RowDim >= 1, "Invalid template argument RowDim");
		static_assert(ColDim == DynamicDim || ColDim >= 1, "Invalid template argument ColDim");
#endif

		typedef detail::RefMatrixInternal<T, RowDim, ColDim> internal_type;
		typedef detail::DenseMatrixFacet<T, internal_type, RefMatrix > facet_type;
		BCS_DEFINE_MATRIX_FACET_INTERNAL

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<RefMatrix<T, RowDim, ColDim>, T> base_type;
		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		typedef const RefMatrix& eval_return_type;

	public:
		BCS_ENSURE_INLINE
		RefMatrix()
		: facet_type(internal_type())
		{
		}

		BCS_ENSURE_INLINE
		RefMatrix(T *refptr, index_t m, index_t n)
		: facet_type(internal_type(refptr, m, n))
		{
		}


		BCS_ENSURE_INLINE
		void reset(T *refptr, index_t m, index_t n)
		{
			internal().reset(refptr, m, n);
		}

		BCS_ENSURE_INLINE
		void move_forward(index_t step)
		{
			internal().move_forward(step);
		}

		BCS_ENSURE_INLINE
		void move_backward(index_t step)
		{
			internal().move_backward(step);
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return *this;
		}

	}; // end class RefMatrix



	template<typename T, int RowDim, int ColDim>
	struct matrix_traits<CRefMatrix<T, RowDim, ColDim> >
	{
		MAT_TRAITS_DEFS(T)

		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		static const bool IsReadOnly = true;

		typedef const CRefMatrix<T, RowDim, ColDim>& eval_return_type;
	};


	template<typename T, int RowDim, int ColDim>
	class CRefMatrix
	: public detail::DenseMatrixFacet<T,
	  	  detail::CRefMatrixInternal<T, RowDim, ColDim>,
	  	  CRefMatrix<T, RowDim, ColDim> >
	{

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(RowDim == DynamicDim || RowDim >= 1, "Invalid template argument RowDim");
		static_assert(ColDim == DynamicDim || ColDim >= 1, "Invalid template argument ColDim");
#endif

		typedef detail::CRefMatrixInternal<T, RowDim, ColDim> internal_type;
		typedef detail::DenseMatrixFacet<T, internal_type, CRefMatrix > facet_type;
		BCS_DEFINE_MATRIX_FACET_INTERNAL

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<RefMatrix<T, RowDim, ColDim>, T> base_type;
		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		typedef const CRefMatrix& eval_return_type;

	public:
		BCS_ENSURE_INLINE
		CRefMatrix()
		: facet_type(internal_type())
		{
		}

		BCS_ENSURE_INLINE
		CRefMatrix(const T *refptr, index_t m, index_t n)
		: facet_type(internal_type(refptr, m, n))
		{
		}

		BCS_ENSURE_INLINE
		CRefMatrix(const RefMatrix<T, RowDim, ColDim>& other)
		: facet_type(internal_type(other.ptr_base(), other.nrows(), other.ncolumns()))
		{

		}

		BCS_ENSURE_INLINE
		void reset(const T *refptr, index_t m, index_t n)
		{
			internal().reset(refptr, m, n);
		}

		BCS_ENSURE_INLINE
		void move_forward(index_t step)
		{
			internal().move_forward(step);
		}

		BCS_ENSURE_INLINE
		void move_backward(index_t step)
		{
			internal().move_backward(step);
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return *this;
		}

	}; // end class CRefMatrix


}


#endif /* REF_MATRIX_H_ */






