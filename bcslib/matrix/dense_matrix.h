/**
 * @file matrix.h
 *
 * The dense matrix and vector class
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_DENSE_MATRIX_H_
#define BCSLIB_DENSE_MATRIX_H_

#include <bcslib/matrix/matrix_base.h>
#include <bcslib/base/block.h>

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
		// storage implementation
		template<typename T, int RowDim, int ColDim> struct MatrixInternal;

		template<typename T, int RowDim, int ColDim>
		struct MatrixInternal
		{

			T arr[RowDim * ColDim];

			BCS_ENSURE_INLINE
			MatrixInternal() { }

			BCS_ENSURE_INLINE
			MatrixInternal(index_t m, index_t n)
			{
				check_arg(m == RowDim && n == ColDim);
			}

			BCS_ENSURE_INLINE
			index_t nelems() const { return RowDim * ColDim; }

			BCS_ENSURE_INLINE
			index_t nrows() const { return RowDim; }

			BCS_ENSURE_INLINE
			index_t ncols() const { return ColDim; }

			BCS_ENSURE_INLINE
			const T *ptr() const { return arr; }

			BCS_ENSURE_INLINE
			T *ptr() { return arr; }

			BCS_ENSURE_INLINE
			const T& at(index_t i) const { return arr[i]; }

			BCS_ENSURE_INLINE
			T& at(index_t i) { return arr[i]; }

			void resize(index_t m, index_t n)
			{
				check_arg(m == RowDim && n == ColDim,
						"Attempted to change a fixed dimension.");
			}
		};

		template<typename T, int ColDim>
		struct MatrixInternal<T, DynamicDim, ColDim>
		{
			index_t n_rows;
			Block<T> blk;

			BCS_ENSURE_INLINE
			MatrixInternal() : n_rows(0) { }

			BCS_ENSURE_INLINE
			MatrixInternal(index_t m, index_t n)
			: n_rows(m), blk(m * check_forward(n, (index_t)ColDim))
			{
			}

			BCS_ENSURE_INLINE
			index_t nelems() const { return n_rows * ColDim; }

			BCS_ENSURE_INLINE
			index_t nrows() const { return n_rows; }

			BCS_ENSURE_INLINE
			index_t ncols() const { return ColDim; }

			BCS_ENSURE_INLINE
			const T *ptr() const { return blk.pbase(); }

			BCS_ENSURE_INLINE
			T *ptr() { return blk.pbase(); }

			BCS_ENSURE_INLINE
			const T& at(index_t i) const { return blk[i]; }

			BCS_ENSURE_INLINE
			T& at(index_t i) { return blk[i]; }

			void resize(index_t m, index_t n)
			{
				check_arg(n == ColDim,
						"Attempted to change a fixed dimension.");

				if (m != n_rows)
				{
					blk.resize(m * ColDim);
					n_rows = m;
				}
			}
		};


		template<typename T, int RowDim>
		struct MatrixInternal<T, RowDim, DynamicDim>
		{
			index_t n_cols;
			Block<T> blk;

			BCS_ENSURE_INLINE
			MatrixInternal() : n_cols(0) { }

			BCS_ENSURE_INLINE
			MatrixInternal(index_t m, index_t n)
			: n_cols(n), blk(check_forward(m, (index_t)RowDim) * n)
			{
			}

			BCS_ENSURE_INLINE
			index_t nelems() const { return RowDim * n_cols; }

			BCS_ENSURE_INLINE
			index_t nrows() const { return RowDim; }

			BCS_ENSURE_INLINE
			index_t ncols() const { return n_cols; }

			BCS_ENSURE_INLINE
			const T *ptr() const { return blk.pbase(); }

			BCS_ENSURE_INLINE
			T *ptr() { return blk.pbase(); }

			BCS_ENSURE_INLINE
			const T& at(index_t i) const { return blk[i]; }

			BCS_ENSURE_INLINE
			T& at(index_t i) { return blk[i]; }

			void resize(index_t m, index_t n)
			{
				check_arg(m == RowDim,
						"Attempted to change a fixed dimension.");

				if (n != n_cols)
				{
					blk.resize(RowDim * n);
					n_cols = n;
				}
			}
		};


		template<typename T>
		struct MatrixInternal<T, DynamicDim, DynamicDim>
		{
			index_t n_rows;
			index_t n_cols;
			Block<T> blk;

			BCS_ENSURE_INLINE
			MatrixInternal() : n_rows(0), n_cols(0) { }

			BCS_ENSURE_INLINE
			MatrixInternal(index_t m, index_t n)
			: n_rows(m), n_cols(n), blk(m * n)
			{
			}

			BCS_ENSURE_INLINE
			index_t nelems() const { return n_rows * n_cols; }

			BCS_ENSURE_INLINE
			index_t nrows() const { return n_rows; }

			BCS_ENSURE_INLINE
			index_t ncols() const { return n_cols; }

			BCS_ENSURE_INLINE
			const T *ptr() const { return blk.pbase(); }

			BCS_ENSURE_INLINE
			T *ptr() { return blk.pbase(); }

			BCS_ENSURE_INLINE
			const T& at(index_t i) const { return blk[i]; }

			BCS_ENSURE_INLINE
			T& at(index_t i) { return blk[i]; }

			void resize(index_t m, index_t n)
			{
				if (m * n != nelems())
				{
					blk.resize(m * n);
				}
				n_rows = m;
				n_cols = n;
			}
		};

	}


	/********************************************
	 *
	 *  Dense matrix
	 *
	 ********************************************/

	template<typename T, int RowDim, int ColDim>
	struct matrix_traits<DenseMatrix<T, RowDim, ColDim> >
	{
		MAT_TRAITS_DEFS(T)

		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		static const bool IsReadOnly = false;

		typedef const DenseMatrix<T, RowDim, ColDim>& eval_return_type;
	};


	template<typename T, int RowDim, int ColDim>
	class DenseMatrix
	: public detail::DenseMatrixFacet<T,
	  	  detail::MatrixInternal<T, RowDim, ColDim>,
	  	  DenseMatrix<T, RowDim, ColDim> >
	{

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(RowDim == DynamicDim || RowDim >= 1, "Invalid template argument RowDim");
		static_assert(ColDim == DynamicDim || ColDim >= 1, "Invalid template argument ColDim");
#endif

		typedef detail::MatrixInternal<T, RowDim, ColDim> internal_type;
		typedef detail::DenseMatrixFacet<T, internal_type, DenseMatrix> facet_type;
		BCS_DEFINE_MATRIX_FACET_INTERNAL

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<DenseMatrix<T, RowDim, ColDim>, T> base_type;
		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		typedef const DenseMatrix<T, RowDim, ColDim>& eval_return_type;

	public:

		BCS_ENSURE_INLINE
		explicit DenseMatrix()
		: facet_type(internal_type())
		{
		}

		BCS_ENSURE_INLINE
		DenseMatrix(index_type m, index_type n)
		: facet_type(internal_type(m, n))
		{
		}

		BCS_ENSURE_INLINE
		DenseMatrix(index_type m, index_type n, const T& v)
		: facet_type(internal_type(m, n))
		{
			this->fill(v);
		}

		BCS_ENSURE_INLINE
		DenseMatrix(index_type m, index_type n, const T *src)
		: facet_type(internal_type(m, n))
		{
			this->copy_from(src);
		}

		BCS_ENSURE_INLINE
		DenseMatrix(const DenseMatrix& other)
		: facet_type(other)
		{
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		DenseMatrix(const IMatrixBase<OtherDerived, T>& other)
		: facet_type(internal_type(other.nrows(), other.ncolumns()))
		{
			other.eval_to(*this);
		}

		BCS_ENSURE_INLINE
		const DenseMatrix& operator = (const DenseMatrix& other)
		{
			if (this != &other)
			{
				if (!is_same_size(*this, other))
				{
					this->resize(other.nrows(), other.ncolumns());
				}
				this->copy_from(other.ptr_base());
			}
			return *this;
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		const DenseMatrix& operator = (const IMatrixBase<OtherDerived, T>& other)
		{
			if (this != &other)
			{
				if (!is_same_size(*this, other))
				{
					this->resize(other.nrows(), other.ncolumns());
				}
				other.eval_to(*this);
			}
			return *this;
		}

		BCS_ENSURE_INLINE
		void resize(index_type m, index_type n)
		{
			internal().resize(m, n);
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return *this;
		}

	}; // end class DenseMatrix



	/********************************************
	 *
	 *  Dense Vectors
	 *
	 ********************************************/

	template<typename T, int RowDim>
	struct matrix_traits<DenseCol<T, RowDim> >
	{
		MAT_TRAITS_DEFS(T)

		static const index_type RowDimension = RowDim;
		static const index_type ColDimension = 1;
		static const bool IsReadOnly = false;

		typedef const DenseCol<T, RowDim>& eval_return_type;
	};


	template<typename T, int RowDim>
	class DenseCol : public DenseMatrix<T, RowDim, 1>
	{
	private:
		typedef DenseMatrix<T, RowDim, 1> base_mat_t;

	public:
		MAT_TRAITS_DEFS(T)

		static const index_t RowDimension = RowDim;
		static const index_t ColDimension = 1;
		typedef const DenseCol<T, RowDim>& eval_return_type;

	public:

		BCS_ENSURE_INLINE
		explicit DenseCol()
		: base_mat_t()
		{
		}

		BCS_ENSURE_INLINE
		explicit DenseCol(index_t m)
		: base_mat_t(m, 1)
		{
		}

		BCS_ENSURE_INLINE
		DenseCol(index_t m, const T& v)
		: base_mat_t(m, 1, v)
		{
		}

		BCS_ENSURE_INLINE
		DenseCol(index_t m, const T *src)
		: base_mat_t(m, 1, src)
		{
		}

		BCS_ENSURE_INLINE
		DenseCol(const base_mat_t& other)
		: base_mat_t(other)
		{
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		DenseCol(const IMatrixBase<OtherDerived, T>& other)
		: base_mat_t(other)
		{
		}

		BCS_ENSURE_INLINE
		void resize(index_t m)
		{
			base_mat_t::resize(m, 1);
		}

	}; // end class DenseCol


	template<typename T, int ColDim>
	struct matrix_traits<DenseRow<T, ColDim> >
	{
		MAT_TRAITS_DEFS(T)

		static const index_type RowDimension = 1;
		static const index_type ColDimension = ColDim;
		static const bool IsReadOnly = false;

		typedef const DenseRow<T, ColDim>& eval_return_type;
	};

	template<typename T, int ColDim>
	class DenseRow : public DenseMatrix<T, 1, ColDim>
	{
	private:
		typedef DenseMatrix<T, 1, ColDim> base_mat_t;

	public:
		MAT_TRAITS_DEFS(T)

		static const index_t RowDimension = 1;
		static const index_t ColDimension = ColDim;
		typedef const DenseRow<T, ColDim>& eval_return_type;

	public:

		BCS_ENSURE_INLINE
		explicit DenseRow()
		: base_mat_t()
		{
		}

		BCS_ENSURE_INLINE
		explicit DenseRow(index_t n)
		: base_mat_t(1, n)
		{
		}

		BCS_ENSURE_INLINE
		DenseRow(index_t n, const T& v)
		: base_mat_t(1, n, v)
		{
		}

		BCS_ENSURE_INLINE
		DenseRow(index_t n, const T *src)
		: base_mat_t(1, n, src)
		{
		}

		BCS_ENSURE_INLINE
		DenseRow(const base_mat_t& other)
		: base_mat_t(other)
		{
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		DenseRow(const IMatrixBase<OtherDerived, T>& other)
		: base_mat_t(other)
		{
		}

		BCS_ENSURE_INLINE
		void resize(index_t n)
		{
			base_mat_t::resize(1, n);
		}

	}; // end class DenseCol


	/********************************************
	 *
	 *  Typedefs
	 *
	 ********************************************/


	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat, DynamicDim, DynamicDim)
	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat22, 2, 2)
	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat23, 2, 3)
	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat32, 3, 2)
	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat33, 3, 3)

	BCS_MATRIX_TYPEDEFS1(DenseCol, col, DynamicDim)
	BCS_MATRIX_TYPEDEFS1(DenseCol, col2, 2)
	BCS_MATRIX_TYPEDEFS1(DenseCol, col3, 3)

	BCS_MATRIX_TYPEDEFS1(DenseRow, row, DynamicDim)
	BCS_MATRIX_TYPEDEFS1(DenseRow, row2, 2)
	BCS_MATRIX_TYPEDEFS1(DenseRow, row3, 3)


}

#endif /* MATRIX_H_ */
