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

#include <bcslib/matrix/matrix_fwd.h>
#include <bcslib/base/block.h>

#include "bits/matrix_helpers.h"

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
			bool is_empty() const { return false; }

			BCS_ENSURE_INLINE
			bool is_vector() const { return RowDim == 1 || ColDim == 1; }

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
			block<T> blk;

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
			bool is_empty() const { return n_rows == 0; }

			BCS_ENSURE_INLINE
			bool is_vector() const { return ColDim == 1 || n_rows == 1; }

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
			block<T> blk;

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
			bool is_empty() const { return n_cols == 0; }

			BCS_ENSURE_INLINE
			bool is_vector() const { return RowDim == 1 || n_cols == 1; }

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
			block<T> blk;

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
			bool is_empty() const { return n_rows == 0 || n_cols == 0; }

			BCS_ENSURE_INLINE
			bool is_vector() const { return n_rows == 1 || n_rows == 1; }

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
	};


	template<typename T, int RowDim, int ColDim>
	class DenseMatrix : public IDenseMatrix<DenseMatrix<T, RowDim, ColDim>, T>
	{

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(RowDim == DynamicDim || RowDim >= 1, "Invalid template argument RowDim");
		static_assert(ColDim == DynamicDim || ColDim >= 1, "Invalid template argument ColDim");
#endif

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<DenseMatrix<T, RowDim, ColDim>, T> base_type;
		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;

	public:

		BCS_ENSURE_INLINE
		explicit DenseMatrix()
		{
		}

		BCS_ENSURE_INLINE
		DenseMatrix(index_type m, index_type n)
		: m_internal(m, n)
		{
		}

		BCS_ENSURE_INLINE
		DenseMatrix(index_type m, index_type n, const T& v)
		: m_internal(m, n)
		{
			this->fill(v);
		}

		BCS_ENSURE_INLINE
		DenseMatrix(index_type m, index_type n, const T *src)
		: m_internal(m, n)
		{
			this->copy_from(src);
		}

		BCS_ENSURE_INLINE
		DenseMatrix(const DenseMatrix& other)
		: m_internal(other.m_internal)
		{
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		DenseMatrix(const IMatrixBase<OtherDerived, T>& other)
		: m_internal(other.nrows(), other.ncolumns())
		{
			evaluate_to(other.derived(), *this);
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
				assign(other);
			}
			return *this;
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		const DenseMatrix& operator = (const IMatrixBase<OtherDerived, T>& other)
		{
			if (!is_same_size(*this, other))
			{
				this->resize(other.nrows(), other.ncolumns());
			}
			assign(other);
			return *this;
		}

		BCS_ENSURE_INLINE
		void assign(const DenseMatrix& other)
		{
			copy_from(other.ptr_base());
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		void assign(const IMatrixBase<OtherDerived, T>& other)
		{
			evaluate_to(other.derived(), *this);
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			if (dst.ptr_base() != this->ptr_base()) bcs::copy(*this, dst);
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_internal.nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return (size_type)nelems();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_internal.ncols();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return m_internal.is_empty();
		}

		BCS_ENSURE_INLINE bool is_vector() const
		{
			return m_internal.is_vector();
		}

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return m_internal.ptr();
		}

		BCS_ENSURE_INLINE pointer ptr_base()
		{
			return m_internal.ptr();
		}

		BCS_ENSURE_INLINE const_pointer col_ptr(index_type j) const
		{
			return ptr_base() + j * lead_dim();
		}

		BCS_ENSURE_INLINE pointer col_ptr(index_type j)
		{
			return ptr_base() + j * lead_dim();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return detail::calc_offset<RowDim, ColDim>(lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_internal.at(offset(i, j));
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return m_internal.at(offset(i, j));
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx) const
		{
			return m_internal.at(idx);
		}

		BCS_ENSURE_INLINE reference operator[] (index_type idx)
		{
			return m_internal.at(idx);
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

	public:

		BCS_ENSURE_INLINE void zero()
		{
			bcs::zero(*this);
		}

		BCS_ENSURE_INLINE void fill(const_reference v)
		{
			bcs::fill(*this, v);
		}

		BCS_ENSURE_INLINE void copy_from(const_pointer src)
		{
			bcs::copy_from_mem(*this, src);
		}

		BCS_ENSURE_INLINE
		void resize(index_type m, index_type n)
		{
			m_internal.resize(m, n);
		}

	private:
		detail::MatrixInternal<T, RowDim, ColDim> m_internal;

	}; // end class DenseMatrix






	/********************************************
	 *
	 *  Dense Vectors
	 *
	 ********************************************/

	template<typename T, int RowDim>
	class DenseCol : public DenseMatrix<T, RowDim, 1>
	{
	private:
		typedef DenseMatrix<T, RowDim, 1> base_mat_t;

	public:
		MAT_TRAITS_DEFS(T)

		static const index_t RowDimension = RowDim;
		static const index_t ColDimension = 1;

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
	class DenseRow : public DenseMatrix<T, 1, ColDim>
	{
	private:
		typedef DenseMatrix<T, 1, ColDim> base_mat_t;

	public:
		MAT_TRAITS_DEFS(T)

		static const index_t RowDimension = 1;
		static const index_t ColDimension = ColDim;

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
