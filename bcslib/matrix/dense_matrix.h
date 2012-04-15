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

namespace bcs
{
	// forward declaration

	const int DynamicDim = -1;
	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class DenseMatrix;
	template<typename T, int RowDim=DynamicDim> class DenseCol;
	template<typename T, int ColDim=DynamicDim> class DenseRow;


	/********************************************
	 *
	 *  Helper
	 *
	 ********************************************/

	namespace detail
	{
		// offset

		template<bool SingleRow, bool SingleColumn> struct offset_helper;

		template<> struct offset_helper<false, false>
		{
			BCS_ENSURE_INLINE
			static inline index_t calc(index_t lead_dim, index_t i, index_t j)
			{
				return i + lead_dim * j;
			}
		};

		template<> struct offset_helper<false, true>
		{
			BCS_ENSURE_INLINE
			static inline index_t calc(index_t lead_dim, index_t i, index_t)
			{
				return i;
			}
		};

		template<> struct offset_helper<true, false>
		{
			BCS_ENSURE_INLINE
			static inline index_t calc(index_t lead_dim, index_t, index_t j)
			{
				return j;
			}
		};


		template<> struct offset_helper<true, true>
		{
			BCS_ENSURE_INLINE
			static inline index_t calc(index_t lead_dim, index_t, index_t)
			{
				return 0;
			}
		};


		template<int RowDim, int ColDim>
		BCS_ENSURE_INLINE
		inline index_t calc_offset(index_t lead_dim, index_t i, index_t j)
		{
			return offset_helper<RowDim == 1, ColDim == 1>::calc(lead_dim, i, j);
		}


		// storage implementation

		template<typename T, int RowDim, int ColDim>
		struct matrix_impl
		{
			T arr[RowDim * ColDim];

			BCS_ENSURE_INLINE
			matrix_impl() { }

			BCS_ENSURE_INLINE
			matrix_impl(index_t m, index_t n)
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
		struct matrix_impl<T, DynamicDim, ColDim>
		{
			index_t n_rows;
			block<T> blk;

			BCS_ENSURE_INLINE
			matrix_impl() { }

			BCS_ENSURE_INLINE
			matrix_impl(index_t m, index_t n)
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
		struct matrix_impl<T, RowDim, DynamicDim>
		{
			index_t n_cols;
			block<T> blk;

			BCS_ENSURE_INLINE
			matrix_impl() { }

			BCS_ENSURE_INLINE
			matrix_impl(index_t m, index_t n)
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
		struct matrix_impl<T, DynamicDim, DynamicDim>
		{
			index_t n_rows;
			index_t n_cols;
			block<T> blk;

			BCS_ENSURE_INLINE
			matrix_impl() { }

			BCS_ENSURE_INLINE
			matrix_impl(index_t m, index_t n)
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

		static const index_type RowDimension = RowDim;
		static const index_type ColDimension = ColDim;

		typedef const DenseMatrix<T, RowDim, ColDim>& eval_return_type;
	};


	template<typename T, int RowDim, int ColDim>
	class DenseMatrix
	: public IDenseMatrix<DenseMatrix<T, RowDim, ColDim>, T>
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(RowDim == DynamicDim || RowDim >= 1, "Invalid template argument RowDim");
		static_assert(ColDim == DynamicDim || ColDim >= 1, "Invalid template argument ColDim");
#endif

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<DenseMatrix<T, RowDim, ColDim>, T> base_type;
		static const index_t RowDimension = RowDim;
		static const index_t ColDimension = ColDim;
		typedef const DenseMatrix<T, RowDim, ColDim>& eval_return_type;

	public:

		BCS_ENSURE_INLINE
		explicit DenseMatrix()
		: m_impl()
		{
		}

		BCS_ENSURE_INLINE
		DenseMatrix(index_type m, index_type n)
		: m_impl(m, n)
		{
		}

		BCS_ENSURE_INLINE
		DenseMatrix(index_type m, index_type n, const T& v)
		: m_impl(m, n)
		{
			fill(v);
		}

		BCS_ENSURE_INLINE
		DenseMatrix(index_type m, index_type n, const T *src)
		: m_impl(m, n)
		{
			copy_from(src);
		}

		BCS_ENSURE_INLINE
		DenseMatrix(const DenseMatrix& other)
		: m_impl(other.m_impl)
		{
		}

		template<class OtherDerived>
		BCS_ENSURE_INLINE
		DenseMatrix(const IMatrixBase<OtherDerived, T>& other)
		: m_impl(other.nrows(), other.ncolumns())
		{
			other.eval_to(*this);
		}

		BCS_ENSURE_INLINE
		void resize(index_type m, index_type n)
		{
			m_impl.resize(m, n);
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_impl.nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return (size_type)nelems();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_impl.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_impl.ncols();
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return nrows() == 0 || ncolumns() == 0;
		}

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return m_impl.ptr();
		}

		BCS_ENSURE_INLINE pointer ptr_base()
		{
			return m_impl.ptr();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_impl.nrows();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return detail::calc_offset<RowDim, ColDim>(lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_impl.at(offset(i, j));
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return m_impl.at(offset(i, j));
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx) const
		{
			return m_impl.at(idx);
		}

		BCS_ENSURE_INLINE reference operator[] (index_type idx)
		{
			return m_impl.at(idx);
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

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to(IDenseMatrix<DstDerived, T>& dst) const
		{
			if (dst.ptr_base() != this->ptr_base())
			{
				copy_elems(this->ptr_base(), dst.ptr_base(), size());
			}
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to_block(IDenseMatrixBlock<DstDerived, T>& dst) const
		{
			copy_elems_2d(nrows(), ncolumns(),
					this->ptr_base(), this->lead_dim(), dst.ptr_base(), dst.lead_dim());
		}

		BCS_ENSURE_INLINE eval_return_type eval() const
		{
			return *this;
		}

		BCS_ENSURE_INLINE void zero()
		{
			zero_elems(size(), ptr_base());
		}

		BCS_ENSURE_INLINE void fill(const_reference v)
		{
			fill_elems(size(), ptr_base(), v);
		}

		BCS_ENSURE_INLINE void copy_from(const_pointer src)
		{
			copy_elems(size(), src, ptr_base());
		}


	private:
		typedef detail::matrix_impl<T, RowDim, ColDim> impl_type;
		impl_type m_impl;

	}; // end class DenseMatrix<T, DynamicDim, DynamicDim>



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

	}; // end class DenseCol


	template<typename T, int ColDim>
	struct matrix_traits<DenseRow<T, ColDim> >
	{
		MAT_TRAITS_DEFS(T)

		static const index_type RowDimension = 1;
		static const index_type ColDimension = ColDim;

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

	}; // end class DenseCol


	/********************************************
	 *
	 *  Typedefs
	 *
	 ********************************************/


	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat, DynamicDim, DynamicDim)
	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat22, 2, 2)
	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat23, 2, 3)
	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat32, 2, 2)
	BCS_MATRIX_TYPEDEFS2(DenseMatrix, mat33, 3, 3)

	BCS_MATRIX_TYPEDEFS1(DenseCol, col, DynamicDim)
	BCS_MATRIX_TYPEDEFS1(DenseCol, col2, 2)
	BCS_MATRIX_TYPEDEFS1(DenseCol, col3, 2)

	BCS_MATRIX_TYPEDEFS1(DenseRow, row, DynamicDim)
	BCS_MATRIX_TYPEDEFS1(DenseRow, row2, 2)
	BCS_MATRIX_TYPEDEFS1(DenseRow, row3, 2)


}

#endif /* MATRIX_H_ */
