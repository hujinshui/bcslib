/**
 * @file ref_matrix.h
 *
 * The reference matrix/vector classes
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REF_MATRIX_H_
#define BCSLIB_REF_MATRIX_H_

#include <bcslib/matrix/matrix_base.h>

namespace bcs
{

	// forward declaration

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class RefMatrix;



	/********************************************
	 *
	 *  Helper
	 *
	 ********************************************/

	namespace detail
	{
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

		typedef const RefMatrix<T, RowDim, ColDim>& eval_return_type;
	};


	template<typename T, int RowDim, int ColDim>
	class RefMatrix : public IDenseMatrix<RefMatrix<T, RowDim, ColDim>, T>
	{
	public:
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(RowDim == DynamicDim || RowDim >= 1, "Invalid template argument RowDim");
		static_assert(ColDim == DynamicDim || ColDim >= 1, "Invalid template argument ColDim");
#endif

	public:
		MAT_TRAITS_DEFS(T)

		typedef IDenseMatrix<RefMatrix<T, RowDim, ColDim>, T> base_type;
		static const int RowDimension = RowDim;
		static const int ColDimension = ColDim;
		typedef const RefMatrix<T, RowDim, ColDim>& eval_return_type;

	public:
		BCS_ENSURE_INLINE
		RefMatrix()
		: m_ptr(BCS_NULL), m_nrows(RowDim), m_ncols(ColDim)
		{
		}

		BCS_ENSURE_INLINE
		RefMatrix(T *refptr, index_t m, index_t n)
		: m_ptr(refptr), m_nrows(m), m_ncols(n)
		{
			check_input_dims(m, n);
		}

		BCS_ENSURE_INLINE
		void reset(T *refptr, index_t m, index_t n)
		{
			check_input_dims(m, n);

			m_ptr = refptr;
			m_nrows = m;
			m_ncols = n;
		}

	private:
		BCS_ENSURE_INLINE
		void check_input_dims(index_t m, index_t n)
		{
			if (RowDim >= 1) check_arg( m == RowDim );
			if (ColDim >= 1) check_arg( n == ColDim );
		}


	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_nrows * m_ncols;
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return (size_type)nelems();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_ncols;
		}

		BCS_ENSURE_INLINE bool is_empty() const
		{
			return nrows() == 0 || ncolumns() == 0;
		}

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return m_ptr;
		}

		BCS_ENSURE_INLINE pointer ptr_base()
		{
			return m_ptr;
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
			return m_nrows;
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return detail::calc_offset<RowDim, ColDim>(lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_ptr[offset(i, j)];
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return m_ptr[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx) const
		{
			return m_ptr[idx];
		}

		BCS_ENSURE_INLINE reference operator[] (index_type idx)
		{
			return m_ptr[idx];
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
				copy_elems(size(), this->ptr_base(), dst.ptr_base());
			}
		}

		template<class DstDerived>
		BCS_ENSURE_INLINE void eval_to_block(IDenseMatrixBlock<DstDerived, T>& dst) const
		{
			copy_elems_2d(size_t(nrows()), size_t(ncolumns()),
					this->ptr_base(), size_t(this->lead_dim()), dst.ptr_base(), size_t(dst.lead_dim()));
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
		pointer m_ptr;
		index_t m_nrows;
		index_t m_ncols;

	}; // end class RefMatrix

}

// Useful Macros for generating reference matrices

#define bcs_ref_mat(Ty, refvar, PtrExpr, m, n) bcs::RefMatrix<Ty> refvar(PtrExpr, m, n)
#define bcs_cref_mat(Ty, crefvar, ConstPtrExpr, m, n) \
	bcs::RefMatrix<Ty> bcs_internal_refmat_##crefvar(const_cast<Ty*>(ConstPtrExpr), m, n); \
	const bcs::RefMatrix<Ty>& crefvar = bcs_internal_refmat_##crefvar;

#define bcs_ref_col(Ty, refvar, PtrExpr, len) bcs::RefMatrix<Ty, bcs::DynamicDim, 1> refvar(PtrExpr, len, 1)
#define bcs_cref_col(Ty, crefvar, ConstPtrExpr, len) \
	bcs::RefMatrix<Ty, bcs::DynamicDim, 1> bcs_internal_refmat_##crefvar(const_cast<Ty*>(ConstPtrExpr), len, 1); \
	const bcs::RefMatrix<Ty, bcs::DynamicDim, 1>& crefvar = bcs_internal_refmat_##crefvar;

#define bcs_ref_row(Ty, refvar, PtrExpr, len) bcs::RefMatrix<Ty, 1, bcs::DynamicDim> refvar(PtrExpr, 1, len)
#define bcs_cref_row(Ty, crefvar, ConstPtrExpr, len) \
	bcs::RefMatrix<Ty, 1, bcs::DynamicDim> bcs_internal_refmat_##crefvar(const_cast<Ty*>(ConstPtrExpr), 1, len); \
	const bcs::RefMatrix<Ty, 1, bcs::DynamicDim>& crefvar = bcs_internal_refmat_##crefvar;


#endif /* REF_MATRIX_H_ */






