/**
 * @file dense_impl_interface.h
 *
 * The plugin for implementing matrix interfaces uniformly
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef DENSE_MATRIX_FACET_H_
#define DENSE_MATRIX_FACET_H_

#include <bcslib/matrix/matrix_base.h>
#include <bcslib/base/type_traits.h>
#include "matrix_helpers.h"


namespace bcs { namespace detail {

	template<typename T, class Internal, class CName>
	class DenseMatrixFacet : public IDenseMatrix<CName, T>
	{
	public:
		MAT_TRAITS_DEFS(T)

		const static int RowDimension = matrix_traits<CName>::RowDimension;
		const static int ColDimension = matrix_traits<CName>::ColDimension;

	private:
		static const bool IsReadOnly = matrix_traits<CName>::IsReadOnly;
		typedef typename detail::adapt_const<pointer, IsReadOnly>::type nc_pointer;
		typedef typename detail::adapt_const<reference, IsReadOnly>::type nc_reference;

	public:

		BCS_ENSURE_INLINE DenseMatrixFacet(const Internal& in)
		: m_internal(in)
		{
		}

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
			return nrows() == 0 || ncolumns() == 0;
		}

		BCS_ENSURE_INLINE const_pointer ptr_base() const
		{
			return m_internal.ptr();
		}

		BCS_ENSURE_INLINE nc_pointer ptr_base()
		{
			return m_internal.ptr();
		}

		BCS_ENSURE_INLINE const_pointer col_ptr(index_type j) const
		{
			return ptr_base() + j * lead_dim();
		}

		BCS_ENSURE_INLINE nc_pointer col_ptr(index_type j)
		{
			return ptr_base() + j * lead_dim();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return calc_offset<RowDimension, ColDimension>(lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_internal.at(offset(i, j));
		}

		BCS_ENSURE_INLINE nc_reference elem(index_type i, index_type j)
		{
			return m_internal.at(offset(i, j));
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type idx) const
		{
			return m_internal.at(idx);
		}

		BCS_ENSURE_INLINE nc_reference operator[] (index_type idx)
		{
			return m_internal.at(idx);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			detail::check_matrix_indices(*this, i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE nc_reference operator() (index_type i, index_type j)
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

		BCS_ENSURE_INLINE void zero()
		{
			if (!IsReadOnly)
			{
				zero_elems(size(), ptr_base());
			}
			else BCS_CREF_NOWRITE
		}

		BCS_ENSURE_INLINE void fill(const_reference v)
		{
			if (!IsReadOnly)
			{
				fill_elems(size(), ptr_base(), v);
			}
			else BCS_CREF_NOWRITE
		}

		BCS_ENSURE_INLINE void copy_from(const_pointer src)
		{
			if (!IsReadOnly)
			{
				copy_elems(size(), src, ptr_base());
			}
			else BCS_CREF_NOWRITE
		}

	protected:
		Internal m_internal;
	};


} }


#define BCS_DEFINE_MATRIX_FACET_INTERNAL \
		BCS_ENSURE_INLINE const internal_type& internal() const { return this->m_internal; } \
		BCS_ENSURE_INLINE internal_type& internal() { return this->m_internal; }


#endif /* DENSE_IMPL_INTERFACE_H_ */
