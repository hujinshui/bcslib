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
#include <bcslib/matrix/matrix_assign.h>

#include <bcslib/matrix/bits/ref_matrix_internal.h>


namespace bcs
{

	/********************************************
	 *
	 *  cref_matrix
	 *
	 ********************************************/

	template<typename T, int CTRows, int CTCols>
	struct matrix_traits<cref_matrix<T, CTRows, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = CTCols;

		static const bool is_linear_indexable = true;
		static const bool is_continuous = true;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef T value_type;
		typedef index_t index_type;
	};


	template<typename T, int CTRows, int CTCols>
	class cref_matrix : public IDenseMatrix<cref_matrix<T, CTRows, CTCols>, T>
	{
	public:
		BCS_MAT_TRAITS_CDEFS(T)

	private:
		typedef detail::dim_helper<CTRows==1, CTCols==1> _dim_helper;

		static const int CTSize = CTRows * CTCols;
		static const bool IsDynamic = (CTSize == 0);

	public:

		BCS_ENSURE_INLINE
		cref_matrix(const T* pdata, index_type m, index_type n)
		: m_internal(pdata, m, n)
		{
		}

	private:
		cref_matrix& operator = (const cref_matrix& );  // no assignment

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return _dim_helper::nelems(nrows(), ncolumns());
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_internal.ncolumns();
		}

		BCS_ENSURE_INLINE const_pointer ptr_data() const
		{
			return m_internal.ptr_data();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.lead_dim();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return _dim_helper::offset(m_internal.lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type i) const
		{
			return m_internal.ptr_data()[i];
		}

		BCS_ENSURE_INLINE void resize(index_type m, index_type n)
		{
			check_arg(nrows() == m && ncolumns() == n,
					"The size of a cref_matrix instance can not be changed.");
		}

	private:
		detail::ref_matrix_internal<const T, CTRows, CTCols> m_internal;

	}; // end class cref_matrix




	/********************************************
	 *
	 *  ref_matrix
	 *
	 ********************************************/


	template<typename T, int CTRows, int CTCols>
	struct matrix_traits<ref_matrix<T, CTRows, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = CTCols;

		static const bool is_linear_indexable = true;
		static const bool is_continuous = true;
		static const bool is_sparse = false;
		static const bool is_readonly = false;

		typedef T value_type;
		typedef index_t index_type;
	};


	template<typename T, int CTRows, int CTCols>
	class ref_matrix : public IDenseMatrix<ref_matrix<T, CTRows, CTCols>, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS(T)

	private:
		typedef detail::dim_helper<CTRows==1, CTCols==1> _dim_helper;

	public:
		ref_matrix(T* pdata, index_type m, index_type n)
		: m_internal(pdata, m, n)
		{
		}

	public:
		BCS_ENSURE_INLINE ref_matrix& operator = (const ref_matrix& r)
		{
			if (this != &r)
			{
				assign_to(r, *this);
			}
			return *this;
		}

		template<class Other>
		BCS_ENSURE_INLINE ref_matrix& operator = (const IMatrixView<Other, T>& r)
		{
			assign_to(r.derived(), *this);
			return *this;
		}

		template<class Expr>
		BCS_ENSURE_INLINE ref_matrix& operator = (const IMatrixXpr<Expr, T>& r)
		{
			assign_to(r.derived(), *this);
			return *this;
		}


	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return _dim_helper::nelems(nrows(), ncolumns());
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_internal.ncolumns();
		}

		BCS_ENSURE_INLINE const_pointer ptr_data() const
		{
			return m_internal.ptr_data();
		}

		BCS_ENSURE_INLINE pointer ptr_data()
		{
			return m_internal.ptr_data();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.lead_dim();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return _dim_helper::offset(m_internal.lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type i) const
		{
			return m_internal.ptr_data()[i];
		}

		BCS_ENSURE_INLINE reference operator[] (index_type i)
		{
			return m_internal.ptr_data()[i];
		}

		BCS_ENSURE_INLINE void resize(index_type m, index_type n)
		{
			check_arg(nrows() == m && ncolumns() == n,
					"The size of a ref_matrix instance can not be changed.");
		}

	private:
		detail::ref_matrix_internal<T, CTRows, CTCols> m_internal;

	}; // end ref_matrix


	/********************************************
	 *
	 *  vectors derived from (c)ref_matrix
	 *
	 ********************************************/

	template<typename T, int CTRows>
	class cref_col: public cref_matrix<T, CTRows, 1>
	{
		typedef cref_matrix<T, CTRows, 1> base_mat_t;

	public:
		typedef index_t index_type;

		BCS_ENSURE_INLINE
		cref_col(const T* pdata, index_type m)
		: base_mat_t(pdata, m, 1) { }
	};

	template<typename T, int CTRows>
	class ref_col: public ref_matrix<T, CTRows, 1>
	{
		typedef ref_matrix<T, CTRows, 1> base_mat_t;

	public:
		typedef index_t index_type;

		BCS_ENSURE_INLINE
		ref_col(T* pdata, index_type m)
		: base_mat_t(pdata, m, 1) { }

		BCS_ENSURE_INLINE ref_col& operator = (const base_mat_t& r)
		{
			base_mat_t::operator = (r);
			return *this;
		}

		template<class Other>
		BCS_ENSURE_INLINE ref_col& operator = (const IMatrixView<Other, T>& r)
		{
			base_mat_t::operator = (r.derived());
			return *this;
		}

		template<class Expr>
		BCS_ENSURE_INLINE ref_col& operator = (const IMatrixXpr<Expr, T>& r)
		{
			base_mat_t::operator = (r.derived());
			return *this;
		}
	};


	template<typename T, int CTCols>
	class cref_row: public cref_matrix<T, 1, CTCols>
	{
		typedef cref_matrix<T, 1, CTCols> base_mat_t;

	public:
		typedef index_t index_type;

		BCS_ENSURE_INLINE
		cref_row(const T* pdata, index_type n)
		: base_mat_t(pdata, 1, n) { }
	};

	template<typename T, int CTCols>
	class ref_row: public ref_matrix<T, 1, CTCols>
	{
		typedef ref_matrix<T, 1, CTCols> base_mat_t;

	public:
		typedef index_t index_type;

		BCS_ENSURE_INLINE
		ref_row(T* pdata, index_type n)
		: base_mat_t(pdata, 1, n) { }

		BCS_ENSURE_INLINE ref_row& operator = (const base_mat_t& r)
		{
			base_mat_t::operator = (r);
			return *this;
		}

		template<class Other>
		BCS_ENSURE_INLINE ref_row& operator = (const IMatrixView<Other, T>& r)
		{
			base_mat_t::operator = (r.derived());
			return *this;
		}

		template<class Expr>
		BCS_ENSURE_INLINE ref_row& operator = (const IMatrixXpr<Expr, T>& r)
		{
			base_mat_t::operator = (r.derived());
			return *this;
		}
	};



	/********************************************
	 *
	 *  cref_matrix_ex
	 *
	 ********************************************/

	template<typename T, int CTRows, int CTCols>
	struct matrix_traits<cref_matrix_ex<T, CTRows, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = CTCols;

		static const bool is_linear_indexable = true;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = true;

		typedef T value_type;
		typedef index_t index_type;
	};


	template<typename T, int CTRows, int CTCols>
	class cref_matrix_ex : public IDenseMatrix<cref_matrix_ex<T, CTRows, CTCols>, T>
	{
	public:
		BCS_MAT_TRAITS_CDEFS(T)

	private:
		typedef detail::dim_helper<CTRows==1, CTCols==1> _dim_helper;

		static const int CTSize = CTRows * CTCols;
		static const bool IsDynamic = (CTSize == 0);

	public:

		BCS_ENSURE_INLINE
		cref_matrix_ex(const T* pdata, index_type m, index_type n, index_type ldim)
		: m_internal(pdata, m, n, ldim)
		{
		}

	private:
		typedef detail::ref_matrix_ex_offset<CTRows==1, CTCols==1> _linearidx_helper;

		cref_matrix_ex& operator = (const cref_matrix_ex& );  // no assignment

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return _dim_helper::nelems(nrows(), ncolumns());
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_internal.ncolumns();
		}

		BCS_ENSURE_INLINE const_pointer ptr_data() const
		{
			return m_internal.ptr_data();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.lead_dim();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return _dim_helper::offset(m_internal.lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type i) const
		{
			return m_internal.ptr_data()[_linearidx_helper::get(lead_dim(), i)];
		}

		BCS_ENSURE_INLINE void resize(index_type m, index_type n)
		{
			check_arg(nrows() == m && ncolumns() == n,
					"The size of a cref_matrix instance can not be changed.");
		}

	private:
		detail::ref_matrix_internal_ex<const T, CTRows, CTCols> m_internal;

	}; // end class cref_matrix_ex



	/********************************************
	 *
	 *  ref_matrix_ex
	 *
	 ********************************************/


	template<typename T, int CTRows, int CTCols>
	struct matrix_traits<ref_matrix_ex<T, CTRows, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = CTCols;

		static const bool is_linear_indexable = true;
		static const bool is_continuous = false;
		static const bool is_sparse = false;
		static const bool is_readonly = false;

		typedef T value_type;
		typedef index_t index_type;
	};


	template<typename T, int CTRows, int CTCols>
	class ref_matrix_ex : public IDenseMatrix<ref_matrix_ex<T, CTRows, CTCols>, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS(T)

	private:
		typedef detail::dim_helper<CTRows==1, CTCols==1> _dim_helper;
		typedef detail::ref_matrix_ex_offset<CTRows==1, CTCols==1> _linearidx_helper;

	public:
		ref_matrix_ex(T* pdata, index_type m, index_type n, index_type ldim)
		: m_internal(pdata, m, n, ldim)
		{
		}

	public:
		BCS_ENSURE_INLINE ref_matrix_ex& operator = (const ref_matrix_ex& r)
		{
			if (this != &r)
			{
				assign_to(r, *this);
			}
			return *this;
		}

		template<class Other>
		BCS_ENSURE_INLINE ref_matrix_ex& operator = (const IMatrixView<Other, T>& r)
		{
			assign_to(r.derived(), *this);
			return *this;
		}

		template<class Expr>
		BCS_ENSURE_INLINE ref_matrix_ex& operator = (const IMatrixXpr<Expr, T>& r)
		{
			assign_to(r.derived(), *this);
			return *this;
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return _dim_helper::nelems(nrows(), ncolumns());
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return static_cast<size_type>(nelems());
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return m_internal.ncolumns();
		}

		BCS_ENSURE_INLINE const_pointer ptr_data() const
		{
			return m_internal.ptr_data();
		}

		BCS_ENSURE_INLINE pointer ptr_data()
		{
			return m_internal.ptr_data();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.lead_dim();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return _dim_helper::offset(m_internal.lead_dim(), i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type i) const
		{
			return m_internal.ptr_data()[_linearidx_helper::get(lead_dim(), i)];
		}

		BCS_ENSURE_INLINE reference operator[] (index_type i)
		{
			return m_internal.ptr_data()[_linearidx_helper::get(lead_dim(), i)];
		}

		BCS_ENSURE_INLINE void resize(index_type m, index_type n)
		{
			check_arg(nrows() == m && ncolumns() == n,
					"The size of a ref_matrix instance can not be changed.");
		}

	private:
		detail::ref_matrix_internal_ex<T, CTRows, CTCols> m_internal;

	}; // end ref_matrix_ex


}


#endif /* REF_MATRIX_H_ */






