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

#include <bcslib/matrix/bits/dense_matrix_internal.h>

namespace bcs
{

	/********************************************
	 *
	 *  dense_matrix
	 *
	 ********************************************/

	template<typename T, int CTRows, int CTCols>
	struct matrix_traits<dense_matrix<T, CTRows, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = CTCols;

		static const bool is_readonly = false;
		static const bool is_resizable = (CTRows == DynamicDim || CTCols == DynamicDim);

		typedef T value_type;
		typedef index_t index_type;
	};

	template<typename T, int CTRows, int CTCols>
	struct has_continuous_layout<dense_matrix<T, CTRows, CTCols> >
	{
		static const bool value = true;
	};

	template<typename T, int CTRows, int CTCols>
	struct is_always_aligned<dense_matrix<T, CTRows, CTCols> >
	{
		static const bool value = true;
	};

	template<typename T, int CTRows, int CTCols>
	struct is_linear_accessible<dense_matrix<T, CTRows, CTCols> >
	{
		static const bool value = true;
	};


	template<typename T, int CTRows, int CTCols>
	class dense_matrix : public IDenseMatrix<dense_matrix<T, CTRows, CTCols>, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS(T)

	public:
		BCS_ENSURE_INLINE dense_matrix()
		: m_internal()
		{
		}

		BCS_ENSURE_INLINE dense_matrix(index_t m, index_t n)
		: m_internal(m, n)
		{
		}

		BCS_ENSURE_INLINE dense_matrix(index_t m, index_t n, const T& v)
		: m_internal(m, n)
		{
			fill_elems(nelems(), ptr_data(), v);
		}

		BCS_ENSURE_INLINE dense_matrix(index_t m, index_t n, const T* src)
		: m_internal(m, n)
		{
			copy_elems(nelems(), src, ptr_data());
		}

		BCS_ENSURE_INLINE dense_matrix(const dense_matrix& s)
		: m_internal(s.m_internal)
		{
		}

		template<class Other>
		BCS_ENSURE_INLINE dense_matrix(const IMatrixView<Other, T>& r)
		: m_internal(r.nrows(), r.ncolumns())
		{
			copy(r.derived(), *this);
		}

		template<class Expr>
		BCS_ENSURE_INLINE dense_matrix(const IMatrixXpr<Expr, T>& r)
		: m_internal(r.nrows(), r.ncolumns())
		{
			evaluate_to(r.derived(), *this);
		}

		BCS_ENSURE_INLINE void swap(dense_matrix& s)
		{
			m_internal.swap(s.m_internal);
		}

	public:
		BCS_ENSURE_INLINE dense_matrix& operator = (const dense_matrix& r)
		{
			if (this != &r)
			{
				assign_to(r, *this);
			}
			return *this;
		}

		template<class Other>
		BCS_ENSURE_INLINE dense_matrix& operator = (const IMatrixView<Other, T>& r)
		{
			assign_to(r.derived(), *this);
			return *this;
		}

		template<class Expr>
		BCS_ENSURE_INLINE dense_matrix& operator = (const IMatrixXpr<Expr, T>& r)
		{
			assign_to(r.derived(), *this);
			return *this;
		}

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return m_internal.nelems();
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
			return m_internal.nrows();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return detail::calc_offset(*this, i, j);
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
			m_internal.resize(m, n);
		}

	private:
		detail::dense_matrix_internal<T, CTRows, CTCols> m_internal;
	};


	template<typename T, int CTRows, int CTCols>
	BCS_ENSURE_INLINE
	inline void swap(dense_matrix<T, CTRows, CTCols>& a, dense_matrix<T, CTRows, CTCols>& b)
	{
		a.swap(b);
	}


	/********************************************
	 *
	 *  derived vectors
	 *
	 ********************************************/

	template<typename T, int CTRows>
	class dense_col : public dense_matrix<T, CTRows, 1>
	{
		typedef dense_matrix<T, CTRows, 1> base_mat_t;

	public:
		BCS_ENSURE_INLINE dense_col() : base_mat_t(CTRows, 1) { }

		BCS_ENSURE_INLINE explicit dense_col(index_t m) : base_mat_t(m, 1) { }

		BCS_ENSURE_INLINE dense_col(index_t m, const T& v) : base_mat_t(m, 1, v) { }

		BCS_ENSURE_INLINE dense_col(index_t m, const T* src) : base_mat_t(m, 1, src) { }

		BCS_ENSURE_INLINE dense_col(const base_mat_t& s) : base_mat_t(s) { }

		BCS_ENSURE_INLINE dense_col(const dense_col& s) : base_mat_t(s) { }

		template<class Other>
		BCS_ENSURE_INLINE dense_col(const IMatrixView<Other, T>& r) : base_mat_t(r) { }

		template<class Expr>
		BCS_ENSURE_INLINE dense_col(const IMatrixXpr<Expr, T>& r) : base_mat_t(r) { }

	public:
		BCS_ENSURE_INLINE dense_col& operator = (const base_mat_t& r)
		{
			base_mat_t::operator = (r);
			return *this;
		}

		template<class Other>
		BCS_ENSURE_INLINE dense_col& operator = (const IMatrixView<Other, T>& r)
		{
			base_mat_t::operator = (r.derived());
			return *this;
		}

		template<class Expr>
		BCS_ENSURE_INLINE dense_col& operator = (const IMatrixXpr<Expr, T>& r)
		{
			base_mat_t::operator = (r.derived());
			return *this;
		}
	};


	template<typename T, int CTCols>
	class dense_row : public dense_matrix<T, 1, CTCols>
	{
		typedef dense_matrix<T, 1, CTCols> base_mat_t;

	public:
		BCS_ENSURE_INLINE dense_row() : base_mat_t(1, CTCols) { }

		BCS_ENSURE_INLINE explicit dense_row(index_t n) : base_mat_t(1, n) { }

		BCS_ENSURE_INLINE dense_row(index_t n, const T& v) : base_mat_t(1, n, v) { }

		BCS_ENSURE_INLINE dense_row(index_t n, const T* src) : base_mat_t(1, n, src) { }

		BCS_ENSURE_INLINE dense_row(const base_mat_t& s) : base_mat_t(s) { }

		BCS_ENSURE_INLINE dense_row(const dense_row& s) : base_mat_t(s) { }

		template<class Expr>
		BCS_ENSURE_INLINE dense_row(const IMatrixXpr<Expr, T>& r) : base_mat_t(r) { }

	public:
		BCS_ENSURE_INLINE dense_row& operator = (const base_mat_t& r)
		{
			base_mat_t::operator = (r);
			return *this;
		}

		template<class Other>
		BCS_ENSURE_INLINE dense_row& operator = (const IMatrixView<Other, T>& r)
		{
			base_mat_t::operator = (r.derived());
			return *this;
		}

		template<class Expr>
		BCS_ENSURE_INLINE dense_row& operator = (const IMatrixXpr<Expr, T>& r)
		{
			base_mat_t::operator = (r.derived());
			return *this;
		}
	};


	template<typename T, int CTRows, int CTCols>
	struct expr_evaluator<dense_matrix<T, CTRows, CTCols> >
	{
		typedef dense_matrix<T, CTRows, CTCols> expr_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, T>& dst)
		{
			copy(expr, dst.derived());
		}
	};


	/********************************************
	 *
	 *  Typedefs
	 *
	 ********************************************/

	BCS_MATRIX_TYPEDEFS2(dense_matrix, mat, DynamicDim, DynamicDim)
	BCS_MATRIX_TYPEDEFS2(dense_matrix, mat22, 2, 2)
	BCS_MATRIX_TYPEDEFS2(dense_matrix, mat23, 2, 3)
	BCS_MATRIX_TYPEDEFS2(dense_matrix, mat32, 3, 2)
	BCS_MATRIX_TYPEDEFS2(dense_matrix, mat33, 3, 3)

	BCS_MATRIX_TYPEDEFS1(dense_col, col, DynamicDim)
	BCS_MATRIX_TYPEDEFS1(dense_col, col2, 2)
	BCS_MATRIX_TYPEDEFS1(dense_col, col3, 3)

	BCS_MATRIX_TYPEDEFS1(dense_row, row, DynamicDim)
	BCS_MATRIX_TYPEDEFS1(dense_row, row2, 2)
	BCS_MATRIX_TYPEDEFS1(dense_row, row3, 3)

}

#endif /* MATRIX_H_ */
