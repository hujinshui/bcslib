/**
 * @file dense_matrix_internal.h
 *
 * The internal implementation for dense_matrix
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_DENSE_MATRIX_INTERNAL_H_
#define BCSLIB_DENSE_MATRIX_INTERNAL_H_

#include <bcslib/matrix/matrix_base.h>
#include <bcslib/matrix/bits/matrix_helpers.h>
#include <bcslib/core/block.h>

#include <algorithm>

namespace bcs { namespace detail {


	template<typename T, int CTRows, int CTCols> class dense_matrix_internal;

	template<typename T, int CTRows, int CTCols>
	class dense_matrix_internal
	{
	public:
		BCS_ENSURE_INLINE
		dense_matrix_internal() { }

		BCS_ENSURE_INLINE
		dense_matrix_internal(index_t m, index_t n)
		{
			check_arg(m == CTRows && n == CTCols);
		}

		inline void assign(const dense_matrix_internal& s)
		{
			m_blk = s.m_blk;
		}

		inline void swap(dense_matrix_internal& s)
		{
			m_blk.swap(s.m_blk);
		}

		BCS_ENSURE_INLINE index_t nelems() const { return CTRows * CTCols; }

		BCS_ENSURE_INLINE index_t nrows() const { return CTRows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return CTCols; }

		BCS_ENSURE_INLINE const T *ptr_data() const { return m_blk.ptr_begin(); }

		BCS_ENSURE_INLINE T *ptr_data() { return m_blk.ptr_begin(); }

		void resize(index_t m, index_t n)
		{
			check_arg(m == CTRows && n == CTCols,
					"Attempted to change a fixed dimension.");
		}

	private:
		static_block<T, CTRows * CTCols> m_blk;
	};


	template<typename T, int CTRows>
	class dense_matrix_internal<T, CTRows, DynamicDim>
	{
	public:
		BCS_ENSURE_INLINE
		dense_matrix_internal() : m_blk(), m_ncols(0) { }

		BCS_ENSURE_INLINE
		dense_matrix_internal(index_t m, index_t n)
		: m_blk(check_forward(m, m == CTRows) * n), m_ncols(n) { }

		inline void assign(const dense_matrix_internal& s)
		{
			if (this != &s)
			{
				m_blk = s.m_blk;
				m_ncols = s.m_ncols;
			}
		}

		inline void swap(dense_matrix_internal& s)
		{
			m_blk.swap(s.m_blk);
			std::swap(m_ncols, s.m_ncols);
		}

		BCS_ENSURE_INLINE index_t nelems() const { return CTRows * m_ncols; }

		BCS_ENSURE_INLINE index_t nrows() const { return CTRows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return m_ncols; }

		BCS_ENSURE_INLINE const T *ptr_data() const { return m_blk.ptr_begin(); }

		BCS_ENSURE_INLINE T *ptr_data() { return m_blk.ptr_begin(); }

		void resize(index_t m, index_t n)
		{
			check_arg(m == CTRows,
					"Attempted to change a fixed dimension.");

			if (n != m_ncols)
			{
				m_blk.resize(m * n);
				m_ncols = n;
			}
		}

	private:
		block<T, aligned_allocator<T> > m_blk;
		index_t m_ncols;
	};


	template<typename T, int CTCols>
	class dense_matrix_internal<T, DynamicDim, CTCols>
	{
	public:
		BCS_ENSURE_INLINE
		dense_matrix_internal() : m_blk(), m_nrows(0) { }

		BCS_ENSURE_INLINE
		dense_matrix_internal(index_t m, index_t n)
		: m_blk(m * check_forward(n, n == CTCols)), m_nrows(m) { }

		inline void assign(const dense_matrix_internal& s)
		{
			if (this != &s)
			{
				m_blk = s.m_blk;
				m_nrows = s.m_nrows;
			}
		}

		inline void swap(dense_matrix_internal& s)
		{
			m_blk.swap(s.m_blk);
			std::swap(m_nrows, s.m_nrows);
		}

		BCS_ENSURE_INLINE index_t nelems() const { return m_nrows * CTCols; }

		BCS_ENSURE_INLINE index_t nrows() const { return m_nrows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return CTCols; }

		BCS_ENSURE_INLINE const T *ptr_data() const { return m_blk.ptr_begin(); }

		BCS_ENSURE_INLINE T *ptr_data() { return m_blk.ptr_begin(); }

		void resize(index_t m, index_t n)
		{
			check_arg(n == CTCols,
					"Attempted to change a fixed dimension.");

			if (m != m_nrows)
			{
				m_blk.resize(m * n);
				m_nrows = m;
			}
		}

	private:
		block<T, aligned_allocator<T> > m_blk;
		index_t m_nrows;
	};


	template<typename T>
	class dense_matrix_internal<T, DynamicDim, DynamicDim>
	{
	public:
		BCS_ENSURE_INLINE
		dense_matrix_internal() : m_blk(), m_nrows(0), m_ncols(0) { }

		BCS_ENSURE_INLINE
		dense_matrix_internal(index_t m, index_t n)
		: m_blk(m * n), m_nrows(m), m_ncols(n) { }

		inline void assign(const dense_matrix_internal& s)
		{
			if (this != &s)
			{
				m_blk = s.m_blk;
				m_nrows = s.m_nrows;
				m_ncols = s.m_ncols;
			}
		}

		inline void swap(dense_matrix_internal& s)
		{
			m_blk.swap(s.m_blk);
			std::swap(m_nrows, s.m_nrows);
			std::swap(m_ncols, s.m_ncols);
		}

		BCS_ENSURE_INLINE index_t nelems() const { return m_nrows * m_ncols; }

		BCS_ENSURE_INLINE index_t nrows() const { return m_nrows; }

		BCS_ENSURE_INLINE index_t ncolumns() const { return m_ncols; }

		BCS_ENSURE_INLINE const T *ptr_data() const { return m_blk.ptr_begin(); }

		BCS_ENSURE_INLINE T *ptr_data() { return m_blk.ptr_begin(); }

		void resize(index_t m, index_t n)
		{
			if (nelems() != m * n)
			{
				m_blk.resize(m * n);
			}

			m_nrows = m;
			m_ncols = n;
		}

	private:
		block<T, aligned_allocator<T> > m_blk;
		index_t m_nrows;
		index_t m_ncols;
	};


} }

#endif /* DENSE_MATRIX_INTERNAL_H_ */
