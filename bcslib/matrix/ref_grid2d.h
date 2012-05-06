/**
 * @file ref_grid2d.h
 *
 * Referenced 2D Grid
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_REF_GRID2D_H_
#define BCSLIB_REF_GRID2D_H_

#include <bcslib/matrix/bits/ref_grid2d_internal.h>
#include <bcslib/matrix/vector_accessors.h>

namespace bcs
{

	// forward declaration of vector readers

	template<typename T, int CTRows, int CTCols>
	class grid2d_linear_reader;

	template<typename T, int CTRows, int CTCols>
	class grid2d_colreaders;

	template<typename T, int CTRows, int CTCols>
	struct vec_reader<cref_grid2d<T, CTRows, CTCols> >
	{
		typedef typename select_type<CTRows == 1 || CTCols == 1,
					grid2d_linear_reader<T, CTRows, CTCols>,
					cache_linear_reader<cref_grid2d<T, CTRows, CTCols> >
				>::type type;
	};

	template<typename T, int CTRows, int CTCols>
	struct vec_reader<ref_grid2d<T, CTRows, CTCols> >
	{
		typedef typename select_type<CTRows == 1 || CTCols == 1,
					grid2d_linear_reader<T, CTRows, CTCols>,
					cache_linear_reader<ref_grid2d<T, CTRows, CTCols> >
				>::type type;
	};

	template<typename T, int CTRows, int CTCols>
	struct colwise_reader_bank<ref_grid2d<T, CTRows, CTCols> >
	{
		typedef grid2d_colreaders<T, CTRows, CTCols> type;
	};

	template<typename T, int CTRows, int CTCols>
	struct colwise_reader_bank<cref_grid2d<T, CTRows, CTCols> >
	{
		typedef grid2d_colreaders<T, CTRows, CTCols> type;
	};



	/********************************************
	 *
	 *  classes
	 *
	 ********************************************/

	template<typename T, int CTRows, int CTCols>
	struct matrix_traits<cref_grid2d<T, CTRows, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = CTCols;

		static const bool is_readonly = true;
		static const bool is_resizable = false;

		typedef T value_type;
		typedef index_t index_type;
	};

	template<typename T, int CTRows, int CTCols>
	struct has_continuous_layout<cref_grid2d<T, CTRows, CTCols> >
	{
		static const bool value = false;
	};

	template<typename T, int CTRows, int CTCols>
	struct is_always_aligned<cref_grid2d<T, CTRows, CTCols> >
	{
		static const bool value = false;
	};

	template<typename T, int CTRows, int CTCols>
	struct is_linear_accessible<cref_grid2d<T, CTRows, CTCols> >
	{
		static const bool value = (CTRows == 1 || CTCols == 1);
	};


	template<typename T, int CTRows, int CTCols>
	class cref_grid2d : public IMatrixView<cref_grid2d<T, CTRows, CTCols>, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS(T)

	public:
		BCS_ENSURE_INLINE
		cref_grid2d(const T* pdata, index_type m, index_type n, index_type step, index_type ldim)
		: m_internal(pdata, m, n, step, ldim)
		{
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

		BCS_ENSURE_INLINE index_type inner_step() const
		{
			return m_internal.inner_step();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.lead_dim();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return m_internal.sub_offset(i, j);
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return m_internal.ptr_data()[offset(i, j)];
		}

		BCS_ENSURE_INLINE const_reference operator[] (index_type i) const
		{
			return m_internal.ptr_data()[m_internal.lin_offset(i)];
		}

	private:
		detail::ref_grid2d_internal<const T, CTRows, CTCols> m_internal;

	}; // end cref_grid2d




	template<typename T, int CTRows, int CTCols>
	struct matrix_traits<ref_grid2d<T, CTRows, CTCols> >
	{
		static const int num_dimensions = 2;
		static const int compile_time_num_rows = CTRows;
		static const int compile_time_num_cols = CTCols;

		static const bool is_readonly = false;
		static const bool is_resizable = false;

		typedef T value_type;
		typedef index_t index_type;
	};

	template<typename T, int CTRows, int CTCols>
	struct has_continuous_layout<ref_grid2d<T, CTRows, CTCols> >
	{
		static const bool value = false;
	};

	template<typename T, int CTRows, int CTCols>
	struct is_always_aligned<ref_grid2d<T, CTRows, CTCols> >
	{
		static const bool value = false;
	};

	template<typename T, int CTRows, int CTCols>
	struct is_linear_accessible<ref_grid2d<T, CTRows, CTCols> >
	{
		static const bool value = (CTRows == 1 || CTCols == 1);
	};


	template<typename T, int CTRows, int CTCols>
	class ref_grid2d : public IMatrixView<ref_grid2d<T, CTRows, CTCols>, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS(T)

	public:
		ref_grid2d(T* pdata, index_type m, index_type n, index_type step, index_type ldim)
		: m_internal(pdata, m, n, step, ldim)
		{
		}

	public:
		BCS_ENSURE_INLINE ref_grid2d& operator = (const ref_grid2d& r)
		{
			if (this != &r)
			{
				assign(r);
			}
			return *this;
		}

		template<class Expr>
		BCS_ENSURE_INLINE ref_grid2d& operator = (const IMatrixXpr<Expr, T>& r)
		{
			assign(r.derived());
			return *this;
		}

	private:
		template<class Mat>
		void assign(const Mat& mat)
		{
			typedef typename colwise_reader_bank<Mat>::type bank_t;
			typedef typename bank_t::reader_type reader_t;

			check_arg( has_same_size(*this, mat),
					"The matrix on the right hand side has inconsistent size.");

			const index_t m = nrows();
			const index_t n = ncolumns();
			const index_t ldim = lead_dim();
			const index_t step = inner_step();

			bank_t bank(mat);
			T *pd = ptr_data();

			for (index_t j = 0; j < n; ++j, pd += ldim)
			{
				reader_t in(bank, j);
				for (index_t i = 0; i < m; ++i)
				{
					pd[i * step] = in.get(i);
				}
			}
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

		BCS_ENSURE_INLINE index_type inner_step() const
		{
			return m_internal.inner_step();
		}

		BCS_ENSURE_INLINE index_type lead_dim() const
		{
			return m_internal.lead_dim();
		}

		BCS_ENSURE_INLINE index_type offset(index_type i, index_type j) const
		{
			return m_internal.sub_offset(i, j);
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
			return m_internal.ptr_data()[m_internal.lin_offset(i)];
		}

		BCS_ENSURE_INLINE reference operator[] (index_type i)
		{
			return m_internal.ptr_data()[m_internal.lin_offset(i)];
		}

	private:
		detail::ref_grid2d_internal<T, CTRows, CTCols> m_internal;

	}; // end ref_grid2d



	/********************************************
	 *
	 *  vector readers
	 *
	 ********************************************/

	template<typename T, int CTRows, int CTCols>
	class grid2d_linear_reader
	{
#ifdef BCS_USE_STATIC_ASSERT
		static_assert(CTRows == 1 || CTCols == 1,
				"Either CTRows or CTCols must equal 1");
#endif
	public:
		BCS_ENSURE_INLINE
		grid2d_linear_reader(const ref_grid2d<T, CTRows, CTCols>& mat)
		: m_data(mat.ptr_data())
		, m_step(CTCols == 1 ? mat.inner_step() : mat.lead_dim())
		{
		}

		BCS_ENSURE_INLINE
		grid2d_linear_reader(const cref_grid2d<T, CTRows, CTCols>& mat)
		: m_data(mat.ptr_data())
		, m_step(CTCols == 1 ? mat.inner_step() : mat.lead_dim())
		{
		}

		BCS_ENSURE_INLINE index_t step() const
		{
			return m_step;
		}

	public:
		BCS_ENSURE_INLINE T get(const index_t i) const
		{
			return m_data[i * m_step];
		}

	private:
		const T *m_data;
		const index_t m_step;
	};


	template<typename T, int CTRows, int CTCols>
	class grid2d_colreaders
	{
	public:
		BCS_ENSURE_INLINE
		grid2d_colreaders(const ref_grid2d<T, CTRows, CTCols>& mat)
		: m_data(mat.ptr_data())
		, m_step(mat.inner_step())
		, m_leaddim(mat.lead_dim())
		{ }

		BCS_ENSURE_INLINE
		grid2d_colreaders(const cref_grid2d<T, CTRows, CTCols>& mat)
		: m_data(mat.ptr_data())
		, m_step(mat.inner_step())
		, m_leaddim(mat.lead_dim())
		{ }

	public:
		class reader_type
		{
		public:
			BCS_ENSURE_INLINE
			reader_type(const grid2d_colreaders& host, const index_t j)
			: m_col(host.m_data + j * host.m_leaddim), m_step(host.m_step)
			{
			}

			BCS_ENSURE_INLINE T get(const index_t i) const
			{
				return m_col[i * m_step];
			}

		private:
			const T* m_col;
			const index_t m_step;
		};

	private:
		const T* m_data;
		const index_t m_step;
		const index_t m_leaddim;
	};


	template<typename T, int CTRows, int CTCols>
	struct expr_evaluator<cref_grid2d<T, CTRows, CTCols> >
	{
		typedef cref_grid2d<T, CTRows, CTCols> expr_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, T>& dst)
		{
			copy(expr, dst.derived());
		}
	};


	template<typename T, int CTRows, int CTCols>
	struct expr_evaluator<ref_grid2d<T, CTRows, CTCols> >
	{
		typedef ref_grid2d<T, CTRows, CTCols> expr_type;

		template<class DMat>
		BCS_ENSURE_INLINE
		static void evaluate(const expr_type& expr, IDenseMatrix<DMat, T>& dst)
		{
			copy(expr, dst.derived());
		}
	};


}

#endif








