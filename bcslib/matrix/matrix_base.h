/**
 * @file matrix_base.h
 *
 * Basic concepts and related definitions for matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_BASE_H_
#define BCSLIB_MATRIX_BASE_H_

#include <bcslib/core.h>
#include <bcslib/utils/arg_check.h>
#include <bcslib/matrix/bits/matrix_helpers.h>


#define BCS_MAT_TRAITS_DEFS_FOR_BASE(D, T) \
	typedef T value_type; \
	typedef typename mat_access<D>::const_pointer const_pointer; \
	typedef typename mat_access<D>::const_reference const_reference; \
	typedef typename mat_access<D>::pointer pointer; \
	typedef typename mat_access<D>::reference reference; \
	typedef size_t size_type; \
	typedef index_t difference_type; \
	typedef index_t index_type;

#define BCS_MAT_TRAITS_CDEFS(T) \
	typedef T value_type; \
	typedef const T* const_pointer; \
	typedef const T& const_reference; \
	typedef size_t size_type; \
	typedef index_t difference_type; \
	typedef index_t index_type;

#define BCS_MAT_TRAITS_DEFS(T) \
	typedef T value_type; \
	typedef const T* const_pointer; \
	typedef const T& const_reference; \
	typedef T* pointer; \
	typedef T& reference; \
	typedef size_t size_type; \
	typedef index_t difference_type; \
	typedef index_t index_type;


namespace bcs
{
	// forward declarations

	template<class Derived> struct matrix_traits;

	template<class Derived, typename T> class IMatrixXpr;
	template<class Derived, typename T> class IMatrixView;
	template<class Derived, typename T> class IRegularMatrix;
	template<class Derived, typename T> class IDenseMatrix;


	/********************************************
	 *
	 *  Meta-programming helpers
	 *
	 ********************************************/

	template<class Derived> struct dim_helper
	{
		static const bool with_single_row = (matrix_traits<Derived>::compile_time_num_rows == 1);
		static const bool with_single_col = (matrix_traits<Derived>::compile_time_num_cols == 1);
		typedef detail::dim_helper<with_single_row, with_single_col> type;
	};

	const int DynamicDim = 0;

	template<class Mat1>
	struct ct_rows
	{
		static const int value = matrix_traits<Mat1>::compile_time_num_rows;
	};

	template<class Mat1>
	struct ct_cols
	{
		static const int value = matrix_traits<Mat1>::compile_time_num_cols;
	};

	template<class Mat1, class Mat2>
	struct binary_ct_rows
	{
		static const int v1 = matrix_traits<Mat1>::compile_time_num_rows;
		static const int v2 = matrix_traits<Mat2>::compile_time_num_rows;
		static const int value = v1 > v2 ? v1 : v2;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(!(v1 > 0 && v2 > 0 && v1 != v2), "Incompatible compile-time dimensions.");
#endif
	};

	template<class Mat1, class Mat2>
	struct binary_ct_cols
	{
		static const int v1 = matrix_traits<Mat1>::compile_time_num_cols;
		static const int v2 = matrix_traits<Mat2>::compile_time_num_cols;
		static const int value = v1 > v2 ? v1 : v2;

#ifdef BCS_USE_STATIC_ASSERT
		static_assert(!(v1 > 0 && v2 > 0 && v1 != v2), "Incompatible compile-time dimensions.");
#endif
	};


	template<class Derived>
	struct mat_access
	{
		typedef typename matrix_traits<Derived>::value_type value_type;
		static const bool is_readonly = matrix_traits<Derived>::is_readonly;

		typedef const value_type* const_pointer;
		typedef const value_type& const_reference;
		typedef typename access_types<value_type, is_readonly>::pointer pointer;
		typedef typename access_types<value_type, is_readonly>::reference reference;
	};


	template<class Derived, template<class D, typename T> class Interface>
	struct has_matrix_interface
	{
		typedef Interface<Derived, typename matrix_traits<Derived>::value_type> expect_base;
		static const bool value = is_base_of<expect_base, Derived>::value;
	};


	// forward declaration of some important types

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class cref_matrix;
	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class  ref_matrix;

	template<typename T, int CTRows=DynamicDim> class cref_col;
	template<typename T, int CTRows=DynamicDim> class  ref_col;
	template<typename T, int CTCols=DynamicDim> class cref_row;
	template<typename T, int CTCols=DynamicDim> class  ref_row;

	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class cref_matrix_ex;
	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class  ref_matrix_ex;

	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class cref_grid2d;
	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class  ref_grid2d;

	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class dense_matrix;
	template<typename T, int CTRows=DynamicDim> class dense_col;
	template<typename T, int CTCols=DynamicDim> class dense_row;


	/********************************************
	 *
	 *  Concepts
	 *
	 *  Each concept is associated with a
	 *  class as a static polymorphism base
	 *
	 ********************************************/

	/**
	 * Concept: Matrix Expression
	 * ------------------------------
	 *
	 *   Any entity that is a 2D map of values or can be
	 *   evaluated into a 2D map of values is called a
	 *   2D array.
	 *
	 * 	 Each 2D array class (say C), must inherit, directly
	 * 	 or indirectly, the IBase2DXpr class, implementing
	 * 	 all the delegate member functions.
	 *
	 *   A specialized version of array_traits<C> must be provided,
	 *   which should contain the following static members:
	 *
	 *   - num_dimensions:				an int value, which must be set to 2
	 *   - compile_time_num_rows:		compile-time number of rows
	 *   - compile_time_num_cols:		compile-time number of columns
	 *
	 *	 - is_linear_indexable:		whether it supports linear indexing, i.e. A[i]
	 *	 - is_continuous:			whether it has a continuous memory layout
	 *   - is_sparse:				whether it adopts a sparse representation
	 *   - is_readonly:				whether it is read-only
	 *
	 *	 - value_type:			the type of element value
	 *	 - index_type:			the (default) type of index
	 *
	 */
	template<class Derived, typename T>
	class IMatrixXpr
	{
	public:
		BCS_MAT_TRAITS_DEFS_FOR_BASE(Derived, T)
		BCS_CRTP_REF

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

	}; // end class IMatrixBase

	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	inline bool is_empty(const IMatrixXpr<Derived, T>& X)
	{
		typedef typename dim_helper<Derived>::type dim_helper_t;
		return dim_helper_t::is_empty(X.nrows(), X.ncolumns());
	}

	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	inline bool is_scalar(const IMatrixXpr<Derived, T>& X)
	{
		typedef typename dim_helper<Derived>::type dim_helper_t;
		return dim_helper_t::is_scalar(X.nrows(), X.ncolumns());
	}

	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	inline bool is_vector(const IMatrixXpr<Derived, T>& X)
	{
		typedef typename dim_helper<Derived>::type dim_helper_t;
		return dim_helper_t::is_vector(X.nrows(), X.ncolumns());
	}

	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	inline bool is_column(const IMatrixXpr<Derived, T>& X)
	{
		typedef typename dim_helper<Derived>::type dim_helper_t;
		return dim_helper_t::is_column(X.ncolumns());
	}

	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	inline bool is_row(const IMatrixXpr<Derived, T>& X)
	{
		typedef typename dim_helper<Derived>::type dim_helper_t;
		return dim_helper_t::is_row(X.nrows());
	}

	template<class Derived1, typename T1, class Derived2, typename T2>
	BCS_ENSURE_INLINE
	inline bool is_same_size(const IMatrixXpr<Derived1, T1>& A, const IMatrixXpr<Derived2, T2>& B)
	{
		return A.nrows() == B.nrows() && A.ncolumns() == B.ncolumns();
	}

	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	inline void check_subscripts_in_range(const IMatrixXpr<Derived, T>& X, index_t i, index_t j)
	{
#ifndef BCSLIB_NO_DEBUG
		check_range(i >= 0 && i < X.nrows() && j >= 0 && j < X.ncolumns(),
				"Matrix element access subscripts are out of range.");
#endif
	}

	BCS_ENSURE_INLINE
	inline void check_with_compile_time_dims(bool cond)
	{
		check_arg(cond,
				"Attempted to set a run-time size that is not consistent with compile-time specification.");
	}



	/**
	 * The interfaces for matrix views
	 */
	template<class Derived, typename T>
	class IMatrixView : public IMatrixXpr<Derived, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS_FOR_BASE(Derived, T)
		BCS_CRTP_REF

		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE value_type elem(index_type i, index_type j) const
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE value_type operator() (index_type i, index_type j) const
		{
			check_subscripts_in_range(*this, i, j);
			return elem(i, j);
		}

	}; // end class IDenseMatrixView



	/**
	 * The interfaces for matrix blocks
	 */
	template<class Derived, typename T>
	class IRegularMatrix : public IMatrixView<Derived, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS_FOR_BASE(Derived, T)
		BCS_CRTP_REF

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			check_subscripts_in_range(derived(), i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			check_subscripts_in_range(derived(), i, j);
			return elem(i, j);
		}

	}; // end class IRegularMatrix


	/**
	 * The interfaces for matrix blocks
	 */
	template<class Derived, typename T>
	class IDenseMatrix : public IRegularMatrix<Derived, T>
	{
	public:
		BCS_MAT_TRAITS_DEFS_FOR_BASE(Derived, T)
		BCS_CRTP_REF

	public:
		BCS_ENSURE_INLINE index_type nelems() const
		{
			return derived().nelems();
		}

		BCS_ENSURE_INLINE size_type size() const
		{
			return derived().size();
		}

		BCS_ENSURE_INLINE index_type nrows() const
		{
			return derived().nrows();
		}

		BCS_ENSURE_INLINE index_type ncolumns() const
		{
			return derived().ncolumns();
		}

		BCS_ENSURE_INLINE const_pointer ptr_data() const
		{
			return derived().ptr_data();
		}

		BCS_ENSURE_INLINE pointer ptr_data()
		{
			return derived().ptr_data();
		}

		BCS_ENSURE_INLINE index_t lead_dim() const
		{
			return derived().lead_dim();
		}

		BCS_ENSURE_INLINE const_reference elem(index_type i, index_type j) const
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE reference elem(index_type i, index_type j)
		{
			return derived().elem(i, j);
		}

		BCS_ENSURE_INLINE const_reference operator() (index_type i, index_type j) const
		{
			check_subscripts_in_range(derived(), i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE reference operator() (index_type i, index_type j)
		{
			check_subscripts_in_range(derived(), i, j);
			return elem(i, j);
		}

		BCS_ENSURE_INLINE void resize(index_type m, index_type n)
		{
			derived().resize(m, n);
		}

	}; // end class IDenseMatrixBlock


	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	typename mat_access<Derived>::const_pointer
	col_ptr(const IDenseMatrix<Derived, T>& X, index_t j)
	{
		return X.ptr_data() + X.lead_dim() * j;
	}

	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	typename mat_access<Derived>::pointer
	col_ptr(IDenseMatrix<Derived, T>& X, index_t j)
	{
		return X.ptr_data() + X.lead_dim() * j;
	}

	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	typename mat_access<Derived>::const_pointer
	row_ptr(const IDenseMatrix<Derived, T>& X, index_t j)
	{
		return X.ptr_data() + j;
	}

	template<class Derived, typename T>
	BCS_ENSURE_INLINE
	typename mat_access<Derived>::pointer
	row_ptr(IDenseMatrix<Derived, T>& X, index_t j)
	{
		return X.ptr_data() + j;
	}


	// manipulation functions

	template<typename T, class LMat, class RMat>
	inline bool is_equal(const IMatrixView<LMat, T>& A, const IMatrixView<RMat, T>& B);

	template<typename T, class LMat, class RMat>
	inline bool is_equal(const IDenseMatrix<LMat, T>& A, const IDenseMatrix<RMat, T>& B);

	template<typename T, class LMat, class RMat>
	inline bool is_approx(const IMatrixView<LMat, T>& A, const IMatrixView<RMat, T>& B, const T& tol);

	template<typename T, class LMat, class RMat>
	inline bool is_approx(const IDenseMatrix<LMat, T>& A, const IDenseMatrix<RMat, T>& B, const T& tol);

	template<typename T, class SMat, class DMat>
	inline void copy(const IDenseMatrix<SMat, T>& src, IDenseMatrix<DMat, T>& dst);

	template<typename T, class SMat, class DMat>
	inline void copy(const IMatrixView<SMat, T>& src, IRegularMatrix<DMat, T>& dst);

	template<typename T, class Mat>
	inline void fill(IDenseMatrix<Mat, T>& A, const T& v);

	template<typename T, class Mat>
	inline void zero(IDenseMatrix<Mat, T>& A);

	template<typename T, class Mat>
	void printf_mat(const char *fmt, const IMatrixView<Mat, T>& X,
			const char *pre_line=BCS_NULL, const char *delim="\n");

}


// some useful macros

#define BCS_CREF_NOWRITE { throw std::logic_error("Writing to an CRef-kind object is not allowed"); }

#define BCS_MATRIX_TYPEDEFS0(TName, prefix) \
	typedef TName<double>   prefix##_f64; \
	typedef TName<float>    prefix##_f32; \
	typedef TName<int32_t>  prefix##_i32; \
	typedef TName<uint32_t> prefix##_u32; \
	typedef TName<int16_t>  prefix##_i16; \
	typedef TName<uint16_t> prefix##_u16; \
	typedef TName<int8_t>   prefix##_i8; \
	typedef TName<uint8_t>  prefix##_u8; \
	typedef TName<bool>     prefix##_bool;

#define BCS_MATRIX_TYPEDEFS1(TName, prefix, Dim) \
	typedef TName<double,   Dim> prefix##_f64; \
	typedef TName<float,    Dim> prefix##_f32; \
	typedef TName<int32_t,  Dim> prefix##_i32; \
	typedef TName<uint32_t, Dim> prefix##_u32; \
	typedef TName<int16_t,  Dim> prefix##_i16; \
	typedef TName<uint16_t, Dim> prefix##_u16; \
	typedef TName<int8_t,   Dim> prefix##_i8; \
	typedef TName<uint8_t,  Dim> prefix##_u8; \
	typedef TName<bool,     Dim> prefix##_bool;

#define BCS_MATRIX_TYPEDEFS2(TName, prefix, RDim, CDim) \
	typedef TName<double,   RDim, CDim> prefix##_f64; \
	typedef TName<float,    RDim, CDim> prefix##_f32; \
	typedef TName<int32_t,  RDim, CDim> prefix##_i32; \
	typedef TName<uint32_t, RDim, CDim> prefix##_u32; \
	typedef TName<int16_t,  RDim, CDim> prefix##_i16; \
	typedef TName<uint16_t, RDim, CDim> prefix##_u16; \
	typedef TName<int8_t,   RDim, CDim> prefix##_i8; \
	typedef TName<uint8_t,  RDim, CDim> prefix##_u8; \
	typedef TName<bool,     RDim, CDim> prefix##_bool;



#endif
