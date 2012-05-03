/**
 * @file matrix_fwd.h
 *
 * A series of forward declarations for matrices
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_FWD_H_
#define BCSLIB_MATRIX_FWD_H_

#include <bcslib/core/basic_defs.h>

namespace bcs
{
	const int DynamicDim = 0;

	// forward declaration of concepts


	/****************************************************************
	 *
	 *   A specialized version of matrix_traits<C> must be provided,
	 *   which should contain the following static members:
	 *
	 *   - num_dimensions:	an int value, which must be set to 2
	 *   					(reserved for future extension)
	 *
	 *   - compile_time_num_rows:	compile-time number of rows
	 *   - compile_time_num_cols:	compile-time number of columns
	 *   - is_readonly:				whether the contents can be modified
	 *   - is_resizable:			whether the dimensions can be changed
	 *   						    at run-time
	 *
	 *	 - value_type:			the type of element value
	 *	 - index_type:			the (default) type of index
	 *
	 ****************************************************************/

	template<class Derived> struct matrix_traits;

	template<class Derived, typename T> class IMatrixXpr;
	template<class Derived, typename T> class IMatrixView;
	template<class Derived, typename T> class IDenseMatrix;

	// forward declaration of some important types

	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class dense_matrix; // guranteed to be aligned
	template<typename T, int CTRows=DynamicDim> class dense_col;
	template<typename T, int CTCols=DynamicDim> class dense_row;

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class cref_matrix;
	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class ref_matrix;

	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class aligned_cref_matrix;
	template<typename T, int RowDim=DynamicDim, int ColDim=DynamicDim> class aligned_ref_matrix;

	template<typename T, int CTRows=DynamicDim> class cref_col;
	template<typename T, int CTRows=DynamicDim> class ref_col;
	template<typename T, int CTCols=DynamicDim> class cref_row;
	template<typename T, int CTCols=DynamicDim> class ref_row;

	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class cref_matrix_ex;
	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class ref_matrix_ex;

	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class cref_grid2d;
	template<typename T, int CTRows=DynamicDim, int CTCols=DynamicDim> class ref_grid2d;


	// important structures for manipulation

	template<class Mat> struct subviews;

	// important structures for evaluation

	template<class Expr> struct expr_optimizer;
	template<class Expr> struct expr_evaluator;

	template<class Expr> struct vec_reader;  	// dispatch Expr --> linear vector reader
	template<class Expr> struct vec_accessor;	// dispatch Expr --> linear vector accessor

	template<class Expr> struct colwise_reader_set;		// dispatch Expr --> column-wise reader set
	template<class Expr> struct colwise_accessor_set; 	// dispatch Expr --> column-wise accessor set


	// forward declaration of useful memory operations

	template<typename T, class LMat, class RMat>
	inline bool is_equal(const IMatrixView<LMat, T>& A, const IMatrixView<RMat, T>& B);

	template<typename T, class LMat, class RMat>
	inline bool is_approx(const IMatrixView<LMat, T>& A, const IMatrixView<RMat, T>& B, const T tol);

	template<typename T, class SExpr, class DMat>
	inline void evaluate_to(const IMatrixXpr<SExpr, T>& src, IDenseMatrix<DMat, T>& dst);

	template<typename T, class SMat, class DMat>
	inline void copy(const IMatrixView<SMat, T>& src, IDenseMatrix<DMat, T>& dst);

	template<typename T, class DMat>
	inline void fill(IDenseMatrix<DMat, T>& dst, const T& v);

	template<typename T, class DMat>
	inline void zero(IDenseMatrix<DMat, T>& dst);

}

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



#endif /* MATRIX_FWD_H_ */





