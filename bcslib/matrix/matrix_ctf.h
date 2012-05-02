/**
 * @file matrix_ctf.h
 *
 * Compile-time reflection tools for matrix library
 *
 * @author Dahua Lin
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BCSLIB_MATRIX_CTF_H_
#define BCSLIB_MATRIX_CTF_H_

#include <bcslib/matrix/matrix_fwd.h>
#include <bcslib/core/type_traits.h>

namespace bcs
{

	/********************************************
	 *
	 *  Dimension related tools
	 *
	 ********************************************/

	template<class Mat>
	struct ct_rows
	{
		static const int value = matrix_traits<Mat>::compile_time_num_rows;
	};

	template<class Mat>
	struct ct_cols
	{
		static const int value = matrix_traits<Mat>::compile_time_num_cols;
	};

	template<class Mat>
	struct ct_size
	{
		static const int value = ct_rows<Mat>::value * ct_cols<Mat>::value;
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

	template<class Mat1, class Mat2>
	struct binary_ct_size
	{
		static const int value = binary_ct_rows<Mat1, Mat2>::value * binary_ct_cols<Mat1, Mat2>::value;
	};


	template<class Mat>
	struct ct_is_row
	{
		static const bool value = ct_rows<Mat>::value == 1;
	};

	template<class Mat>
	struct ct_is_col
	{
		static const bool value = ct_cols<Mat>::value == 1;
	};

	template<class Mat>
	struct ct_is_vector
	{
		static const bool value = ct_is_row<Mat>::value && ct_is_col<Mat>::value;
	};

	template<class Mat>
	struct has_static_nrows
	{
		static const bool value = ct_rows<Mat>::value > 0;
	};

	template<class Mat>
	struct has_static_ncols
	{
		static const bool value = ct_cols<Mat>::value > 0;
	};

	template<class Mat>
	struct has_dynamic_nrows
	{
		static const bool value = ct_rows<Mat>::value == DynamicDim;
	};

	template<class Mat>
	struct has_dynamic_ncols
	{
		static const bool value = ct_cols<Mat>::value == DynamicDim;
	};


	template<class Mat>
	struct has_static_size
	{
		static const bool value = has_static_nrows<Mat>::value && has_static_ncols<Mat>::value;
	};

	template<class LMat, class RMat>
	struct ct_has_same_nrows
	{
		static const bool value =
				ct_rows<LMat>::value > 0 &&
				ct_rows<LMat>::value == ct_rows<RMat>::value;
	};

	template<class LMat, class RMat>
	struct ct_has_same_ncols
	{
		static const bool value =
				ct_cols<LMat>::value > 0 &&
				ct_cols<LMat>::value == ct_cols<RMat>::value;
	};


	/********************************************
	 *
	 *  Concept test tools
	 *
	 ********************************************/

	template<class T>
	struct is_supported_matrix_value_type
	{
		static const bool value = bcs::is_pod<T>::value;
	};

	template<class Derived, template<class D, typename T> class Interface>
	struct has_matrix_interface
	{
		typedef Interface<Derived, typename matrix_traits<Derived>::value_type> expect_base;
		static const bool value = is_base_of<expect_base, Derived>::value;
	};

	template<class Mat>
	struct is_mat_xpr
	{
		static const bool value = has_matrix_interface<Mat, IMatrixXpr>::value;
	};

	template<class Mat>
	struct is_mat_view
	{
		static const bool value = has_matrix_interface<Mat, IMatrixView>::value;
	};

	template<class Mat>
	struct is_dense_mat
	{
		static const bool value = has_matrix_interface<Mat, IDenseMatrix>::value;
	};


	/********************************************
	 *
	 *  Matrix access and manipulation
	 *
	 ********************************************/

	template<class Mat>
	struct is_readonly_mat
	{
		static const bool value = matrix_traits<Mat>::is_readonly;
	};

	template<class Mat>
	struct is_resizable_mat
	{
		static const bool value = matrix_traits<Mat>::is_resizable;
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


	/********************************************
	 *
	 *  Default attributes
	 *
	 ********************************************/

	template<class Mat>
	struct has_continuous_layout { static const bool value = false; };

	template<class Mat>
	struct is_continuous_mat
	{
		static const bool value = is_dense_mat<Mat>::value && has_continuous_layout<Mat>::value;
	};

	template<class Mat>
	struct is_always_aligned { static const bool value = false; };

	template<class Mat>
	struct is_linear_accessible { static const bool value = false; };

	template<class Mat>
	struct is_readable_as_single_vector
	{
		static const bool value = is_linear_accessible<Mat>::value;
	};

	template<class Mat>
	struct is_readable_colwise
	{
		static const bool value = is_dense_mat<Mat>::value;
	};


	template<class Mat>
	struct is_accessible_as_single_vector
	{
		static const bool value =
				is_linear_accessible<Mat>::value &&
				!is_readonly_mat<Mat>::value;
	};


	template<class Mat>
	struct is_accessible_colwise
	{
		static const bool value =
				is_readable_colwise<Mat>::value &&
				!is_readonly_mat<Mat>::value;
	};


	/********************************************
	 *
	 *  Functional dispatchers
	 *
	 ********************************************/

	template<class Mat> struct subview_dispatcher;
	template<class Mat> struct transpose_dispatcher;

}

#endif /* MATRIX_CTF_H_ */





