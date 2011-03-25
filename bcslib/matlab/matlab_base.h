/**
 * @file matlab_base.h
 *
 * The basic definitions for matlab port
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_MATLAB_BASE_H
#define BCSLIB_MATLAB_BASE_H

#include <mex.h>

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_mem.h>
#include <bcslib/array/array1d.h>
#include <bcslib/array/array2d.h>

#include <string>

namespace bcs
{

namespace matlab
{

	// typedef for vectors

	typedef const_aview1d<double>   double_vector_cview;
	typedef const_aview1d<float>    single_vector_cview;
	typedef const_aview1d<int32_t>  int32_vector_cview;
	typedef const_aview1d<uint32_t> uint32_vector_cview;
	typedef const_aview1d<int16_t>  int16_vector_cview;
	typedef const_aview1d<uint16_t> uint16_vector_cview;
	typedef const_aview1d<int8_t>   int8_vector_cview;
	typedef const_aview1d<uint8_t>  uint8_vector_cview;
	typedef const_aview1d<bool>     bool_vector_cview;
	typedef const_aview1d<char>     char_vector_cview;

	typedef aview1d<double>   double_vector_view;
	typedef aview1d<float>    single_vector_view;
	typedef aview1d<int32_t>  int32_vector_view;
	typedef aview1d<uint32_t> uint32_vector_view;
	typedef aview1d<int16_t>  int16_vector_view;
	typedef aview1d<uint16_t> uint16_vector_view;
	typedef aview1d<int8_t>   int8_vector_view;
	typedef aview1d<uint8_t>  uint8_vector_view;
	typedef aview1d<bool>     bool_vector_view;
	typedef aview1d<char>     char_vector_view;

	typedef array1d<double>   double_vector;
	typedef array1d<float>    single_vector;
	typedef array1d<int32_t>  int32_vector;
	typedef array1d<uint32_t> uint32_vector;
	typedef array1d<int16_t>  int16_vector;
	typedef array1d<uint16_t> uint16_vector;
	typedef array1d<int8_t>   int8_vector;
	typedef array1d<uint8_t>  uint8_vector;
	typedef array1d<bool>     bool_vector;
	typedef array1d<char>     char_vector;


	// typedef for matrices

	typedef const_aview2d<double, column_major_t>   double_matrix_cview;
	typedef const_aview2d<float, column_major_t>    single_matrix_cview;
	typedef const_aview2d<int32_t, column_major_t>	int32_matrix_cview;
	typedef const_aview2d<uint32_t, column_major_t> uint32_matrix_cview;
	typedef const_aview2d<int16_t, column_major_t>	int16_matrix_cview;
	typedef const_aview2d<uint16_t, column_major_t> uint16_matrix_cview;
	typedef const_aview2d<int8_t, column_major_t>	int8_matrix_cview;
	typedef const_aview2d<uint8_t, column_major_t>  uint8_matrix_cview;
	typedef const_aview2d<bool, column_major_t>     bool_matrix_cview;
	typedef const_aview2d<char, column_major_t>     char_matrix_cview;

	typedef aview2d<double, column_major_t>   double_matrix_view;
	typedef aview2d<float, column_major_t>    single_matrix_view;
	typedef aview2d<int32_t, column_major_t>  int32_matrix_view;
	typedef aview2d<uint32_t, column_major_t> uint32_matrix_view;
	typedef aview2d<int16_t, column_major_t>  int16_matrix_view;
	typedef aview2d<uint16_t, column_major_t> uint16_matrix_view;
	typedef aview2d<int8_t, column_major_t>	  int8_matrix_view;
	typedef aview2d<uint8_t, column_major_t>  uint8_matrix_view;
	typedef aview2d<bool, column_major_t>     bool_matrix_view;
	typedef aview2d<char, column_major_t>     char_matrix_view;

	typedef array2d<double, column_major_t>   double_matrix;
	typedef array2d<float, column_major_t>    single_matrix;
	typedef array2d<int32_t, column_major_t>  int32_matrix;
	typedef array2d<uint32_t, column_major_t> uint32_matrix;
	typedef array2d<int16_t, column_major_t>  int16_matrix;
	typedef array2d<uint16_t, column_major_t> uint16_matrix;
	typedef array2d<int8_t, column_major_t>	  int8_matrix;
	typedef array2d<uint8_t, column_major_t>  uint8_matrix;
	typedef array2d<bool, column_major_t>     bool_matrix;
	typedef array2d<char, column_major_t>     char_matrix;


	// exception class

	class mexception
	{
	public:
		mexception(const char *id, const char *msg) : m_identifier(id), m_message(msg)
		{
		}

		const char *identifier() const
		{
			return m_identifier.c_str();
		}

		const char *message() const
		{
			return m_message.c_str();
		}

	private:
		std::string m_identifier;
		std::string m_message;
	};



	// type specific stuff

	template<typename T> struct mtype_traits;

	template<>
	struct mtype_traits<double>
	{
		static const bool is_float = true;
		static const bool is_numeric = true;
		static const mxClassID class_id = mxDOUBLE_CLASS;
	};

	template<>
	struct mtype_traits<float>
	{
		static const bool is_float = true;
		static const bool is_numeric = true;
		static const mxClassID class_id = mxSINGLE_CLASS;
	};

	template<>
	struct mtype_traits<int32_t>
	{
		static const bool is_float = false;
		static const bool is_numeric = true;
		static const mxClassID class_id = mxINT32_CLASS;
	};

	template<>
	struct mtype_traits<uint32_t>
	{
		static const bool is_float = false;
		static const bool is_numeric = true;
		static const mxClassID class_id = mxUINT32_CLASS;
	};

	template<>
	struct mtype_traits<int16_t>
	{
		static const bool is_float = false;
		static const bool is_numeric = true;
		static const mxClassID class_id = mxINT16_CLASS;
	};

	template<>
	struct mtype_traits<uint16_t>
	{
		static const bool is_float = false;
		static const bool is_numeric = true;
		static const mxClassID class_id = mxUINT16_CLASS;
	};

	template<>
	struct mtype_traits<int8_t>
	{
		static const bool is_float = false;
		static const bool is_numeric = true;
		static const mxClassID class_id = mxINT8_CLASS;
	};

	template<>
	struct mtype_traits<uint8_t>
	{
		static const bool is_float = false;
		static const bool is_numeric = true;
		static const mxClassID class_id = mxUINT8_CLASS;
	};

	template<>
	struct mtype_traits<bool>
	{
		static const bool is_float = false;
		static const bool is_numeric = false;
		static const mxClassID class_id = mxLOGICAL_CLASS;
	};


}


}

#endif 
