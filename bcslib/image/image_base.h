/**
 * @file image_base.h
 *
 * The basic definitions for images
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_IMAGE_BASE_H
#define BCSLIB_IMAGE_BASE_H

#include <bcslib/base/basic_defs.h>
#include <bcslib/base/basic_mem.h>
#include <bcslib/geometry/geometry_base.h>

namespace bcs
{
	typedef geo_index_t img_index_t;


	// multi-channel pixel

	template<typename T, size_t N>
	struct mcpixel
	{
		typedef T value_type;
		static const size_t num_channels = N;

		const value_type& operator[] (size_t i) const
		{
			return values[i];
		}

		value_type& operator[] (size_t i)
		{
			return values[i];
		}

		value_type values[N];
	};

	template<typename T>
	inline bool operator == (const mcpixel<T, 1>& lhs, const mcpixel<T, 1>& rhs)
	{
		return lhs[0] == rhs[0];
	}

	template<typename T>
	inline bool operator == (const mcpixel<T, 2>& lhs, const mcpixel<T, 2>& rhs)
	{
		return lhs[0] == rhs[0] && lhs[1] == rhs[1];
	}

	template<typename T>
	inline bool operator == (const mcpixel<T, 3>& lhs, const mcpixel<T, 3>& rhs)
	{
		return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
	}

	template<typename T>
	inline bool operator == (const mcpixel<T, 4>& lhs, const mcpixel<T, 4>& rhs)
	{
		return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] && lhs[3] == rhs[3];
	}

	template<typename T, size_t N>
	inline bool operator == (const mcpixel<T, N>& lhs, const mcpixel<T, N>& rhs)
	{
		return elements_equal(lhs.values, rhs.values, N);
	}

	template<typename T, size_t N>
	inline bool operator != (const mcpixel<T, N>& lhs, const mcpixel<T, N>& rhs)
	{
		return !(lhs == rhs);
	}



	// pixel traits

	template<typename TPixel> struct pixel_traits;

	template<>
	struct pixel_traits<uint8_t>
	{
		typedef uint8_t value_type;
		static const size_t num_channels = 1;
	};

	template<>
	struct pixel_traits<int8_t>
	{
		typedef int8_t value_type;
		static const size_t num_channels = 1;
	};


	template<>
	struct pixel_traits<uint16_t>
	{
		typedef uint16_t value_type;
		static const size_t num_channels = 1;
	};

	template<>
	struct pixel_traits<int16_t>
	{
		typedef int16_t value_type;
		static const size_t num_channels = 1;
	};


	template<>
	struct pixel_traits<uint32_t>
	{
		typedef uint32_t value_type;
		static const size_t num_channels = 1;
	};

	template<>
	struct pixel_traits<int32_t>
	{
		typedef int32_t value_type;
		static const size_t num_channels = 1;
	};

	template<typename T, size_t N>
	struct pixel_traits<mcpixel<T, N> >
	{
		typedef T value_type;
		static const size_t num_channels = N;
	};




}

#endif 
