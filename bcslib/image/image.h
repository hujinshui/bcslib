/**
 * @file image.h
 *
 * The class to represent an image
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_IMAGE_H
#define BCSLIB_IMAGE_H

#include <bcslib/image/image_base.h>

namespace bcs
{

	template<typename TPixel>
	class const_imview
	{
	public:
		typedef TPixel pixel_type;
		typedef typename bcs::pixel_traits<TPixel>::value_type value_type;
		static const size_t num_channels = bcs::pixel_traits<TPixel>::num_channels;

	public:
		const_imview(const pixel_type *base, size_t w, size_t h, size_t stride)
		: m_base(const_cast<uint8_t*>((const uint8_t*)base))
		, m_width(w), m_height(h), m_stride(stride)
		{
		}

		const_imview(const pixel_type *base, size_t w, size_t h)
		: m_base(const_cast<uint8_t*>((const uint8_t*)base))
		, m_width(w), m_height(h), m_stride(w * sizeof(pixel_type))
		{
		}

		size_t width() const
		{
			return m_width;
		}

		size_t height() const
		{
			return m_height;
		}

		size_t stride() const
		{
			return m_stride;
		}

		size_t npixels() const
		{
			return m_width * m_height;
		}

		bool with_dense_layout() const
		{
			return m_stride == m_width * sizeof(pixel_type);
		}

	public:

		const uint8_t *byte_base() const
		{
			return m_base;
		}

		const pixel_type *pixel_base() const
		{
			return (const pixel_type*)m_base;
		}

		const pixel_type *prow(img_index_t i) const
		{
			return (const pixel_type*)(m_base + i * m_stride);
		}

		const pixel_type *ptr(img_index_t i, img_index_t j) const
		{
			return prow(i) + j;
		}

		const pixel_type& operator() (img_index_t i, img_index_t j) const
		{
			return *(ptr(i, j));
		}

	protected:
		uint8_t *m_base;
		size_t m_width;
		size_t m_height;
		size_t m_stride;  // # of bytes per row

	}; // end class const_imview


	template<typename TPixel>
	class imview : public const_imview<TPixel>
	{
	public:
		typedef TPixel pixel_type;
		typedef typename bcs::pixel_traits<TPixel>::value_type value_type;
		static const size_t num_channels = bcs::pixel_traits<TPixel>::num_channels;

	public:
		imview(const pixel_type *base, size_t w, size_t h, size_t stride)
		: const_imview<TPixel>(base, w, h, stride)
		{
		}

		imview(const pixel_type *base, size_t w, size_t h)
		: const_imview<TPixel>(base, w, h)
		{
		}

		const uint8_t *byte_base() const
		{
			return this->m_base;
		}

		uint8_t *byte_base()
		{
			return this->m_base;
		}

		const pixel_type *pixel_base() const
		{
			return (const pixel_type*)(this->m_base);
		}

		pixel_type *pixel_base()
		{
			return (pixel_type*)(this->m_base);
		}

		const pixel_type *prow(img_index_t i) const
		{
			return (const pixel_type*)(this->m_base + i * this->m_stride);
		}

		pixel_type *prow(img_index_t i)
		{
			return (pixel_type*)(this->m_base + i * this->m_stride);
		}

		const pixel_type *ptr(img_index_t i, img_index_t j) const
		{
			return this->prow(i) + j;
		}

		pixel_type *ptr(img_index_t i, img_index_t j)
		{
			return this->prow(i) + j;
		}

		const pixel_type& operator() (img_index_t i, img_index_t j) const
		{
			return *(this->ptr(i, j));
		}

		pixel_type& operator() (img_index_t i, img_index_t j)
		{
			return *(this->ptr(i, j));
		}

	}; // end class imview


	template<typename TPixel1, typename TPixel2>
	inline bool is_same_size(const const_imview<TPixel1>& lhs, const const_imview<TPixel2>& rhs)
	{
		return lhs.width() == rhs.width() && lhs.height() == rhs.height();
	}


}

#endif 
