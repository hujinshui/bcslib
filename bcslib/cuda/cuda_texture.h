/**
 * @file  cuda_texture.h
 *
 * The classes to help texture binding
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_CUDA_TEXTURE_H_
#define BCSLIB_CUDA_TEXTURE_H_

#include <bcslib/cuda/cuda_base.h>
#include <bcslib/cuda/cuda_mat.h>

namespace bcs { namespace cuda {


	/**
	 * The class to represent a binding between
	 * a texture and a device memory block
	 */
	class texture_binding : private noncopyable
	{
	public:
		template<typename T>
		texture_binding(const textureReference& tex, device_cview2d<T> a, const cudaChannelFormatDesc& desc)
		: m_texture_ref(tex), m_offset(0)
		{
			cudaError_t ret = ::cudaBindTexture2D(&m_offset, &tex,
					(const void*)a.pbase().get(), &desc,
					(size_t)a.width(), (size_t)a.height(), (size_t)a.pitch());
		}

		template<typename T, int dim, cudaTextureReadMode readMode>
		texture_binding(const texture<T, dim, readMode>& tex, device_cview2d<T> a)
		: m_texture_ref(tex), m_offset(0)
		{
			cudaError_t ret = ::cudaBindTexture2D(&m_offset, tex,
					(const void*)a.pbase().get(),
					(size_t)a.width(), (size_t)a.height(), (size_t)a.pitch());

			if (ret != cudaSuccess)
			{
				throw cuda_error(ret);
			}
		}

		~texture_binding()
		{
			::cudaUnbindTexture(&m_texture_ref);
		}

	public:
		const textureReference& texture_ref() const
		{
			return m_texture_ref;
		}

		size_t offset() const
		{
			return m_offset;
		}

	private:
		const textureReference& m_texture_ref;
		size_t m_offset;

	}; // end class texture_binding


} }

#endif 
