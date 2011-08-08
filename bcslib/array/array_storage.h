/**
 * @file array_storage.h
 *
 * The class serve as the base for array storage
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_ARRAY_STORAGE_H_
#define BCSLIB_ARRAY_STORAGE_H_

#include <bcslib/base/block.h>
#include <memory>

namespace bcs
{

    struct do_share { };

    template<typename T, class Allocator=aligned_allocator<T> >
    class sharable_storage_base
    {
    public:
    	typedef block<T, Allocator> block_type;

    	sharable_storage_base(size_t n)
    	: m_sp_block(new block_type(n))
    	{
    	}

    	sharable_storage_base(size_t n, const T& v)
    	: m_sp_block(new block_type(n, v))
    	{
    	}

    	sharable_storage_base(size_t n, const T *src)
    	: m_sp_block(new block_type(copy_blk(src, n)))
    	{
    	}

    	sharable_storage_base(const sharable_storage_base& r)
    	: m_sp_block(new block_type(*(r.m_sp_block)))
    	{
    	}

    	sharable_storage_base(sharable_storage_base&& r)
    	: m_sp_block(std::move(r.m_sp_block))
    	{
    	}

    	sharable_storage_base(const sharable_storage_base& r, do_share)
    	: m_sp_block(r.m_sp_block)
    	{
    	}

    public:
    	T* pointer_to_base()
    	{
    		return m_sp_block->pbase();
    	}

    	const T* pointer_to_base() const
    	{
    		return m_sp_block->pbase();
    	}

    public:
    	void operator = (const sharable_storage_base& r)
    	{
    		if (this != &r)
    		{
    			m_sp_block.reset(new block_type(*(r.m_sp_block)));
    		}
    	}

    	void operator = (sharable_storage_base&& r)
    	{
    		if (this != &r)
    		{
    			m_sp_block = std::move(r.m_sp_block);
    		}
    	}

    	void swap(sharable_storage_base& r)
    	{
    		m_sp_block.swap(r.m_sp_block);
    	}

    	bool is_unique() const
    	{
    		return m_sp_block.unique();
    	}

    	void make_unique()
    	{
    		if (!is_unique())
    		{
    			m_sp_block.reset(new block_type(*m_sp_block));
    		}
    	}

    private:
    	std::shared_ptr<block_type> m_sp_block;

    }; // end class sharable_storage_base


}

#endif 
