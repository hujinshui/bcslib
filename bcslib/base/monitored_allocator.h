/**
 * @file monitored_allocator.h
 *
 * The class that implements the monitored allocator
 * 
 * @author Dahua Lin
 */

#ifndef BCSLIB_MONITORED_ALLOCATOR_H
#define BCSLIB_MONITORED_ALLOCATOR_H

#include <bcslib/base/basic_defs.h>
#include <unordered_map>
#include <stdexcept>
#include <limits>

namespace bcs
{

	class memory_allocation_monitor : private noncopyable
	{
	public:
		memory_allocation_monitor()
		: m_pending_bytes(0)
		{
		}

		~memory_allocation_monitor()
		{
		}

		bool has_pending() const
		{
			return !m_ptrmap.empty();
		}

		size_t num_pending_sections() const
		{
			return m_ptrmap.size();
		}

		size_t num_pending_bytes() const
		{
			return m_pending_bytes;
		}

		bool verify(void *p, size_t nbytes) const
		{
			char *pc = static_cast<char*>(p);
			auto it = m_ptrmap.find(pc);
			return it != m_ptrmap.end() && it->second == nbytes;
		}

		void* request(size_t nbytes)
		{
			char *p = new char[nbytes];

			m_ptrmap.insert(make_pair(p, nbytes));
			m_pending_bytes += nbytes;

			return p;
		}

		void release(void *p, size_t nbytes)
		{
			char *pc = static_cast<char*>(p);

			auto it = m_ptrmap.find(pc);
			if (it != m_ptrmap.end())
			{
				if (it->second == nbytes)
				{
					m_ptrmap.erase(it);
					m_pending_bytes -= nbytes;
				}
				else
				{
					throw std::runtime_error("The size to be released does not match the record.");
				}
			}
			else
			{
				throw std::runtime_error("captured invalid pointer to be released.");
			}

			delete[] static_cast<char*>(p);
		}

	private:
		size_t m_pending_bytes;

		std::unordered_map<char*, size_t> m_ptrmap;

	}; // end class memory_allocation_monitor

	extern memory_allocation_monitor global_memory_allocation_monitor;

    template<typename T>
    class monitored_allocator
    {
    public:
    	typedef T value_type;
    	typedef T* pointer;
    	typedef T& reference;
    	typedef const T* const_pointer;
    	typedef const T& const_reference;
    	typedef size_t size_type;
    	typedef ptrdiff_t difference_type;

    	template<typename TOther>
    	struct rebind
    	{
    		typedef monitored_allocator<TOther> other;
    	};

    public:
    	monitored_allocator()
    	{
    	}

    	template<typename U>
    	monitored_allocator(const monitored_allocator<U>& r)
    	{
    	}

    	pointer address( reference x ) const
    	{
    		return &x;
    	}

    	const_pointer address( const_reference x ) const
    	{
    		return &x;
    	}

    	size_type max_size() const
    	{
    		return std::numeric_limits<size_type>::max() / sizeof(value_type);
    	}

    	pointer allocate(size_type n, const void* hint=0)
    	{
    		return static_cast<pointer>(
    				global_memory_allocation_monitor.request(n * sizeof(value_type)));
    	}

    	void deallocate(pointer p, size_type n)
    	{
    		global_memory_allocation_monitor.release(p, n * sizeof(value_type));
    	}

    	void construct (pointer p, const_reference val)
    	{
    		new ((void*)p) value_type(val);
    	}

    	void destroy (pointer p)
    	{
    		p->~value_type();
    	}

    }; // end class monitored_allocator

}

#endif 

