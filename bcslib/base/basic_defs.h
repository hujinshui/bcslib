/**
 * @file basic_defs.h
 *
 * Library-wide basic definitions
 *
 * @author dhlin
 */


#ifndef BCSLIB_BASIC_DEFS_H
#define BCSLIB_BASIC_DEFS_H

#include <bcslib/base/config.h>
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <utility>


namespace bcs
{

	using ::uint8_t;
	using ::int8_t;
	using ::uint16_t;
	using ::int16_t;
	using ::uint32_t;
	using ::int32_t;
	using ::uint64_t;
	using ::int64_t;

	using ::ptrdiff_t;
	using ::size_t;

	typedef uint8_t byte;

	template<typename T>
	inline T* null_p()
	{
		return (T*)(0);
	}


	// value associated with index (particularly useful for sorting or re-arrangement)

	template<typename T, typename TIndex=uint32_t>
	struct indexed_entry
	{
		typedef T value_type;
		typedef TIndex index_type;

		value_type value;
		index_type index;

		void set(const value_type& v, const index_type& i)
		{
			value = v;
			index = i;
		}
	};


	template<typename T, typename TIndex>
	inline bool operator == (const indexed_entry<T, TIndex>& lhs, const indexed_entry<T, TIndex>& rhs)
	{
		return lhs.value == rhs.value;
	}

	template<typename T, typename TIndex>
	inline bool operator != (const indexed_entry<T, TIndex>& lhs, const indexed_entry<T, TIndex>& rhs)
	{
		return lhs.value != rhs.value;
	}

	template<typename T, typename TIndex>
	inline bool operator < (const indexed_entry<T, TIndex>& lhs, const indexed_entry<T, TIndex>& rhs)
	{
		return lhs.value < rhs.value;
	}

	template<typename T, typename TIndex>
	inline bool operator <= (const indexed_entry<T, TIndex>& lhs, const indexed_entry<T, TIndex>& rhs)
	{
		return lhs.value <= rhs.value;
	}

	template<typename T, typename TIndex>
	inline bool operator > (const indexed_entry<T, TIndex>& lhs, const indexed_entry<T, TIndex>& rhs)
	{
		return lhs.value > rhs.value;
	}

	template<typename T, typename TIndex>
	inline bool operator >= (const indexed_entry<T, TIndex>& lhs, const indexed_entry<T, TIndex>& rhs)
	{
		return lhs.value >= rhs.value;
	}


	template<typename T1, typename T2>
	struct ref_bind
	{
		T1& r1;
		T2& r2;

		ref_bind(T1& r1_, T2& r2_) : r1(r1_), r2(r2_) { }

		ref_bind& operator = (const std::pair<T1, T2>& in)
		{
			r1 = in.first;
			r2 = in.second;

			return *this;
		}
	};

	template<typename T1, typename T2>
	inline ref_bind<T1, T2> rbind(T1& r1, T2& r2)
	{
		return ref_bind<T1, T2>(r1, r2);
	}


	class base_exception
	{
	public:
		virtual ~base_exception() { }

		base_exception(const char *msg) : m_message(msg)
		{
		}

		const char *message() const
		{
			return m_message.c_str();
		}

	private:
		std::string m_message;
	};

}

#endif



