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

#if BCSLIB_COMPILER == BCSLIB_MSVC

#include <array>
#include <tuple>
#include <memory>
#include <type_traits>

#elif BCSLIB_COMPILER == BCSLIB_GCC

#include <tr1/array>
#include <tr1/tuple>
#include <tr1/memory>
#include <tr1/type_traits>

#endif

#include <string>
#include <utility>


namespace bcs
{

#ifdef BCS_STDINT_INCLUDED

	using ::uint8_t;
	using ::int8_t;
	using ::uint16_t;
	using ::int16_t;
	using ::uint32_t;
	using ::int32_t;
	using ::uint64_t;
	using ::int64_t;

#else

	typedef signed char int8_t;
	typedef short int16_t;
	typedef int int32_t;

	typedef unsigned char uint8_t;
	typedef unsigned short uint16_t;
	typedef unsigned int uint32_t;

#endif

	using ::ptrdiff_t;
	using ::size_t;

	typedef uint8_t byte;


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


	// copy tags

	struct clone_t { };
	struct own_t { };
	struct ref_t { };


	template<typename T1, typename T2>
	struct ref_bind
	{
		T1& r1;
		T2& r2;

		ref_bind(T1& r1_, T2& r2_) : r1(r1_), r2(r2_) { };

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


	class invalid_argument : public base_exception
	{
	public:
		invalid_argument(const char *msg) : base_exception(msg)
		{
		}
	};



	namespace tr1
	{

		// import tr1 names

		using std::tr1::shared_ptr;
		using std::tr1::array;
		using std::tr1::tuple;

	}

}


#endif
