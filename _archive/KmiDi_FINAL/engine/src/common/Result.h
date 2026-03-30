#pragma once

/**
 * Result.h - RT-Safe Result Type for Error Handling
 * ==================================================
 *
 * Provides Result<T, Error> type for RT-safe error handling without exceptions.
 * Similar to Rust's Result type or std::expected (C++23).
 */

#include <type_traits>
#include <utility>
#include <string>

namespace kelly {

/**
 * Result<T, Error> - RT-safe error handling
 *
 * Represents either a success value of type T or an error of type Error.
 * No exceptions, safe for real-time audio callbacks.
 */
template<typename T, typename Error>
class Result {
public:
    // Constructors
    Result() = delete; // Must use success() or error()

    // Move constructor
    Result(Result&& other) noexcept
        : has_value_(other.has_value_)
        , value_(std::move(other.value_))
        , error_(std::move(other.error_))
    {
        other.has_value_ = false;
    }

    // Move assignment
    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            destroy();
            has_value_ = other.has_value_;
            value_ = std::move(other.value_);
            error_ = std::move(other.error_);
            other.has_value_ = false;
        }
        return *this;
    }

    // Destructor
    ~Result() {
        destroy();
    }

    // Factory methods
    static Result success(T&& value) {
        Result result;
        result.has_value_ = true;
        new (&result.value_) T(std::move(value));
        return result;
    }

    static Result success(const T& value) {
        Result result;
        result.has_value_ = true;
        new (&result.value_) T(value);
        return result;
    }

    static Result error(Error&& error) {
        Result result;
        result.has_value_ = false;
        new (&result.error_) Error(std::move(error));
        return result;
    }

    static Result error(const Error& error) {
        Result result;
        result.has_value_ = false;
        new (&result.error_) Error(error);
        return result;
    }

    // Check if result has value
    bool hasValue() const noexcept {
        return has_value_;
    }

    // Get value (undefined if hasValue() == false)
    const T& value() const& {
        return value_;
    }

    T& value() & {
        return value_;
    }

    T&& value() && {
        return std::move(value_);
    }

    // Get error (undefined if hasValue() == true)
    const Error& error() const& {
        return error_;
    }

    Error& error() & {
        return error_;
    }

    // Operator bool (true if has value)
    explicit operator bool() const noexcept {
        return has_value_;
    }

    // Dereference operator
    const T& operator*() const& {
        return value_;
    }

    T& operator*() & {
        return value_;
    }

    // Arrow operator
    const T* operator->() const {
        return &value_;
    }

    T* operator->() {
        return &value_;
    }

private:
    bool has_value_ = false;
    union {
        T value_;
        Error error_;
    };

    void destroy() {
        if (has_value_) {
            value_.~T();
        } else {
            error_.~Error();
        }
    }
};

// Specialization for void
template<typename Error>
class Result<void, Error> {
public:
    Result() = delete;

    Result(Result&& other) noexcept
        : has_value_(other.has_value_)
        , error_(std::move(other.error_))
    {
        other.has_value_ = false;
    }

    Result& operator=(Result&& other) noexcept {
        if (this != &other) {
            if (!has_value_) {
                error_.~Error();
            }
            has_value_ = other.has_value_;
            error_ = std::move(other.error_);
            other.has_value_ = false;
        }
        return *this;
    }

    ~Result() {
        if (!has_value_) {
            error_.~Error();
        }
    }

    static Result success() {
        Result result;
        result.has_value_ = true;
        return result;
    }

    static Result error(Error&& error) {
        Result result;
        result.has_value_ = false;
        new (&result.error_) Error(std::move(error));
        return result;
    }

    static Result error(const Error& error) {
        Result result;
        result.has_value_ = false;
        new (&result.error_) Error(error);
        return result;
    }

    bool hasValue() const noexcept {
        return has_value_;
    }

    const Error& error() const& {
        return error_;
    }

    Error& error() & {
        return error_;
    }

    explicit operator bool() const noexcept {
        return has_value_;
    }

private:
    bool has_value_ = false;
    union {
        Error error_;
    };
};

} // namespace kelly
