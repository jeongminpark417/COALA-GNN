#pragma once


#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

class seqlock {
    simt::atomic<uint32_t, simt::thread_scope_device>  ticket_;
    simt::atomic<uint32_t, simt::thread_scope_device>  current_;

    public:

    __device__
    unsigned 
    read_lock(){
        unsigned ticket = current_.load(simt::memory_order_acquire);
        return ticket & ~0x1;
    }

    __device__
    unsigned
    read_unlock(){
        return current_.fetch_or(0, simt::memory_order_release);
    }

    __device__
    unsigned 
    read_busy_block(){
        unsigned delay = 8;
        unsigned ticket = current_.load(simt::memory_order_acquire);
        while(ticket & 0x1){
            if(delay < 256) delay <<= 1;
            else delay = 8;

            ticket = current_.load(simt::memory_order_acquire);
        }
        return ticket & ~(0x1U << 1);
    }

    __device__
    unsigned 
    read_busy_unlock(){
        return current_.fetch_or(0, simt::memory_order_release) & ~(0X1U << 1);
    }

    __device__
    void
    busy_wait(unsigned ticket){
        unsigned current = current_.load(simt::memory_order_acquire);
        while(current != ticket){
            //__nanosleep(8);
            current = current_.load(simt::memory_order_acquire);
        }
        current_.fetch_add(1, simt::memory_order_release);
    }

    __device__
    unsigned
    write_lock(){
        unsigned ticket = ticket_.fetch_add(2, simt::memory_order_acquire);
        busy_wait(ticket);
    }

    __device__
    unsigned
    write_unlock(){
        current_.fetch_add(1, simt::memory_order_release);
    }

    __device__
    unsigned
    write_busy_lock(){
        unsigned ticket = ticket_.fetch_add(4, simt::memory_order_acquire);
        busy_wait(ticket);
    }

    __device__
    unsigned
    write_unbusy(){
        current_.fetch_add(1, simt::memory_order_release);
    }

    __device__
    unsigned
    write_busy_unlock(){
        current_.fetch_add(2, simt::memory_order_release);
    }
};