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


struct seqlock {
    simt::atomic<uint64_t, simt::thread_scope_device>  sequence;
    public:
    __device__
    void write_begin() {
        unsigned out = sequence.fetch_add(1, simt::memory_order_acq_rel); // make it odd
    }
    __device__
    void write_end() {
        sequence.fetch_add(1, simt::memory_order_release); // make it even
    }
    __device__
    uint64_t read_begin() const {
        uint64_t seq;
        do {
            seq = sequence.load(simt::memory_order_acquire);
        } while (seq & 1);  // wait if odd (writer active)
        return seq;
    }
    __device__
    bool read_retry(uint32_t start_seq) const {
        return sequence.load(simt::memory_order_acquire) != start_seq;
    }
};

