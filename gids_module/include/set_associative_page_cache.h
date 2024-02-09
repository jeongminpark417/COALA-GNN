#ifndef __SET_ASSOCIATIVE_PAGE_CACHE_H__
#define __SET_ASSOCIATIVE_PAGE_CACHE_H__

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include "util.h"
#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "buffer.h"
#include "ctrl.h"
#include <iostream>
#include "nvm_parallel_queue.h"
#include "nvm_cmd.h"
//#include "window_buffer.h"

#define  RR 0x01
#define  EP_WINDOW 0x02

#define FULL_MASK 0xFFFFFFFF
#define FF_64 0xFFFFFFFFFFFFFFFF
#define FF_48 0xFFFFFFFFFFFFFF
#define FF_16 0xFFFF

template <typename T = float>
 __forceinline__
__device__
void warp_memcpy(void* src, void* dst, size_t size, uint32_t mask){
     T* src_ptr = (T*) src;
     T* dst_ptr = (T*) dst;
     
     uint32_t count = __popc(mask);
     uint32_t lane_id = threadIdx.x %32;
     
     uint32_t my_id = count - (__popc(mask>>(lane_id)));
     for(; my_id < size/sizeof(T); my_id += count){
          dst_ptr[my_id] =  src_ptr[my_id]; 
     }
 }



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
            __nanosleep(8);
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


class Victim_Buffer {

    public:
    uint32_t num_buffers;
    uint32_t buffer_depth;
    uint64_t* queue_counters;

    //Pinned in CPU Memory
    // 2D Array (Number of buffers, depth of buffer)
    uint64_t* index_arrays;
    // 2D Array (Number of buffers, depth of buffer * CL)
    uint64_t* data_arrays;

    public:
    __device__ __host__
    Victim_Buffer(){
        num_buffers = 0;
        buffer_depth = 0;
        queue_counters = nullptr;
        index_arrays = nullptr;
        data_arrays = nullptr;
    }
    __device__ __host__
    Victim_Buffer(uint32_t n_buffer, uint32_t b_depth, uint64_t* q_counters, uint64_t* i_array, uint64_t* d_array) :
        num_buffers(n_buffer),
        buffer_depth(b_depth),
        queue_counters(q_counters),
        index_arrays(i_array),
        data_arrays(d_array)
    {}

    __device__ __host__
    void
    init(uint32_t n_buffer, uint32_t b_depth, uint64_t* q_counters, uint64_t* i_array, uint64_t* d_array) {
        num_buffers = n_buffer;
        buffer_depth = b_depth;
        queue_counters = q_counters;
        index_arrays = i_array;
        data_arrays = d_array;
    }

    // Only supports warp-level memcpy
    __device__
    void
    evict_to_pvp(uint64_t evicted_batch_index, uint64_t* cl_src_ptr, uint64_t cl_size, uint64_t reuse_val, uint32_t head_ptr, unsigned tid){
        uint32_t q_id = (reuse_val + head_ptr) % num_buffers;
        uint64_t buffer_idx  = 0;
        if(tid == 0){
            buffer_idx = atomicAdd((unsigned long long*)queue_counters + q_id, (unsigned long long)1);
            if(buffer_idx < buffer_depth){
                printf("buffer_depth:%llu\n", buffer_depth);
                index_arrays[buffer_idx + buffer_depth * q_id] = evicted_batch_index;
            }
        }
        buffer_idx = __shfl_sync(FULL_MASK, buffer_idx, 0);

        if(buffer_idx < buffer_depth){
            uint64_t ele_per_CL = cl_size / sizeof(uint64_t);
            uint64_t ele_per_buffer = ele_per_CL * buffer_depth;

            uint64_t queue_offset =  q_id * ele_per_buffer + buffer_idx * ele_per_CL;

            warp_memcpy<uint64_t>(cl_src_ptr, (void*)(data_arrays + queue_offset), cl_size, FULL_MASK );
        }
    }
  

    __host__
    void prefetch_from_victim_queue(uint64_t* PVP_pinned_data, uint64_t* PVP_pinned_idx, uint64_t cl_size, uint64_t num_evicted_cl, uint32_t head_ptr, cudaStream_t stream){
    

        uint64_t* wb_queue_head = data_arrays + cl_size / sizeof(uint64_t) * buffer_depth * head_ptr;
        uint64_t* wb_id_array_head = index_arrays + buffer_depth * head_ptr;

        cudaMemcpy(PVP_pinned_data, wb_queue_head, num_evicted_cl * cl_size, cudaMemcpyHostToDevice);	
        cudaMemcpy(PVP_pinned_idx, wb_id_array_head, num_evicted_cl * sizeof(uint64_t), cudaMemcpyHostToDevice);	
       // cudaMemcpyAsync(PVP_pinned_data, wb_queue_head, num_evicted_cl* cl_size, cudaMemcpyHostToDevice, stream);	
       // cudaMemcpyAsync(PVP_pinned_idx, wb_id_array_head, tnum_evicted_cl * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);	
    }
};


template<typename T>
struct SA_cache_d_t {

    //Victim_Buffer victim_buffer;

    //public:
    uint32_t num_buffers;
    uint32_t buffer_depth;
    uint64_t* queue_counters;

    unsigned num_gpus = 1;

    // uint64_t* index_arrays;
    // // 2D Array (Number of buffers, depth of buffer * CL)
    // uint64_t* data_arrays;



    //Pinned in CPU Memory
    // 2D Array (Number of buffers, depth of buffer)
    uint64_t* index_arrays_;
    // 2D Array (Number of buffers, depth of buffer * CL)
    uint64_t* data_arrays_;

    seqlock* set_locks_;
    seqlock* way_locks_;

    uint64_t num_sets;
    uint64_t num_ways;
    uint64_t CL_SIZE;


    uint64_t* keys_;
    uint32_t* set_cnt_;

    //Meta data for window buffering
    //(2B for Reuse Val, 1B for GPU_ID, 5B for batch Index)
    uint64_t* next_reuse_;

    unsigned int my_GPU_id_;

    //need to be changed (RR : RoundRobin, EP_WINDOW: Window Buffering)
    //uint8_t evict_policy = EP_WINDOW;
    uint8_t evict_policy = RR;


    bool use_WB;
    bool use_PVP;

    //nvme controllers
    Controller** d_ctrls_;
    uint32_t n_ctrls_;

    uint64_t n_blocks_per_page_;

    uint8_t* base_addr_;
    uint64_t* prp1_;                  
    uint64_t* prp2_;   
    bool prps_;


    simt::atomic<uint64_t, simt::thread_scope_device>* q_head;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_tail;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_lock;
    simt::atomic<uint64_t, simt::thread_scope_device>* extra_reads;


    simt::atomic<uint64_t, simt::thread_scope_device> double_read;
    simt::atomic<uint64_t, simt::thread_scope_device> hit_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> miss_cnt;



    SA_cache_d_t(seqlock* set_locks, seqlock* way_locks,uint64_t n_sets, uint64_t n_ways, uint64_t cl_size, uint64_t* keys, uint32_t* set_cnt,
                 Controller** d_ctrls, uint32_t n_ctrls, uint64_t n_blocks_per_page, uint8_t* base_addr, uint64_t* prp1, uint64_t* prp2, bool prps,
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_head, simt::atomic<uint64_t, simt::thread_scope_device>* queue_tail, 
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_lock, simt::atomic<uint64_t, simt::thread_scope_device>* queue_extra_reads,
                uint64_t* next_reuse, bool WB_flag, bool PVP_flag,
                // Victim_Buffer VB,
                uint32_t n_buffers, uint32_t b_depth, uint64_t* q_counters,   uint64_t* index_arrays, uint64_t* data_arrays,
                unsigned int my_GPU_id, unsigned n_gpus
                 ) :
        set_locks_(set_locks),
        way_locks_(way_locks),
        num_sets(n_sets),
        num_ways(n_ways),
        CL_SIZE(cl_size),
        keys_(keys),
        set_cnt_(set_cnt),
        base_addr_(base_addr),
        prp1_(prp1),
        prp2_(prp2),
        prps_(prps),
        d_ctrls_(d_ctrls),
        n_ctrls_(n_ctrls),
        n_blocks_per_page_(n_blocks_per_page),
        q_head(queue_head),
        q_tail(queue_tail),
        q_lock(queue_lock),
        extra_reads(queue_extra_reads),
        use_WB(WB_flag),
        use_PVP(PVP_flag),
        next_reuse_(next_reuse),
        //
        buffer_depth(b_depth),
        num_buffers(n_buffers),
        queue_counters(q_counters),
        index_arrays_(index_arrays),
        data_arrays_(data_arrays),
        my_GPU_id_(my_GPU_id),
        num_gpus(n_gpus)
        //victim_buffer(VB) 

        {
        double_read = 0;
        hit_cnt = 0;
        miss_cnt = 0;
      //  victim_buffer = VB;

        if(WB_flag){
            printf("eviction policy is EP_WINDOW\n");
            evict_policy = EP_WINDOW;
        }
    }

    __forceinline__
    __device__
    void
    print_stats(){
        printf("hit count: %llu\n", hit_cnt);        
        printf("miss count: %llu\n", miss_cnt);        
        printf("double reads: %llu\n", double_read);   
        float hit_ratio = (float) hit_cnt / (hit_cnt + miss_cnt);
        printf("Hit ratio: %f\n", hit_ratio);
        hit_cnt = 0;
        miss_cnt = 0;
        double_read = 0;
    }


    __device__
        void
        evict_to_pvp(uint64_t evicted_batch_index, uint64_t* cl_src_ptr, uint64_t cl_size, uint64_t reuse_val, uint32_t head_ptr, unsigned tid){
            uint32_t q_id = (reuse_val + head_ptr) % num_buffers;
            uint64_t buffer_idx  = 0;
            if(tid == 0){
                buffer_idx = atomicAdd((unsigned long long*)queue_counters + q_id, (unsigned long long)1);
                if(buffer_idx < buffer_depth){
                 //   printf("buffer_depth2 :%llu\n", (unsigned long long)buffer_depth);
                    index_arrays_[buffer_idx + buffer_depth * q_id] = evicted_batch_index;
                }
            }
            buffer_idx = __shfl_sync(FULL_MASK, buffer_idx, 0);

            if(buffer_idx < buffer_depth){
                uint64_t ele_per_CL = cl_size / sizeof(uint64_t);
                uint64_t ele_per_buffer = ele_per_CL * buffer_depth;

                uint64_t queue_offset =  q_id * ele_per_buffer + buffer_idx * ele_per_CL;

                warp_memcpy<uint64_t>(cl_src_ptr, (void*)(data_arrays_ + queue_offset), cl_size, FULL_MASK );
            }
        }
    

        __host__
        void prefetch_from_victim_queue_(uint64_t* PVP_pinned_data, uint64_t* PVP_pinned_idx, uint64_t cl_size, uint64_t num_evicted_cl, uint32_t head_ptr, cudaStream_t stream){
        

            uint64_t* wb_queue_head = data_arrays_ + cl_size / sizeof(uint64_t) * buffer_depth * head_ptr;
            uint64_t* wb_id_array_head = index_arrays_ + buffer_depth * head_ptr;

            cudaMemcpy(PVP_pinned_data, wb_queue_head, num_evicted_cl * cl_size, cudaMemcpyHostToDevice);	
            cudaMemcpy(PVP_pinned_idx, wb_id_array_head, num_evicted_cl * sizeof(uint64_t), cudaMemcpyHostToDevice);	
        // cudaMemcpyAsync(PVP_pinned_data, wb_queue_head, num_evicted_cl* cl_size, cudaMemcpyHostToDevice, stream);	
        // cudaMemcpyAsync(PVP_pinned_idx, wb_id_array_head, tnum_evicted_cl * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);	
        }


    __forceinline__
    __device__
    unsigned 
    search_ways(uint64_t tag_id, uint32_t thread_mask, uint64_t set_id){

        bool found = false;
        unsigned way = num_ways;
        uint64_t way_offset = set_id * num_ways;
        uint64_t* cur_ways = keys_ + way_offset;

        unsigned effective_warp_size = __popc(thread_mask);
        uint32_t lane = lane_id();
        uint32_t effective_lane_id = __popc(thread_mask & ((1 << lane) - 1));
        uint32_t effective_warp_leader = __ffs(thread_mask) - 1;
        
        __syncwarp(thread_mask);

        for(uint32_t i = effective_lane_id; i < num_ways; i += effective_warp_size){
            found = tag_id == cur_ways[i];
            if(found){
                way = i;
                break;
            }
        }
        unsigned found_mask = __ballot_sync(thread_mask, found);
        unsigned leader = found_mask ? (__ffs(found_mask) - 1) : (__ffs(thread_mask) - 1);

        way = __shfl_sync(thread_mask, way, leader);
        return way;
    }

    __forceinline__
    __device__
    uint64_t get_cl_id(uint64_t key){
        uint64_t cl_id = key / CL_SIZE;
        return cl_id;
    }
    
    __forceinline__
    __device__
    uint64_t get_set_id(uint64_t key){
        uint64_t set_id = key % num_sets;
        return set_id;
    }

    __forceinline__
    __device__
    uint32_t 
    round_robin_evict(uint32_t lane, uint32_t leader, uint32_t mask, uint64_t set_id){
        uint32_t way = 0;
        if(lane == leader){
            way = (set_cnt_[set_id]++) % num_ways;
            
        }
        else{
            way =  0;
        }
        way = __shfl_sync(mask, way, leader);
        return way;
    }

    __forceinline__
    __device__
    uint32_t 
    reuse_based_evict(uint32_t lane, uint32_t leader, uint32_t mask,  uint64_t set_id){

        unsigned next_reuse = 0xFFFF;
        // No Memory Consistency is guaranteed here for now, but it should not affect accuracy
        // NEED TO FIX
        if(lane < num_ways){
            uint64_t next_reuse_full = next_reuse_[set_id * num_ways + lane];
            next_reuse = (unsigned) (next_reuse_full >> 48);
            if(next_reuse < 128){
               // printf("next reuse:%u\n", next_reuse);
            }
        }

        unsigned next_reuse_min = __reduce_max_sync(mask, next_reuse);
        // No reuse
        if(next_reuse_min == 0xFFFF){
          //  if(lane == leader) printf("No reuse\n");
            return round_robin_evict(lane, leader, mask, set_id);
        }

        unsigned way = 0xFFFF;
        if(next_reuse == next_reuse_min){
            way = lane;
        }
        way = __reduce_min_sync(mask, way);
        //if(lane == leader) printf("evicted way: %u next_reuse_min:%u\n", way, next_reuse_min);
        return way;

      //   return (set_cnt_[set_id]++) % num_ways;
    }

    __forceinline__
    __device__
    uint32_t evict(uint32_t lane, uint32_t leader, uint32_t mask, uint64_t set_id, uint8_t eviction_policy){
        uint32_t evict_way;
        switch(eviction_policy){
            case RR :
                evict_way = round_robin_evict(lane, leader, mask, set_id);
                return evict_way;
            case EP_WINDOW:
                evict_way = reuse_based_evict(lane, leader, mask, set_id);
                return evict_way;
            default:
                return 0;
        }

    }

    inline 
    __device__ 
    void 
    enqueue_second( QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head) {
        nvm_cmd_rw_blks(cmd, starting_lba, 1);
        unsigned int ns = 8;
        do {
            uint64_t cur_pc_head = q_head->load(simt::memory_order_relaxed);
            bool sec = ((cur_pc_head < pc_prev_head) && (pc_prev_head <= pc_pos)) ||
                ((pc_prev_head <= pc_pos) && (pc_pos < cur_pc_head)) ||
                ((pc_pos < cur_pc_head) && (cur_pc_head < pc_prev_head));

            if (sec) break;

            //if not
            uint64_t qlv = q_lock->load(simt::memory_order_relaxed);
            //got lock
            if (qlv == 0) {
                qlv = q_lock->fetch_or(1, simt::memory_order_acquire);
                if (qlv == 0) {
                    uint64_t cur_pc_tail;// = pc->q_tail.load(simt::memory_order_acquire);

                    uint16_t sq_pos = sq_enqueue(&qp->sq, cmd, q_tail, &cur_pc_tail);
                    uint32_t head, head_;
                    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

                    q_head->store(cur_pc_tail, simt::memory_order_release);
                    q_lock->store(0, simt::memory_order_release);
                    extra_reads->fetch_add(1, simt::memory_order_relaxed);
                    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);



                    break;
                }
            }
            #if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                    __nanosleep(ns);
                    if (ns < 256) {
                        ns *= 2;
                    }
            #endif    
            } while(true);

    }


    inline __device__ 
    void read_data(QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {

        nvm_cmd_t cmd;
        uint16_t cid = get_cid(&(qp->sq));

        nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
        uint64_t prp1 = prp1_[pc_entry];
        uint64_t prp2 = 0;
        if (prps_)
            prp2 = prp2_[pc_entry];
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
        uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
        uint32_t head, head_;
        uint64_t pc_pos;
        uint64_t pc_prev_head;

        uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

        qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
        pc_prev_head = q_head->load(simt::memory_order_relaxed);
        pc_pos = q_tail->fetch_add(1, simt::memory_order_acq_rel);

        cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
        enqueue_second(qp, starting_lba, &cmd, cid, pc_pos, pc_prev_head);

        put_cid(&qp->sq, cid);

    }

    __forceinline__
    __device__
    void read_page(uint64_t pg_id, uint64_t evicted_page, uint32_t queue){

        //Need to be fixed (Assuming Page Start is 0)
        uint64_t ctrl = pg_id % n_ctrls_;
        uint64_t b_page = pg_id / n_ctrls_;



        Controller* c = d_ctrls_[ctrl];
        c->access_counter.fetch_add(1, simt::memory_order_relaxed);
		read_data((c->d_qps)+queue, ((b_page)*n_blocks_per_page_), n_blocks_per_page_, evicted_page);
        return;

    }

    __forceinline__
    __device__
    void 
    get_data(uint64_t id, T* output_ptr, uint32_t head_ptr, int64_t num_idx = 0){

        uint32_t lane = lane_id();
        uint32_t mask = __activemask();

        uint64_t cl_id = id / num_gpus;
        uint64_t set_id = get_set_id(cl_id);
        uint64_t set_offset = set_id * num_ways;

        seqlock* cur_set_lock = set_locks_ + set_id;
        seqlock* cur_cl_seqlock = way_locks_ + set_id * num_ways;

        uint32_t warp_leader = __ffs(mask) - 1;
        bool cont = true;
        bool done = false;

        do {
            uint64_t set_before, set_after;
            if(lane == warp_leader) {
                set_before = cur_set_lock->read_lock();
            }
            set_before = __shfl_sync(mask, set_before, warp_leader);
            unsigned way = search_ways(cl_id, mask, set_id);

            if(way < num_ways){
                if(way < num_ways){
                    seqlock* way_lock = way_locks_ + set_offset + way;
                    uint64_t way_before;
                    if(lane == warp_leader) {
                        way_before = way_lock->read_busy_block();
                    }
                    way_before = __shfl_sync(mask, way_before, warp_leader);

                    bool hit = false;
                    auto way_offset = set_offset + way;
                    auto way_key = keys_ + way_offset;

                    if(cl_id == (*way_key)){
                        hit = true;
                        //memcpy
                        void* src = ((void*)base_addr_)+ (set_offset + way) * CL_SIZE;
                        warp_memcpy<T>(src, output_ptr, CL_SIZE, mask);
                    }
                    __syncwarp(mask);

                    uint64_t way_after;
                    if(lane == warp_leader) {
                        way_after = way_lock->read_busy_unlock();
                    }
                    way_after = __shfl_sync(mask, way_after, warp_leader);

                    if(lane == warp_leader){
                        if(hit && (way_after != way_before)){
                            double_read.fetch_add(1, simt::memory_order_relaxed);
                        }

                    }


                    if(!done)
                        done = hit && (way_before == way_after);

                    unsigned not_done_mask = __ballot_sync(mask, !done);
                    if(not_done_mask == 0){
                        if(lane == warp_leader) hit_cnt.fetch_add(1, simt::memory_order_relaxed);
                        return;
                    }

                }
            }
            if(lane == warp_leader) set_after = cur_set_lock->read_unlock();
            set_after = __shfl_sync(mask, set_after, warp_leader);
            cont = set_before != set_after;
        } while(cont);

        //miss
        
        if(lane == warp_leader) {
            cur_set_lock->write_lock();
        }
        __syncwarp(mask);

        //EVICTION
        uint32_t way;
        way = evict(lane, warp_leader, mask, set_id, evict_policy);
       // way = __shfl_sync(mask, way, warp_leader);
        if(lane == warp_leader) {
            (cur_cl_seqlock + way) -> write_busy_lock();
        }
        //Check
        uint64_t old_key =  keys_[set_offset + way];
        keys_[set_offset + way] = cl_id;

        if(lane == warp_leader) {
            cur_set_lock->write_unlock();
        }
        __syncwarp(mask);

        //HANDLE MISS
        //printf("EVICT\n");

        //Premptive Victim-buffer Prefetcher is enabled
        if (use_PVP){
            void* cl_src = ((void*)base_addr_)+ (set_offset + way) * CL_SIZE;
            uint64_t reuse_line = next_reuse_[set_offset + way];

            //GPUID + Batch Index
            uint64_t evicted_batch_id = (reuse_line & (0x0000FFFFFFFFFFFF));
            uint64_t evicted_batch_id_test = (reuse_line & (0x000000FFFFFFFFFF));

            uint64_t reuse_val = reuse_line >> 48;
            uint64_t GPU_id = (reuse_line >> 40) & (0x00FF);

            if(reuse_val != FF_16){
                //if(evicted_batch_id_test >=8000 )printf("wrong evict: %llu GPU: %llu\n", evicted_batch_id_test, GPU_id);
                evict_to_pvp(evicted_batch_id, (uint64_t*) cl_src, CL_SIZE, reuse_val, head_ptr, lane);
                //victim_buffer.evict_to_pvp(evicted_batch_id, (uint64_t*) cl_src, CL_SIZE, reuse_val, head_ptr, lane);

                //if(threadIdx.x % 32 == 0) printf("GPU: %llu evict to pvp KEY:%llu batch_id:%llu my GPU id:%u reuse_val:%llu GPU_id:%llu num_idx:%llu \n",(unsigned long long) GPU_id, (unsigned long long)old_key, (unsigned long long)evicted_batch_id_test, (unsigned) my_GPU_id_, (unsigned long long)reuse_val,(unsigned long long) GPU_id, (unsigned long long)num_idx);
            }
        }

        uint32_t queue;
        if(lane == warp_leader) {
            queue = get_smid() % (d_ctrls_[0]->n_qps);
            read_page(cl_id, set_offset + way, queue);
        }

        __syncwarp(mask);

        if(lane == warp_leader) {
            (cur_cl_seqlock + way)->write_unbusy();
        }
        __syncwarp(mask);

        void* src = ((void*)base_addr_)+ (set_offset + way) * CL_SIZE;
        warp_memcpy<T>(src, output_ptr, CL_SIZE, mask);
        if(lane == warp_leader) {
            miss_cnt.fetch_add(1, simt::memory_order_relaxed);
            if(use_WB){
                next_reuse_[set_offset + way] = FF_64;
            }
        }
        __syncwarp(mask);

        if(lane == warp_leader) {
            (cur_cl_seqlock + way)->write_busy_unlock();
        }
        __syncwarp(mask);
        

    }

    // Only supported when Metadata for PVP is enabled
    __device__
    void
    update_reuse_val(uint64_t node_id, uint32_t reuse_time, uint32_t GPU_id, uint64_t batch_idx, uint64_t num_idx=0){
        uint64_t key = node_id / num_gpus;
        uint64_t set_id = get_set_id(key);
        uint64_t set_offset = set_id * num_ways;
        uint64_t* cur_keys = keys_ + set_offset;
        uint64_t* cur_next_reuse = next_reuse_ + set_offset;

        uint32_t lane = lane_id();
        uint32_t mask = __activemask();
        uint32_t active_threads = __popc(mask);

        for(uint32_t i = lane; i < num_ways; i += active_threads){
            if(key == cur_keys[i]){
                uint64_t update_val = reuse_time;
                //printf("WRITE GPU_id:%u key: %llu update val: %llu num_idx:%llu\n",(unsigned) GPU_id, (unsigned long long) key, (unsigned long long) update_val, (unsigned long long) num_idx);

                update_val = update_val << 48;
                uint64_t gpu_b = (uint64_t)GPU_id << 40;

                update_val = (update_val | batch_idx | gpu_b);

               // atomicMin((unsigned long long int*) (cur_next_reuse + i), (unsigned long long int) update_val);
                atomicMin((unsigned long long int*) (cur_next_reuse + i), (unsigned long long int) 0);


            }
        }
        return;
    }

    __device__
    void
    print_vals(){
        printf("Queue depth:%llu\n", (unsigned long long) (buffer_depth));
    }


  
    
};



template<typename T>
struct GIDS_SA_handle{
    SA_cache_d_t<T>* cache_ptr = nullptr;

    seqlock* d_set_locks;
    seqlock* d_way_locks;

    uint64_t num_sets_;
    uint64_t num_ways_;
    uint64_t CL_SIZE_;


    uint64_t* keys_;
    uint32_t* set_cnt_;

    //WB Meta data
    uint64_t* next_reuse_ = nullptr;


    Controller* ctrls_;

    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    bool prps;

    uint8_t* base_addr;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf; 


    BufferPtr page_ticket_buf;
    BufferPtr ctrl_counter_buf;
    BufferPtr q_head_buf;
    BufferPtr q_tail_buf;
    BufferPtr q_lock_buf;
    BufferPtr extra_reads_buf;

    BufferPtr double_reads_buf;

    BufferPtr d_ctrls_buff;


    bool use_WB;
    bool use_PVP;

    //Victim_Buffer victim_buffer;
        //public:
    // uint32_t num_buffers;
    // uint32_t buffer_depth;
  //  uint64_t* queue_counters;


    uint64_t*  queue_counters;

    uint64_t* host_index_array;
    uint64_t* host_data_array;

    uint64_t* device_index_array;
    uint64_t* device_data_array;

    const uint32_t num_buffers;
    const uint32_t buffer_depth;

    unsigned int my_GPU_id_;
    unsigned num_gpus = 1;

    uint64_t total_evicted_cl = 0;

    __host__
    void flush_next_reuse(){
        cudaMemset(next_reuse_, 0xFF, (uint64_t) sizeof(uint64_t) * num_sets_ * num_ways_);
    }

    __host__ 
    GIDS_SA_handle(uint64_t num_sets, uint64_t num_ways, uint64_t cl_size, const Controller& ctrl, const std::vector<Controller*>& ctrls,  const uint32_t cudaDevice,
                    bool use_WB_flag, bool use_PVP_flag, unsigned int my_GPU_id, unsigned n_gpus, uint32_t n_buffer = 1, uint32_t b_depth = 1) :
    num_sets_(num_sets),
    num_ways_(num_ways),
    CL_SIZE_(cl_size),
    use_WB(use_WB_flag),
    use_PVP(use_PVP_flag),
    num_buffers(n_buffer),
    buffer_depth(b_depth),
    my_GPU_id_(my_GPU_id),
    num_gpus(n_gpus)
    {
        uint64_t* prp1;
        uint64_t* prp2;

        cudaMalloc((void**)&d_set_locks, sizeof(seqlock) * num_sets_);
        cudaMalloc((void**)&d_way_locks, sizeof(seqlock) * num_sets_ * num_ways_);

        cudaMalloc((void**)&set_cnt_, sizeof(uint32_t) * num_sets_);
        cudaMalloc((void**)&keys_, sizeof(uint64_t) * num_sets_ * num_ways_);

        cudaMemset(d_set_locks, 0, sizeof(seqlock) * num_sets_);
        cudaMemset(d_way_locks, 0, sizeof(seqlock) * num_sets_ * num_ways_);
        cudaMemset(set_cnt_, 0, sizeof(seqlock) * num_sets_);
        cudaMemset(keys_, 0xFF, sizeof(uint64_t) * num_sets_ * num_ways_);

        cudaMalloc((void**)&cache_ptr, sizeof(SA_cache_d_t<T>));


        if(use_WB){
            cudaMalloc((void**)&next_reuse_, (uint64_t) sizeof(uint64_t) * num_sets_ * num_ways_);
            cudaMemset(next_reuse_, 0xFF, (uint64_t) sizeof(uint64_t) * num_sets_ * num_ways_);

        }

        if(use_PVP){      
            printf("initializing PVP buffer num_buffer:%lu buffer_depth:%lu\n", num_buffers, buffer_depth);
            cudaMalloc(&queue_counters, sizeof(uint64_t) * num_buffers);
            cudaMemset(queue_counters, 0, sizeof(uint64_t) * num_buffers);


            cudaHostAlloc((uint64_t **)&host_index_array, sizeof(uint64_t) * num_buffers * buffer_depth , cudaHostAllocMapped);
            cudaHostGetDevicePointer((uint64_t **)&device_index_array, (uint64_t *)host_index_array, 0);

            cudaHostAlloc((uint64_t **)&host_data_array, CL_SIZE_ * num_buffers * buffer_depth, cudaHostAllocMapped);
            cudaHostGetDevicePointer((uint64_t **)&device_data_array, (uint64_t *)host_data_array, 0);

           // victim_buffer.init(num_buffers, buffer_depth, queue_counters, device_index_array, device_data_array);
        }

        uint64_t cache_size = CL_SIZE_ * num_sets_ * num_ways_;
        pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), cudaDevice);
        base_addr = (uint8_t*) pages_dma.get()->vaddr;
        const uint32_t uints_per_page = ctrl.ctrl->page_size / sizeof(uint64_t);


        ctrl_counter_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_head_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_tail_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_lock_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        extra_reads_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        


        auto cache_ctrl_counter = (simt::atomic<uint64_t, simt::thread_scope_device>*)ctrl_counter_buf.get();

        auto cache_q_head = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_head_buf.get();
        auto cache_q_tail = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_tail_buf.get();
        auto cache_q_lock = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_lock_buf.get();
        auto cache_extra_reads = (simt::atomic<uint64_t, simt::thread_scope_device>*)extra_reads_buf.get();


        uint32_t n_ctrls = ctrls.size();
        d_ctrls_buff = createBuffer(n_ctrls * sizeof(Controller*), cudaDevice);
        auto d_ctrls = (Controller**) d_ctrls_buff.get();
        auto n_blocks_per_page = (CL_SIZE_/ctrl.blk_size);
        
        for (size_t k = 0; k < n_ctrls; k++)
            cuda_err_chk(cudaMemcpy(d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));
        

        if (CL_SIZE_ <= pages_dma.get()->page_size) {
            std::cout << "Cond1\n";
            uint64_t how_many_in_one = ctrl.ctrl->page_size/CL_SIZE_;

            this->prp1_buf = createBuffer(num_sets_ * num_ways_ * sizeof(uint64_t), cudaDevice);
            prp1 = (uint64_t*) this->prp1_buf.get();

            std::cout << (num_sets_ * num_ways_) << " " << sizeof(uint64_t) << " " << how_many_in_one << " " << this->pages_dma.get()->n_ioaddrs <<std::endl;
            uint64_t* temp = new uint64_t[how_many_in_one * pages_dma.get()->n_ioaddrs];
            std::memset(temp, 0, how_many_in_one *  pages_dma.get()->n_ioaddrs);

            if (temp == NULL)
                std::cout << "NULL\n";

            for (size_t i = 0; (i < this->pages_dma.get()->n_ioaddrs) ; i++) {
                for (size_t j = 0; (j < how_many_in_one); j++) {
                    temp[i*how_many_in_one + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*CL_SIZE_;
                }
            }
            cuda_err_chk(cudaMemcpy(prp1, temp, num_sets_ * num_ways_ * sizeof(uint64_t), cudaMemcpyHostToDevice));

            delete temp;
            prps = false;
        }

        SA_cache_d_t<T> cache_host(d_set_locks, d_way_locks, num_sets_, num_ways_, CL_SIZE_, keys_, set_cnt_, d_ctrls, n_ctrls, n_blocks_per_page, base_addr, prp1, prp2, prps,
        cache_q_head, cache_q_tail, cache_q_lock, cache_extra_reads, 
        next_reuse_, use_WB, use_PVP, 
        //victim_buffer,           
        num_buffers,  buffer_depth,  queue_counters, device_index_array,  device_data_array,
        my_GPU_id_, num_gpus);

        //std::cout << "Cache host: " << cache_host.victim_buffer.buffer_depth;

        cuda_err_chk(cudaMemcpy(cache_ptr, &cache_host, sizeof(SA_cache_d_t<T>), cudaMemcpyHostToDevice));
    }

    __host__ 
    SA_cache_d_t<T>* 
    get_ptr(){
        return cache_ptr;
    }


    __host__
    void print_victim_buffer_index(uint64_t offset, uint64_t len){
        std::cout << "Printing VB index\n";
        for(auto i = 0; i < len; i++){
            std::cout << host_index_array[offset+i] << std::endl;
        }
    }

     __host__
    void print_victim_buffer_data(uint64_t offset, uint64_t len){
        std::cout << "Printing VB data\n";
        std::cout << host_index_array[offset] << "\t";

        float* h_data_array = (float*) host_data_array;
        for(auto i = 0; i < len; i++){
            std::cout << h_data_array[offset*CL_SIZE_/sizeof(T) + i] << " ";
        }
        std::cout << std::endl;
    }

     __host__
        void prefetch_from_victim_queue_(uint64_t* PVP_pinned_data, uint64_t* PVP_pinned_idx, uint64_t cl_size, uint64_t num_evicted_cl, uint32_t head_ptr, cudaStream_t stream){
        

            uint64_t* wb_queue_head = host_data_array + cl_size / sizeof(uint64_t) * buffer_depth * head_ptr;
            uint64_t* wb_id_array_head = host_index_array + buffer_depth * head_ptr;

            //cudaMemcpy(PVP_pinned_data, wb_queue_head, num_evicted_cl * cl_size, cudaMemcpyHostToDevice);	
            //cudaMemcpy(PVP_pinned_idx, wb_id_array_head, num_evicted_cl * sizeof(uint64_t), cudaMemcpyHostToDevice);	
            cudaMemcpyAsync(PVP_pinned_data, wb_queue_head, num_evicted_cl* cl_size, cudaMemcpyHostToDevice, stream);	
            cudaMemcpyAsync(PVP_pinned_idx, wb_id_array_head, num_evicted_cl * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);	
        }

    __host__
    uint64_t prefetch_from_victim_queue(T* PVP_pinned_data, uint64_t* PVP_pinned_idx, uint32_t head_ptr, cudaStream_t stream){
    
     //prefetch_from_victim_queue(T* PVP_pinned_data, uint64_t* PVP_pinned_idx, 
     //uint64_t cl_size, uint64_t num_evicted_cl, uint32_t head_ptr, cudaStream_t stream){
        uint64_t num_evicted_cl = 0;
        cudaMemcpy(&num_evicted_cl, queue_counters + head_ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if(num_evicted_cl > buffer_depth) 
            num_evicted_cl = buffer_depth;

        total_evicted_cl += num_evicted_cl;
        cudaMemset(queue_counters+head_ptr, 0, sizeof(uint64_t));
        prefetch_from_victim_queue_((uint64_t*)PVP_pinned_data, PVP_pinned_idx, CL_SIZE_, num_evicted_cl, head_ptr, stream);

   

        return num_evicted_cl;
    }

    __host__
    void print_evicted_cl(){
        printf("Evicted cache line: %llu\n", total_evicted_cl);
        total_evicted_cl =0 ;
    }

};






#endif // __SET_ASSOCIATIVE_PAGE_CACHE_H__
