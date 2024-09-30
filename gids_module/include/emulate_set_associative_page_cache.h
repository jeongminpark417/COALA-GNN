#ifndef __EMULATE_SET_ASSOCIATIVE_PAGE_CACHE_H__
#define __EMULATE_SET_ASSOCIATIVE_PAGE_CACHE_H__

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

#include "set_associative_page_cache.h"




template<typename T>
struct Emulate_SA_cache_d_t {

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

    //uint8_t evict_policy = EP_WINDOW;
    uint8_t evict_policy;
    uint8_t* static_info_;


    bool use_WB;
    bool use_PVP;

    //nvme controllers
    uint8_t* base_addr_; 
    bool prps_;


    simt::atomic<uint64_t, simt::thread_scope_device>* q_head;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_tail;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_lock;
    simt::atomic<uint64_t, simt::thread_scope_device>* extra_reads;


    simt::atomic<uint64_t, simt::thread_scope_device> double_read;
    simt::atomic<uint64_t, simt::thread_scope_device> hit_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> miss_cnt;



    Emulate_SA_cache_d_t(seqlock* set_locks, seqlock* way_locks,uint64_t n_sets, uint64_t n_ways, uint64_t cl_size, uint64_t* keys, uint32_t* set_cnt,
                 uint8_t* base_addr, 
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_head, simt::atomic<uint64_t, simt::thread_scope_device>* queue_tail, 
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_lock, simt::atomic<uint64_t, simt::thread_scope_device>* queue_extra_reads,
                uint64_t* next_reuse, bool WB_flag, bool PVP_flag,
                // Victim_Buffer VB,
                uint32_t n_buffers, uint32_t b_depth, uint64_t* q_counters,   uint64_t* index_arrays, uint64_t* data_arrays,
                unsigned int my_GPU_id, unsigned n_gpus, uint8_t ep, uint8_t* static_info_p
                 ) :
        set_locks_(set_locks),
        way_locks_(way_locks),
        num_sets(n_sets),
        num_ways(n_ways),
        CL_SIZE(cl_size),
        keys_(keys),
        set_cnt_(set_cnt),
        base_addr_(base_addr),
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
        num_gpus(n_gpus),
        evict_policy(ep),
        static_info_(static_info_p)
        //victim_buffer(VB) 

        {
        double_read = 0;
        hit_cnt = 0;
        miss_cnt = 0;


        }

    __forceinline__
    __device__
    void
    print_stats(){
        printf("hit count: %llu\n", (unsigned long long) hit_cnt);        
        printf("miss count: %llu\n", (unsigned long long) miss_cnt);        
        printf("double reads: %llu\n", double_read);   
        float hit_ratio = (float) hit_cnt / (hit_cnt + miss_cnt);
        printf("Hit ratio: %f\n", hit_ratio);
        hit_cnt = 0;
        miss_cnt = 0;
        double_read = 0;
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
    dynamic_evict(uint32_t lane, uint32_t leader, uint32_t mask,  uint64_t set_id){

        unsigned next_reuse = 0xFFFF;
        if(lane < num_ways){
            uint64_t next_reuse_full = next_reuse_[set_id * num_ways + lane];
            next_reuse = (unsigned) (next_reuse_full >> 48);
        }

        unsigned next_reuse_min = __reduce_max_sync(mask, next_reuse);

        unsigned way = 0xFFFF;
        if(next_reuse == next_reuse_min){
            way = lane;
        }
        way = __reduce_min_sync(mask, way);
        return way;

     }


    __forceinline__
    __device__
    uint32_t
    static_evict(uint32_t lane, uint32_t leader, uint32_t mask,  uint64_t set_id){
        
       unsigned static_val = static_info_[set_id * num_ways + lane];
        
        unsigned min_static_val = __reduce_min_sync(mask, static_val);
        unsigned way = 0xFFFF;
        if(static_val == min_static_val){
            way = lane;
        }
        way = __reduce_min_sync(mask, way);
        return way;

    }


// 
// 1. From nodes with no reuse (dynamic information), pick the lowest page rank value (static information)
// 2. From nodes with reuse value that is lower than the threshold, pick the lowest page rank value
// 3. From the recently inserted nodes (No dynamic information), pick the lowest page rank value
// 4. From nodes with reuse value that is higher than the threshold, pick the lowest page rank value

    __forceinline__
    __device__
    uint32_t
    hybrid_evict(uint32_t lane, uint32_t leader, uint32_t mask,  uint64_t set_id){

        //FIX
        uint32_t th = num_buffers / 8;

        uint64_t next_reuse_full = next_reuse_[set_id * num_ways + lane];
        unsigned next_reuse = (unsigned) (next_reuse_full >> 48);
        uint8_t static_val = static_info_[set_id * num_ways + lane];
        
        unsigned way = 0xFFFF;

        unsigned  temp = 0x7000;
        if(next_reuse == 0xFFFF) temp = static_val;
        unsigned  reduce_temp = __reduce_min_sync(mask, temp);
        //Case 1
        if(reduce_temp != 0x7000){
            if(reduce_temp == temp){
                way = lane;
            }
            way = __reduce_min_sync(mask, way);
            return way;
        }
        
        temp = 0x7000;
        if(next_reuse > th && next_reuse != 0xFFFE) temp = static_val;
        reduce_temp = __reduce_min_sync(mask, temp);
        //Case 2
        if(reduce_temp != 0x7000){
            if(reduce_temp == temp){
                way = lane;
            }
            way = __reduce_min_sync(mask, way);
            return way;
        }   

        //Case 3
        if(next_reuse == 0xFFFE) temp = static_val;
        reduce_temp = __reduce_min_sync(mask, temp);
        if(reduce_temp != 0x7000){
            if(reduce_temp == temp){
                way = lane;
            }
            way = __reduce_min_sync(mask, way);
            return way;
        }


        temp = static_val;
        reduce_temp = __reduce_min_sync(mask, temp);
        //Case 4
        if(reduce_temp == temp){
            way = lane;
        }
        way = __reduce_min_sync(mask, way);
        return way;
        
    }



    __forceinline__
    __device__
    uint32_t evict(uint32_t lane, uint32_t leader, uint32_t mask, uint64_t set_id, uint8_t eviction_policy){
        uint32_t evict_way;
        switch(eviction_policy){
            //RR
            case 0 :
                evict_way = round_robin_evict(lane, leader, mask, set_id);
                return evict_way;
            //Static
            case 1:
                evict_way = static_evict(lane, leader, mask, set_id);
                return evict_way;
            //Dynamic
            case 2:
                evict_way = dynamic_evict(lane, leader, mask, set_id);
                return evict_way;
            //Hybrid
            case 3:
            
                evict_way = hybrid_evict(lane, leader, mask, set_id);
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


    __forceinline__
    __device__
    void read_page(uint64_t pg_id, uint64_t evicted_page){

        //Need to be fixed (Assuming Page Start is 0)
        // uint64_t ctrl = pg_id % n_ctrls_;
        // uint64_t b_page = pg_id / n_ctrls_;

        //write page from CPU

        return;
        // Controller* c = d_ctrls_[ctrl];
        // c->access_counter.fetch_add(1, simt::memory_order_relaxed);
		// read_data((c->d_qps)+queue, ((b_page)*n_blocks_per_page_), n_blocks_per_page_, evicted_page);
        // return;

    }


    __forceinline__
    __device__
    void 
    get_data(uint64_t id, T* output_ptr, uint64_t b_id, uint8_t* static_info_ptr){

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


        uint32_t queue;
        if(lane == warp_leader) {
            //queue = get_smid() % (d_ctrls_[0]->n_qps);
            read_page(cl_id, set_offset + way);
        }

        __syncwarp(mask);

        if(lane == warp_leader) {
            (cur_cl_seqlock + way)->write_unbusy();
        }
        __syncwarp(mask);

        
        void* src = ((void*)base_addr_)+ (set_offset + way) * CL_SIZE;
        //warp_memcpy<T>(src, output_ptr, CL_SIZE, mask);
        if(lane == warp_leader) {
            miss_cnt.fetch_add(1, simt::memory_order_relaxed);
           // printf("miss cnt inc\n");
            // if(use_WB){
            //     next_reuse_[set_offset + way] = FE_64;
            // }
            // if(evict_policy == 1 || evict_policy == 3){
            //     // STATIC INFO
            //     static_info_[set_offset+way] = static_info_ptr[b_id];
            // }
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

                atomicMin((unsigned long long int*) (cur_next_reuse + i), (unsigned long long int) update_val);
               // atomicMin((unsigned long long int*) (cur_next_reuse + i), (unsigned long long int) 0);


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
struct Emulate_SA_handle{
    Emulate_SA_cache_d_t<T>* cache_ptr = nullptr;

    seqlock* d_set_locks;
    seqlock* d_way_locks;

    uint64_t num_sets_;
    uint64_t num_ways_;
    uint64_t CL_SIZE_;


    uint64_t* keys_;
    uint32_t* set_cnt_;

    //WB Meta data
    uint64_t* next_reuse_ = nullptr;

    uint8_t* base_addr;


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


    Eviction_Policy cache_ep;
    uint8_t* static_info_;



    __host__ 
    Emulate_SA_handle(uint64_t num_sets, uint64_t num_ways, uint64_t cl_size, const uint32_t cudaDevice, uint8_t eviction_p, uint32_t n_buffer = 1, uint32_t b_depth = 1)  :
    num_sets_(num_sets),
    num_ways_(num_ways),
    num_buffers(n_buffer),
    buffer_depth(b_depth),
    CL_SIZE_(cl_size)
    {
       

        cudaMalloc((void**)&d_set_locks, sizeof(seqlock) * num_sets_);
        cudaMalloc((void**)&d_way_locks, sizeof(seqlock) * num_sets_ * num_ways_);

        cudaMalloc((void**)&set_cnt_, sizeof(uint32_t) * num_sets_);
        cudaMalloc((void**)&keys_, sizeof(uint64_t) * num_sets_ * num_ways_);

        cudaMemset(d_set_locks, 0, sizeof(seqlock) * num_sets_);
        cudaMemset(d_way_locks, 0, sizeof(seqlock) * num_sets_ * num_ways_);
        cudaMemset(set_cnt_, 0, sizeof(seqlock) * num_sets_);
        cudaMemset(keys_, 0xFF, sizeof(uint64_t) * num_sets_ * num_ways_);

        cudaMalloc((void**)&cache_ptr, sizeof(SA_cache_d_t<T>));


        if(eviction_p == 1 || eviction_p == 3){
            cudaMalloc((void**)&static_info_, (uint64_t) sizeof(uint8_t) * num_sets_ * num_ways_);
            cudaMemset(static_info_, 0x0, (uint64_t) sizeof(uint8_t) * num_sets_ * num_ways_);
        }

        uint64_t cache_size = CL_SIZE_ * num_sets_ * num_ways_;
        // pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), cudaDevice);
        // base_addr = (uint8_t*) pages_dma.get()->vaddr;
        cudaMalloc(&base_addr, cache_size);


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



        Emulate_SA_cache_d_t<T> cache_host(d_set_locks, d_way_locks, num_sets_, num_ways_, CL_SIZE_, keys_, set_cnt_, base_addr, 
        cache_q_head, cache_q_tail, cache_q_lock, cache_extra_reads, 
        next_reuse_, use_WB, use_PVP, 
        //victim_buffer,           
        num_buffers,  buffer_depth,  queue_counters, device_index_array,  device_data_array,
        my_GPU_id_, num_gpus,
        eviction_p, static_info_);

        //std::cout << "Cache host: " << cache_host.victim_buffer.buffer_depth;

        cuda_err_chk(cudaMemcpy(cache_ptr, &cache_host, sizeof(SA_cache_d_t<T>), cudaMemcpyHostToDevice));
    }

    __host__ 
    Emulate_SA_cache_d_t<T>* 
    get_ptr(){
        return cache_ptr;
    }

    // __host__
    // void
    // print_counters(){

        
    //     cache_ptr -> print_stats();
    //     return ;
    // }

};






#endif // __SET_ASSOCIATIVE_PAGE_CACHE_H__
