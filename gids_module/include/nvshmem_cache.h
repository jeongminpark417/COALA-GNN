#ifndef __NVSHMEMCACHE_H__
#define __NVSHMEMCACHE_H__


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
#include "nvshmem_cache_buffer.h"
#include "ctrl.h"
#include <iostream>
#include "nvm_parallel_queue.h"
#include "nvm_cmd.h"

//#include <cuda_runtime.h>



enum Eviction_Policy {
    RR,
    Static,
    Dynamic,
    Hybrid
};

#define FULL_MASK 0xFFFFFFFF
#define FF_64 0xFFFFFFFFFFFFFFFF
#define FE_64 0xFFFEFFFFFFFFFFFF

#define FF_48 0xFFFFFFFFFFFFFF
#define FF_16 0xFFFF
#define FE_16 0xFFFE



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

template<typename T>
struct NVSHMEM_cache_d_t {

    unsigned num_gpus = 1;
    unsigned int my_GPU_id_;

    seqlock* set_locks_;
    seqlock* way_locks_;

    uint64_t num_sets;
    uint64_t num_ways;
    uint64_t cpu_num_ways;
    uint64_t gpu_num_ways;

    uint64_t CL_SIZE;


    uint64_t* keys_;
    uint32_t* set_cnt_;

    //nvme controllers
    Controller** d_ctrls_;
    uint32_t n_ctrls_;

    uint64_t n_blocks_per_page_;

    uint8_t* base_addr_;
    uint64_t* prp1_;                  
    uint64_t* prp2_;   
    bool prps_;

    uint8_t* cpu_base_addr_;
    uint64_t* cpu_prp1_;  

    simt::atomic<uint64_t, simt::thread_scope_device>* q_head;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_tail;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_lock;
    simt::atomic<uint64_t, simt::thread_scope_device>* extra_reads;


    simt::atomic<uint64_t, simt::thread_scope_device> gpu_double_read;
    simt::atomic<uint64_t, simt::thread_scope_device> gpu_hit_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> gpu_miss_cnt;

    simt::atomic<uint64_t, simt::thread_scope_device> cpu_double_read;
    simt::atomic<uint64_t, simt::thread_scope_device> cpu_hit_cnt;
    simt::atomic<uint64_t, simt::thread_scope_device> cpu_miss_cnt;



    uint8_t evict_policy = 0;

    bool is_simulation = false;
    void* sim_buf = nullptr;


    NVSHMEM_cache_d_t(seqlock* set_locks, seqlock* way_locks,uint64_t n_sets, uint64_t g_n_ways, uint64_t cl_size, uint64_t* keys, uint32_t* set_cnt,
                 Controller** d_ctrls, uint32_t n_ctrls, uint64_t n_blocks_per_page, uint8_t* base_addr, uint64_t* prp1, uint64_t* prp2, bool prps,
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_head, simt::atomic<uint64_t, simt::thread_scope_device>* queue_tail, 
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_lock, simt::atomic<uint64_t, simt::thread_scope_device>* queue_extra_reads,
        
                  
                    unsigned int my_GPU_id, unsigned n_gpus, uint64_t cpu_n_ways, uint8_t* cpu_base_addr, uint64_t* cpu_prp1, bool simulation, void* sim_buf_ptr
                 ) :
        set_locks_(set_locks),
        way_locks_(way_locks),
        num_sets(n_sets),
        gpu_num_ways(g_n_ways),
        cpu_num_ways(cpu_n_ways),
        CL_SIZE(cl_size),
        keys_(keys),
        set_cnt_(set_cnt),
        base_addr_(base_addr),
        cpu_base_addr_(cpu_base_addr),
        prp1_(prp1),
        prp2_(prp2),
        prps_(prps),
        cpu_prp1_(cpu_prp1),
        d_ctrls_(d_ctrls),
        n_ctrls_(n_ctrls),
        n_blocks_per_page_(n_blocks_per_page),
        q_head(queue_head),
        q_tail(queue_tail),
        q_lock(queue_lock),
        extra_reads(queue_extra_reads),


        my_GPU_id_(my_GPU_id),
        num_gpus(n_gpus),
        is_simulation(simulation),
        sim_buf(sim_buf_ptr)

        {
            num_ways = g_n_ways + cpu_n_ways;
            //double_read = 0;
            gpu_hit_cnt = 0;
            gpu_miss_cnt = 0;
            cpu_hit_cnt = 0;
            cpu_miss_cnt = 0;

        }

    __forceinline__
    __device__
    void
    print_stats(){
        printf("GPU hit count: %llu CPU hit count: %llu\n", (unsigned long long) gpu_hit_cnt,  (unsigned long long) cpu_hit_cnt);            
        printf("GPU miss count: %llu CPU miss count: %llu\n", (unsigned long long) gpu_miss_cnt,  (unsigned long long) cpu_miss_cnt);            
//        printf("GPU double reads: %llu CPU double reads: %llu\n", (unsigned long long) gpu_double_read,  (unsigned long long) cpu_double_read);            

        uint64_t system_hit = gpu_hit_cnt + cpu_hit_cnt;
        uint64_t system_miss = gpu_miss_cnt + cpu_miss_cnt;
        
        float overall_hit_ratio = (float) (system_hit) / (system_hit + system_miss);
        float gpu_hit_ratio = (float) (gpu_hit_cnt) / (system_hit + system_miss);
        float cpu_hit_ratio = (float) (cpu_hit_cnt) / (system_hit + system_miss);

        printf("System Hit ratio: %f GPU hit ratio: %f CPU hit ratio:%f\n", overall_hit_ratio, gpu_hit_ratio, cpu_hit_ratio);

        gpu_hit_cnt = 0;
        gpu_miss_cnt = 0;
       // gpu_double_read = 0;
        cpu_hit_cnt = 0;
        cpu_miss_cnt = 0;
        //cpu_double_read = 0;
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
    uint64_t get_dist_set_id(uint64_t key, int num_gpus){
        uint64_t set_id = (key / num_gpus) % num_sets;
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

 
    // __forceinline__
    // __device__
    // uint32_t
    // static_evict(uint32_t lane, uint32_t leader, uint32_t mask,  uint64_t set_id){
        
    //    unsigned static_val = static_info_[set_id * num_ways + lane];
        
    //     unsigned min_static_val = __reduce_min_sync(mask, static_val);
    //     unsigned way = 0xFFFF;
    //     if(static_val == min_static_val){
    //         way = lane;
    //     }
    //     way = __reduce_min_sync(mask, way);
    //     return way;

    // }



    __forceinline__
    __device__
    uint32_t evict(uint32_t lane, uint32_t leader, uint32_t mask, uint64_t set_id, uint8_t eviction_policy=0){
        uint32_t evict_way;
        switch(eviction_policy){
            //RR
            case 0 :
                evict_way = round_robin_evict(lane, leader, mask, set_id);
                return evict_way;
            //Static
            // case 1:
            //     evict_way = static_evict(lane, leader, mask, set_id);
            //     return evict_way;

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
    void read_data(QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry, bool to_cpu ) {

        nvm_cmd_t cmd;
        uint16_t cid = get_cid(&(qp->sq));

        nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
        uint64_t prp1, prp2;
        if(to_cpu){
            prp1 = cpu_prp1_[pc_entry];
            prp2 = 0;
        }
        else{
            prp1 = prp1_[pc_entry];
            prp2 = 0;
            if (prps_)
                prp2 = prp2_[pc_entry];
        }
        
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
    void read_page(uint64_t pg_id, uint64_t evicted_page, uint32_t queue, bool to_cpu){


      
        uint64_t ctrl = pg_id % n_ctrls_;
        uint64_t b_page = pg_id / n_ctrls_;

        Controller* c = d_ctrls_[ctrl];
        c->access_counter.fetch_add(1, simt::memory_order_relaxed);
        read_data((c->d_qps)+queue, ((b_page)*n_blocks_per_page_), n_blocks_per_page_, evicted_page, to_cpu);
        return;
        
    }
     __forceinline__
    __device__
    void read_page_simulation(uint64_t pg_id, void* dst, uint32_t mask){

        void* src = sim_buf + pg_id * CL_SIZE;
         warp_memcpy<float>(src, dst, CL_SIZE, mask);

        return;
    }
                


    __forceinline__
    __device__
    void 
    get_data( uint64_t id, T* output_ptr, bool use_nvshmem = false, int rank = 0, int dst_gpu = 0){

        uint32_t lane = lane_id();
        uint32_t mask = __activemask();

        //uint64_t cl_id = id / num_gpus;
        uint64_t cl_id = id ;

        uint64_t set_id = get_dist_set_id(cl_id, num_gpus);
        uint64_t set_offset = set_id * (num_ways);
        uint64_t gpu_set_offset = set_id * (gpu_num_ways);
        uint64_t cpu_set_offset = set_id * (cpu_num_ways);

        seqlock* cur_set_lock = set_locks_ + set_id;
        seqlock* cur_cl_seqlock = way_locks_ + set_id * (num_ways );

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
                    bool is_gpu_cache = false;
                    if(way < gpu_num_ways)
                        is_gpu_cache = true;
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
                        void* src ;
                        //CPU memory
                        if(way >= gpu_num_ways){
                            src = ((void*)cpu_base_addr_)+ (cpu_set_offset + way - gpu_num_ways) * CL_SIZE;                   
                        }
                        else{
                            src = ((void*)base_addr_)+ (gpu_set_offset + way) * CL_SIZE;
                        }
                        if(use_nvshmem){
                            nvshmemx_float_put_warp(output_ptr, (float*) src, CL_SIZE/sizeof(float), dst_gpu);
                        }
                        else{
                            warp_memcpy<float>(src, output_ptr, CL_SIZE, mask);
                        }
                    

                    }
                    __syncwarp(mask);

                    uint64_t way_after;
                    if(lane == warp_leader) {
                        way_after = way_lock->read_busy_unlock();
                    }
                    way_after = __shfl_sync(mask, way_after, warp_leader);

                    // if(lane == warp_leader){
                    //     if(hit && (way_after != way_before)){
                    //         double_read.fetch_add(1, simt::memory_order_relaxed);
                    //     }
                    // }


                    if(!done)
                        done = hit && (way_before == way_after);

                    unsigned not_done_mask = __ballot_sync(mask, !done);
                    if(not_done_mask == 0){
                        if(lane == warp_leader) {
                            if(is_gpu_cache)
                                gpu_hit_cnt.fetch_add(1, simt::memory_order_relaxed);
                            else
                                cpu_hit_cnt.fetch_add(1, simt::memory_order_relaxed);
                        }
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
        bool to_cpu = (way >= gpu_num_ways) ? true : false;
        if(is_simulation){
            void* src ;
            if(way >= gpu_num_ways)
                src = ((void*)cpu_base_addr_)+ (cpu_set_offset + way - gpu_num_ways) * CL_SIZE;                   
            else{
                src = ((void*)base_addr_)+ (gpu_set_offset + way) * CL_SIZE;
            }
            read_page_simulation(cl_id, src, mask);
        }
        else{
            if(lane == warp_leader) {
                queue = get_smid() % (d_ctrls_[0]->n_qps);
                
                if(to_cpu){
                    read_page(cl_id, cpu_set_offset + way - gpu_num_ways, queue, to_cpu);
                }
                else
                    read_page(cl_id, gpu_set_offset + way, queue, to_cpu);
            }
        }

        __syncwarp(mask);

        if(lane == warp_leader) {
            (cur_cl_seqlock + way)->write_unbusy();
        }
        __syncwarp(mask);

         void* src ;
        //CPU memory
        if(way >= gpu_num_ways)
            src = ((void*)cpu_base_addr_)+ (cpu_set_offset + way - gpu_num_ways) * CL_SIZE;                   
        else{
            src = ((void*)base_addr_)+ (gpu_set_offset + way) * CL_SIZE;
        }
        if(use_nvshmem){
            nvshmemx_float_put_warp(output_ptr, (float*) src, CL_SIZE/sizeof(float), dst_gpu);
        }
        else{
            warp_memcpy<float>(src, output_ptr, CL_SIZE, mask);
        }

        
        if(lane == warp_leader) {
            if(to_cpu)
                cpu_miss_cnt.fetch_add(1, simt::memory_order_relaxed);
            else
                gpu_miss_cnt.fetch_add(1, simt::memory_order_relaxed);
            // if(evict_policy == 1 ){
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

};



template<typename T>
struct NVSHMEM_cache_handle{
    NVSHMEM_cache_d_t<T>* cache_ptr = nullptr;

    seqlock* d_set_locks;
    seqlock* d_way_locks;

    uint64_t num_sets_;
    uint64_t num_ways_;
    uint64_t cpu_num_ways_;
    uint64_t gpu_num_ways_;

    uint64_t CL_SIZE_;


    uint64_t* keys_;
    uint32_t* set_cnt_;



    Controller* ctrls_;

    DmaPtr pages_dma;
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


    unsigned int my_GPU_id_;
    unsigned num_gpus = 1;
    
    DmaPtr cpu_pages_dma;
    uint8_t* cpu_base_addr;

    BufferPtr cpu_prp1_buf;
    BufferPtr cpu_prp2_buf; 

    bool is_simulation = false;
    void* sim_buf;

    __host__ 
    NVSHMEM_cache_handle(uint64_t num_sets, uint64_t  gpu_num_ways, uint64_t cl_size,/* const Controller& ctrl,*/ const std::vector<Controller*>& ctrls,  const uint32_t cudaDevice,
                         unsigned int my_GPU_id, unsigned n_gpus,  uint64_t cpu_num_ways, bool simulation, void* sim_buf_ptr) :
    num_sets_(num_sets),
    gpu_num_ways_(gpu_num_ways),
    cpu_num_ways_(cpu_num_ways),
    CL_SIZE_(cl_size),
    my_GPU_id_(my_GPU_id),
    num_gpus(n_gpus),
    is_simulation(simulation),
    sim_buf(sim_buf_ptr)
    {
       // auto ctrl = ctrls[0][0];
        bool use_cpu_cache = false;
        if(cpu_num_ways > 0){
            use_cpu_cache = true;
        }
        
        num_ways_ = gpu_num_ways_ + cpu_num_ways_;
        cudaSetDevice(cudaDevice);

        uint64_t* prp1;
        uint64_t* prp2;

        uint64_t* cpu_prp1;
        uint64_t* cpu_prp2;

        printf("num sets: %llu num ways: %llu  gpu_ways: %llu cpu_ways: %llu  num_gpus in NVLink domain:%u \n", num_sets, num_ways_, gpu_num_ways, cpu_num_ways, num_gpus);

        assert(num_ways_ != 0 && "Error: Variable 'ways' should not be zero.");

        cuda_err_chk(cudaMalloc((void**)&d_set_locks, sizeof(seqlock) * num_sets_));
        cuda_err_chk(cudaMalloc((void**)&d_way_locks, sizeof(seqlock) * num_sets_ * (num_ways_ )));

        cuda_err_chk(cudaMalloc((void**)&set_cnt_, sizeof(uint32_t) * num_sets_));
        cuda_err_chk(cudaMalloc((void**)&keys_, sizeof(uint64_t) * num_sets_ * (num_ways_ )));

        cuda_err_chk(cudaMemset(d_set_locks, 0, sizeof(seqlock) * num_sets_));
        cuda_err_chk(cudaMemset(d_way_locks, 0, sizeof(seqlock) * num_sets_ * (num_ways_ )));
        cuda_err_chk(cudaMemset(set_cnt_, 0, sizeof(uint32_t) * num_sets_));
        cuda_err_chk(cudaMemset(keys_, 0xFF, sizeof(uint64_t) * num_sets_ * (num_ways_ )));

        cuda_err_chk(cudaMalloc((void**)&cache_ptr, sizeof(NVSHMEM_cache_d_t<T>)));


       

        uint64_t GPU_cache_size = CL_SIZE_ * num_sets_ * gpu_num_ways_;
        uint64_t CPU_cache_size = CL_SIZE_ * num_sets_ * cpu_num_ways_;

        if(is_simulation){
            cuda_err_chk(cudaMalloc((void**)&base_addr, GPU_cache_size));
            cuda_err_chk(cudaHostAlloc((void**)&cpu_base_addr, CPU_cache_size, cudaHostAllocDefault));
        }
        else{
            const Controller& ctrl = ctrls[0][0]; 

            if(gpu_num_ways_ > 0){
                printf("GPU DMA setup\n");
                pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(GPU_cache_size, 1UL << 16), cudaDevice);
                base_addr = (uint8_t*) pages_dma.get()->vaddr;
            }
        // const uint32_t uints_per_page = ctrl.ctrl->page_size / sizeof(uint64_t);

            if(use_cpu_cache){
                printf("CPU DMA setup\n");
                cpu_pages_dma = createDmaHost(ctrl.ctrl, NVM_PAGE_ALIGN(CPU_cache_size, 1UL << 16));
                cpu_base_addr = (uint8_t*) cpu_pages_dma.get()->vaddr;
            }
        }
        printf("DMA setup done\n");

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
        uint64_t n_blocks_per_page = 0;
        
        if(simulation == false){
            const Controller& ctrl = ctrls[0][0]; 

            
            n_blocks_per_page = (CL_SIZE_/ctrl.blk_size);

            for (size_t k = 0; k < n_ctrls; k++)
                cuda_err_chk(cudaMemcpy(d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));
            
            uint64_t how_many_in_one = ctrl.ctrl->page_size/CL_SIZE_;

            if(gpu_num_ways_ > 0){
                if (CL_SIZE_ <= pages_dma.get()->page_size) {
                    std::cout << "Setting GPU cache\n";
                    this->prp1_buf = createBuffer(num_sets_ * gpu_num_ways_ * sizeof(uint64_t), cudaDevice);
                    prp1 = (uint64_t*) this->prp1_buf.get();

                    std::cout << (num_sets_ * gpu_num_ways_) << " " << sizeof(uint64_t) << " " << how_many_in_one << " " << this->pages_dma.get()->n_ioaddrs <<std::endl;
                    uint64_t* temp = new uint64_t[how_many_in_one * pages_dma.get()->n_ioaddrs];
                    std::memset(temp, 0, how_many_in_one *  pages_dma.get()->n_ioaddrs);

                    if (temp == NULL)
                        std::cout << "NULL\n";

                    for (size_t i = 0; (i < this->pages_dma.get()->n_ioaddrs) ; i++) {
                        for (size_t j = 0; (j < how_many_in_one); j++) {
                            temp[i*how_many_in_one + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*CL_SIZE_;
                        }
                    }
                    cuda_err_chk(cudaMemcpy(prp1, temp, num_sets_ * gpu_num_ways_ * sizeof(uint64_t), cudaMemcpyHostToDevice));

                    delete temp;
                }
            }

            if(cpu_num_ways_ > 0){
                if (CL_SIZE_ <= cpu_pages_dma.get()->page_size) {
                    std::cout << "Setting CPU cache\n";
                    this->cpu_prp1_buf = createBuffer(num_sets_ * cpu_num_ways_ * sizeof(uint64_t));
                    cpu_prp1 = (uint64_t*) this->cpu_prp1_buf.get();

                    std::cout << (num_sets_ * cpu_num_ways_) << " " << sizeof(uint64_t) << " " << how_many_in_one << " " << this->cpu_pages_dma.get()->n_ioaddrs <<std::endl;
                    uint64_t* cpu_temp = new uint64_t[how_many_in_one * cpu_pages_dma.get()->n_ioaddrs];
                    std::memset(cpu_temp, 0, how_many_in_one *  cpu_pages_dma.get()->n_ioaddrs);

                    if (cpu_temp == NULL)
                        std::cout << "NULL\n";

                    for (size_t i = 0; (i < this->cpu_pages_dma.get()->n_ioaddrs) ; i++) {
                        for (size_t j = 0; (j < how_many_in_one); j++) {
                            cpu_temp[i*how_many_in_one + j] = ((uint64_t)this->cpu_pages_dma.get()->ioaddrs[i]) + j*CL_SIZE_;
                        }
                    }
                    cuda_err_chk(cudaMemcpy(cpu_prp1, cpu_temp, num_sets_ * cpu_num_ways_ * sizeof(uint64_t), cudaMemcpyHostToDevice));

                    delete cpu_temp;

                }
                
            }
        }
        prps = false;

        NVSHMEM_cache_d_t<T> cache_host(d_set_locks, d_way_locks, num_sets_, gpu_num_ways_, CL_SIZE_, keys_, set_cnt_, d_ctrls, n_ctrls, n_blocks_per_page, base_addr, prp1, prp2, prps,
        cache_q_head, cache_q_tail, cache_q_lock, cache_extra_reads, 
        my_GPU_id_, num_gpus, cpu_num_ways_, cpu_base_addr, cpu_prp1, is_simulation, sim_buf);


        cuda_err_chk(cudaMemcpy(cache_ptr, &cache_host, sizeof(NVSHMEM_cache_d_t<T>), cudaMemcpyHostToDevice));
        std::cout << "Cache Device Setting Done\n";
    }

    __host__ 
    NVSHMEM_cache_d_t<T>* 
    get_ptr(){
        return cache_ptr;
    }



};



#endif // __SET_ASSOCIATIVE_PAGE_CACHE_H__
